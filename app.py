"""
app.py — Flask front door for the sales/revenue chatbot.

Serves a single-page chat UI and two API routes:

    POST /api/chat     { "message": str }        -> { "answer": str }
    POST /api/upload    multipart file            -> { "chunks": int }

Design choices, on purpose:
  - The Vertex AI service account credentials are only ever touched inside
    query_agent.py, server-side. Nothing here or in the shipped HTML/JS
    ever sees a key. Visitors don't need any Google auth of their own.
  - Every visitor gets a private, in-memory (ephemeral) Chroma collection
    tied to a signed session cookie. Anything they upload is answerable
    only in their own session and is never written into your permanent
    knowledge base (built separately via `python ingest.py <folder>`).
    Restarting the process clears all of it.
  - Since every request spends *your* GCP quota regardless of who's
    asking, both chat and upload are rate-limited per IP. The limiter
    here is in-memory, which is fine for a single-process deploy; if you
    ever run multiple gunicorn workers behind this, swap it for Redis
    (flask-limiter + redis storage) or the per-worker limits won't add up
    correctly.
"""

import os
import time
import uuid
import traceback
import threading
import tempfile
from collections import defaultdict, deque

from flask import Flask, request, jsonify, session, render_template_string, Response, stream_with_context

import chromadb

from ingest import get_embedder, get_persistent_client, get_or_create_collection, ingest_file, SUPPORTED_EXT
from query_agent import retrieve_context, needs_clarification, build_prompt, get_client, GEMINI_MODEL

# =========================
# CONFIG
# =========================

MAX_UPLOAD_MB = 15
CHAT_LIMIT_PER_HOUR = 30
UPLOAD_LIMIT_PER_HOUR = 8
SESSION_TTL_SECONDS = 2 * 60 * 60  # purge ephemeral upload data after 2h idle

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(32))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

embedder = get_embedder()
main_client = get_persistent_client()
main_collection = get_or_create_collection(main_client)

session_client = chromadb.EphemeralClient()
_session_lock = threading.Lock()
_session_state = {}  # sid -> {"history": [...], "last_seen": ts}


def get_session_id():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return session["sid"]


def get_session_collection(sid):
    with _session_lock:
        state = _session_state.setdefault(sid, {"history": [], "files": [], "last_seen": time.time()})
        state["last_seen"] = time.time()
    return session_client.get_or_create_collection(f"session_{sid}")


def _cleanup_loop():
    while True:
        time.sleep(600)
        cutoff = time.time() - SESSION_TTL_SECONDS
        with _session_lock:
            stale = [sid for sid, s in _session_state.items() if s["last_seen"] < cutoff]
            for sid in stale:
                try:
                    session_client.delete_collection(f"session_{sid}")
                except Exception:
                    pass
                _session_state.pop(sid, None)


threading.Thread(target=_cleanup_loop, daemon=True).start()


# =========================
# RATE LIMITING (in-memory, per-IP)
# =========================

_rate_buckets = defaultdict(deque)
_rate_lock = threading.Lock()


def rate_limited(key, limit_per_hour):
    now = time.time()
    with _rate_lock:
        bucket = _rate_buckets[key]
        while bucket and now - bucket[0] > 3600:
            bucket.popleft()
        if len(bucket) >= limit_per_hour:
            return True
        bucket.append(now)
        return False


def client_ip():
    return request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()


# =========================
# API ROUTES
# =========================

@app.route("/api/chat", methods=["POST"])
def api_chat():
    if rate_limited(f"chat:{client_ip()}", CHAT_LIMIT_PER_HOUR):
        return jsonify({"error": "Rate limit reached. Try again in a bit."}), 429

    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Empty message."}), 400
    if len(message) > 2000:
        return jsonify({"error": "Message too long."}), 400

    sid = get_session_id()
    sess_col = get_session_collection(sid)
    history = _session_state[sid]["history"]

    ctx_parts = []
    if sess_col.count() > 0:
        ctx_parts.append(retrieve_context(message, sess_col, embedder))
    if main_collection.count() > 0:
        ctx_parts.append(retrieve_context(message, main_collection, embedder))
    context = "\n\n---\n\n".join(p for p in ctx_parts if p)

    # These two short-circuit cases don't call the model at all, so there's
    # nothing to stream — send them as a single plain-text response.
    if not context.strip():
        answer = (
            "There's nothing in the knowledge base yet to answer that from. "
            "Try uploading a sales report, deck, or spreadsheet."
        )
        history.append({"user": message, "assistant": answer})
        return Response(answer, mimetype="text/plain")

    if needs_clarification(message, context):
        answer = (
            "I need a bit more to go on — could you name the file, sheet, "
            "metric, or time period you mean?"
        )
        history.append({"user": message, "assistant": answer})
        return Response(answer, mimetype="text/plain")

    prompt = build_prompt(message, context, history)

    def generate():
        # Streams the answer to the browser as Gemini generates it, piece
        # by piece, instead of buffering the whole thing server-side
        # first. History still gets the complete assembled text once the
        # stream ends — appended after the last yield, which Flask still
        # executes before closing the response.
        full_parts = []
        try:
            try:
                stream = get_client().models.generate_content_stream(
                    model=GEMINI_MODEL, contents=prompt
                )
            except AttributeError:
                # Older/different google-genai SDK version without
                # streaming support — fall back to one non-streamed chunk
                # rather than breaking entirely.
                resp = get_client().models.generate_content(model=GEMINI_MODEL, contents=prompt)
                full_parts.append(resp.text or "")
                yield resp.text or ""
                history.append({"user": message, "assistant": "".join(full_parts)})
                return

            for chunk in stream:
                piece = getattr(chunk, "text", None) or ""
                if piece:
                    full_parts.append(piece)
                    yield piece
        except Exception as e:
            err = f"\n\nModel call failed: {e}"
            full_parts.append(err)
            yield err

        history.append({"user": message, "assistant": "".join(full_parts)})

    return Response(stream_with_context(generate()), mimetype="text/plain")


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if rate_limited(f"upload:{client_ip()}", UPLOAD_LIMIT_PER_HOUR):
        return jsonify({"error": "Upload limit reached. Try again later."}), 429

    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "No file provided."}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in SUPPORTED_EXT:
        return jsonify({"error": f"Unsupported file type '{ext}'. Use PDF, PPTX, or XLSX."}), 400

    sid = get_session_id()
    tmp_path = None
    try:
        sess_col = get_session_collection(sid)

        # Get a path without holding our own handle open on it — on
        # Windows, NamedTemporaryFile keeps a handle, and f.save() trying
        # to open that same path again fails with WinError 32 ("used by
        # another process").
        fd, tmp_path = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        f.save(tmp_path)

        n = ingest_file(tmp_path, sess_col, embedder, project=f"session_{sid}", source_label=f.filename)

        with _session_lock:
            _session_state[sid].setdefault("files", []).append({
                "filename": f.filename,
                "chunks": n,
                "uploaded_at": time.time(),
            })

    except Exception as e:
        # Anything that goes wrong here — bad file, disk/permission issue,
        # a parser error deep in ingest.py — gets logged with a full
        # traceback server-side and reported back as real JSON, instead
        # of falling through to Flask's default HTML error page (which is
        # what shows up client-side as a generic "network error").
        traceback.print_exc()
        return jsonify({"error": f"Upload failed: {e}"}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if n == 0:
        return jsonify({"error": "No readable text or tables found in that file."}), 422

    return jsonify({"chunks": n, "filename": f.filename})


@app.route("/api/files")
def api_files():
    """Lists what's been ingested: the permanent knowledge base, plus
    whatever this visitor has uploaded in their own session."""
    def file_list(collection):
        try:
            res = collection.get(where={"type": "file_meta"}, include=["metadatas"])
            names = sorted({m["filename"] for m in res.get("metadatas", []) if m.get("filename")})
            return names
        except Exception:
            return []

    sid = get_session_id()
    sess_col = get_session_collection(sid)
    with _session_lock:
        session_uploads = list(_session_state.get(sid, {}).get("files", []))

    return jsonify({
        "library": file_list(main_collection),
        "session_files": session_uploads or [{"filename": n, "chunks": None, "uploaded_at": None}
                                              for n in file_list(sess_col)],
    })


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": f"File too large — max {MAX_UPLOAD_MB}MB."}), 413


@app.errorhandler(Exception)
def handle_uncaught(e):
    # Last-resort net so any route that forgets its own try/except still
    # returns JSON the frontend can parse, instead of an HTML error page.
    traceback.print_exc()
    return jsonify({"error": "Something went wrong on the server."}), 500


@app.route("/api/health")
def health():
    resp = jsonify({"status": "ok"})
    # Scoped to this one endpoint only, so a portfolio shell on another
    # subdomain can show a live/offline dot for this project without
    # opening CORS up anywhere sensitive (chat/upload stay same-origin).
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


# =========================
# UI
# =========================

@app.route("/")
def index():
    return render_template_string(PAGE_HTML)


PAGE_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Revenue Desk — Sales &amp; Revenue Q&amp;A</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root{
    --ink:#070A12;
    --panel:#0E1420;
    --panel-2:#151D2E;
    --line:#212A3D;
    --text:#E7ECF5;
    --muted:#828CA6;
    --blue:#2F7DFF;
    --blue-light:#6EB4FF;
    --blue-deep:#123B82;
    --blue-wash:rgba(47,125,255,0.08);
    --radius:14px;
  }
  *{box-sizing:border-box;}
  html,body{height:100%;}
  body{
    margin:0;
    background:
      radial-gradient(1000px 560px at 85% -10%, var(--blue-wash), transparent 55%),
      var(--ink);
    color:var(--text);
    font-family:'Inter',sans-serif;
  }

  .app-shell{
    display:grid;
    grid-template-columns:300px 1fr;
    height:100vh;
  }

  /* ===== sidebar ===== */
  aside{
    background:var(--panel);
    border-right:1px solid var(--line);
    display:flex;
    flex-direction:column;
    overflow-y:auto;
  }
  .brand{
    padding:26px 22px 20px;
    border-bottom:1px solid var(--line);
  }
  .brand .mark{
    font-family:'IBM Plex Mono',monospace;
    color:var(--blue-light);
    font-size:11px;
    letter-spacing:.16em;
    text-transform:uppercase;
  }
  .brand h1{
    font-family:'Source Serif 4',serif;
    font-weight:600;
    font-size:22px;
    margin:6px 0 4px;
    letter-spacing:.01em;
  }
  .brand .tag{
    font-size:12.5px;
    color:var(--muted);
    line-height:1.4;
  }

  .sb-section{padding:20px 22px;border-bottom:1px solid var(--line);}
  .sb-section h2{
    font-family:'IBM Plex Mono',monospace;
    font-size:11px;
    letter-spacing:.1em;
    text-transform:uppercase;
    color:var(--muted);
    margin:0 0 12px;
    display:flex;
    align-items:center;
    justify-content:space-between;
  }
  .sb-section h2 .count{
    color:var(--blue-light);
    background:var(--blue-wash);
    border-radius:6px;
    padding:1px 7px;
    font-size:11px;
  }
  .file-row{
    display:flex;
    flex-direction:column;
    gap:2px;
    padding:9px 10px;
    border-radius:9px;
    margin-bottom:6px;
    background:var(--panel-2);
    border:1px solid var(--line);
  }
  .file-row .name{font-size:13px;word-break:break-word;}
  .file-row .meta{
    font-family:'IBM Plex Mono',monospace;
    font-size:10.5px;
    color:var(--blue-light);
  }
  .file-empty{
    color:var(--muted);
    font-size:12.5px;
    line-height:1.5;
  }
  .sb-upload{padding:18px 22px;margin-top:auto;}
  .sb-upload-btn{
    width:100%;
    display:flex;
    align-items:center;
    justify-content:center;
    gap:8px;
    padding:11px;
    border-radius:10px;
    border:1px dashed var(--line);
    background:transparent;
    color:var(--muted);
    font-size:13px;
    cursor:pointer;
    transition:.15s;
  }
  .sb-upload-btn:hover{border-color:var(--blue);color:var(--blue-light);}
  .sb-note{
    font-family:'IBM Plex Mono',monospace;
    font-size:10.5px;
    color:var(--muted);
    text-align:center;
    margin-top:10px;
    line-height:1.5;
  }
  #file-input{display:none;}

  /* ===== main chat column ===== */
  .chat-main{
    display:flex;
    flex-direction:column;
    height:100vh;
    overflow:hidden;
  }
  .chat-header{
    padding:22px clamp(16px,3vw,40px) 16px;
    border-bottom:1px solid var(--line);
    display:flex;
    align-items:baseline;
    justify-content:space-between;
  }
  .chat-header .title{
    font-family:'Source Serif 4',serif;
    font-size:17px;
    font-weight:600;
  }
  .chat-header .status{
    font-family:'IBM Plex Mono',monospace;
    font-size:11px;
    color:var(--muted);
    display:flex;
    align-items:center;
    gap:6px;
  }
  .chat-header .status .dot{
    width:6px;height:6px;border-radius:50%;
    background:var(--blue);
    box-shadow:0 0 6px rgba(47,125,255,0.7);
  }

  #thread{
    flex:1;
    overflow-y:auto;
    padding:28px clamp(16px,3vw,40px);
    display:flex;
    flex-direction:column;
    gap:18px;
  }
  .empty{
    margin:auto;
    text-align:center;
    color:var(--muted);
    max-width:420px;
  }
  .empty .big{
    font-family:'Source Serif 4',serif;
    font-weight:600;
    font-size:clamp(22px,3vw,30px);
    color:var(--text);
    margin-bottom:10px;
  }
  .msg{
    display:flex;
    flex-direction:column;
    gap:8px;
    max-width:74%;
    animation:rise .3s ease both;
  }
  @keyframes rise{from{opacity:0;transform:translateY(6px);}to{opacity:1;transform:translateY(0);}}
  .msg.user{align-self:flex-end;align-items:flex-end;}
  .msg.bot{align-self:flex-start;align-items:flex-start;}
  .bubble{
    padding:13px 16px;
    border-radius:var(--radius);
    line-height:1.55;
    font-size:14.5px;
    white-space:pre-wrap;
  }
  .msg.user .bubble{
    background:var(--blue-deep);
    border:1px solid var(--blue);
    border-bottom-right-radius:4px;
  }
  .msg.bot .bubble{
    background:var(--panel-2);
    border:1px solid var(--line);
    border-bottom-left-radius:4px;
  }
  /* signature element: a slim "data trace" readout under single-fact
     answers — a technical citation line, not a decorative flourish */
  .receipt{
    font-family:'IBM Plex Mono',monospace;
    font-size:11.5px;
    color:var(--blue-light);
    background:var(--blue-wash);
    border:1px solid rgba(47,125,255,0.3);
    border-left:2px solid var(--blue);
    border-radius:0 8px 8px 0;
    padding:8px 12px;
    max-width:100%;
  }
  .receipt .lbl{
    color:var(--blue-light);
    letter-spacing:.12em;
    opacity:.75;
    margin-right:6px;
  }
  .bubble p{margin:0 0 10px;}
  .bubble p:last-child{margin-bottom:0;}
  .bubble ul,.bubble ol{margin:0 0 10px;padding-left:20px;}
  .bubble li{margin-bottom:4px;}
  .bubble strong{color:var(--text);font-weight:600;}
  .bubble h1,.bubble h2,.bubble h3,.bubble h4{
    font-family:'Source Serif 4',serif;
    font-weight:600;
    font-size:15px;
    margin:14px 0 6px;
  }
  .bubble h1:first-child,.bubble h2:first-child,.bubble h3:first-child{margin-top:0;}
  .bubble code{
    font-family:'IBM Plex Mono',monospace;
    font-size:12.5px;
    background:rgba(255,255,255,0.06);
    padding:1px 5px;
    border-radius:4px;
  }
  .typing{display:flex;gap:4px;padding:14px 16px;}
  .typing span{
    width:6px;height:6px;border-radius:50%;
    background:var(--muted);
    animation:blink 1.2s infinite ease-in-out;
  }
  .typing span:nth-child(2){animation-delay:.15s;}
  .typing span:nth-child(3){animation-delay:.3s;}
  @keyframes blink{0%,80%,100%{opacity:.25;}40%{opacity:1;}}

  .composer-wrap{
    padding:16px clamp(16px,3vw,40px) 22px;
    border-top:1px solid var(--line);
  }
  .composer{
    background:var(--panel-2);
    border:1px solid var(--line);
    border-radius:16px;
    padding:8px 8px 8px 16px;
    display:flex;
    align-items:center;
    gap:8px;
  }
  .composer input[type=text]{
    flex:1;
    background:transparent;
    border:none;
    outline:none;
    color:var(--text);
    font-family:'Inter',sans-serif;
    font-size:14.5px;
    padding:10px 0;
  }
  .composer input[type=text]::placeholder{color:var(--muted);}
  .send-btn{
    background:var(--blue);
    color:#04101F;
    border:none;
    font-weight:600;
    font-family:'Inter',sans-serif;
    font-size:14px;
    padding:10px 18px;
    border-radius:10px;
    cursor:pointer;
    flex-shrink:0;
    transition:.15s;
  }
  .send-btn:hover{filter:brightness(1.1);}
  .send-btn:disabled{opacity:.5;cursor:default;}

  .toast{
    position:fixed;top:18px;left:50%;transform:translateX(-50%);
    background:var(--panel-2);border:1px solid var(--blue);
    padding:10px 16px;border-radius:10px;font-size:13px;
    color:var(--text);opacity:0;transition:opacity .25s;
    z-index:10;pointer-events:none;
  }
  .toast.show{opacity:1;}
  @media (prefers-reduced-motion: reduce){
    .msg,.typing span{animation:none;}
  }
  @media (max-width:760px){
    .app-shell{grid-template-columns:1fr;}
    aside{
      position:fixed;inset:0 auto 0 0;width:280px;
      transform:translateX(-100%);
      transition:transform .2s;
      z-index:20;
    }
    aside.open{transform:translateX(0);}
  }
</style>
</head>
<body>

<div class="app-shell">

  <aside id="sidebar">
    <div class="brand">
      <div class="mark">Revenue Desk</div>
      <h1>Sales &amp; revenue intelligence</h1>
      <div class="tag">Answers cited to the row, sourced from your own reports.</div>
    </div>

    <div class="sb-section">
      <h2>Knowledge base <span class="count" id="library-count">—</span></h2>
      <div id="library-files"><div class="file-empty">Loading…</div></div>
    </div>

    <div class="sb-section" style="border-bottom:none;">
      <h2>This session <span class="count" id="session-count">—</span></h2>
      <div id="session-files"><div class="file-empty">Loading…</div></div>
    </div>

    <div class="sb-upload">
      <button class="sb-upload-btn" id="upload-btn">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12"/><path d="m7 8 5-5 5 5"/><path d="M5 21h14"/></svg>
        Upload a report
      </button>
      <input type="file" id="file-input" accept=".pdf,.ppt,.pptx,.xlsx,.xls,.csv">
      <div class="sb-note">Private to this session, never saved permanently.</div>
    </div>
  </aside>

  <div class="chat-main">
    <div class="chat-header">
      <div class="title">Ask about your numbers</div>
      <div class="status"><span class="dot"></span>Connected</div>
    </div>

    <div id="thread">
      <div class="empty" id="empty-state">
        <div class="big">Ask it about the numbers.</div>
        <div>Try &ldquo;what was Q3 revenue by region?&rdquo; or upload a report from the sidebar.</div>
      </div>
    </div>

    <div class="composer-wrap">
      <div class="composer">
        <input type="text" id="msg-input" placeholder="Ask about revenue, pipeline, quota…" autocomplete="off">
        <button class="send-btn" id="send-btn">Ask</button>
      </div>
    </div>
  </div>

</div>

<div class="toast" id="toast"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/dompurify/3.0.6/purify.min.js"></script>
<script>
const thread = document.getElementById('thread');
const emptyState = document.getElementById('empty-state');
const input = document.getElementById('msg-input');
const sendBtn = document.getElementById('send-btn');
const uploadBtn = document.getElementById('upload-btn');
const fileInput = document.getElementById('file-input');
const toast = document.getElementById('toast');

function showToast(text){
  toast.textContent = text;
  toast.classList.add('show');
  setTimeout(()=>toast.classList.remove('show'), 2600);
}

function escapeHtml(s){
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function addMessage(role, text, sources){
  emptyState.style.display = 'none';
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + role;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  wrap.appendChild(bubble);
  if (sources){
    const r = document.createElement('div');
    r.className = 'receipt';
    r.innerHTML = '<span class="lbl">SOURCE</span> ' + escapeHtml(sources);
    wrap.appendChild(r);
  }
  thread.appendChild(wrap);
  thread.scrollTop = thread.scrollHeight;
  return wrap;
}

function renderMarkdown(el, text){
  try{
    el.innerHTML = DOMPurify.sanitize(marked.parse(text));
  }catch(e){
    el.textContent = text;
  }
}

// Pull a trailing "Source: ..." line out of the answer text so it can
// render in the receipt strip — but only when there's exactly one
// citation in the whole answer. Multi-item breakdowns (revenue tables,
// expense lists) legitimately carry a "Source:" after every line, and
// trying to peel those into the receipt strip would swallow the entire
// answer into one box. Those stay inline, as written.
function splitSources(text){
  const citationCount = (text.match(/Source:/gi) || []).length;
  if (citationCount !== 1) return {body: text, sources: null};

  const match = text.match(/(Source:.*)$/is);
  if (!match) return {body: text, sources: null};
  const body = text.slice(0, match.index).trim();
  const sources = match[1].replace(/^Source:\s*/i, '').trim();
  return {body, sources};
}

async function sendMessage(){
  const text = input.value.trim();
  if (!text) return;
  addMessage('user', text);
  input.value = '';
  sendBtn.disabled = true;

  emptyState.style.display = 'none';
  const botWrap = document.createElement('div');
  botWrap.className = 'msg bot';
  const bubble = document.createElement('div');
  bubble.className = 'bubble typing';
  bubble.innerHTML = '<span></span><span></span><span></span>';
  botWrap.appendChild(bubble);
  thread.appendChild(botWrap);
  thread.scrollTop = thread.scrollHeight;

  let full = '';
  let firstChunk = true;

  try{
    const res = await fetch('/api/chat', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({message: text})
    });

    if (!res.ok){
      let data = {};
      try{ data = await res.json(); }catch(e){}
      bubble.className = 'bubble';
      bubble.textContent = data.error || 'Something went wrong.';
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    while (true){
      const {done, value} = await reader.read();
      if (done) break;
      const piece = decoder.decode(value, {stream: true});
      if (!piece) continue;
      if (firstChunk){
        bubble.className = 'bubble';
        bubble.textContent = '';
        firstChunk = false;
      }
      full += piece;
      bubble.textContent = full;   // plain text while streaming, typewriter-style
      thread.scrollTop = thread.scrollHeight;
    }

    // Once the full answer is in, swap to properly formatted markdown
    // and peel off a single trailing citation into the receipt strip.
    const {body, sources} = splitSources(full);
    renderMarkdown(bubble, body);
    if (sources){
      const r = document.createElement('div');
      r.className = 'receipt';
      r.innerHTML = '<span class="lbl">SOURCE</span> ' + escapeHtml(sources);
      botWrap.appendChild(r);
    }
    thread.scrollTop = thread.scrollHeight;
  }catch(e){
    bubble.className = 'bubble';
    bubble.textContent = 'Network error — try again.';
  }finally{
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener('click', sendMessage);
input.addEventListener('keydown', e => { if (e.key === 'Enter') sendMessage(); });

uploadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', async () => {
  const file = fileInput.files[0];
  if (!file) return;
  const form = new FormData();
  form.append('file', file);
  showToast('Uploading ' + file.name + '…');
  try{
    const res = await fetch('/api/upload', {method:'POST', body: form});
    let data;
    try{
      data = await res.json();
    }catch(parseErr){
      showToast('Upload failed — server returned an unexpected response (check the server console).');
      fileInput.value = '';
      return;
    }
    if (!res.ok){
      showToast(data.error || 'Upload failed.');
    } else {
      showToast(file.name + ' indexed — ask away.');
      loadFiles();
    }
  }catch(e){
    showToast('Upload failed — could not reach the server.');
  }
  fileInput.value = '';
});

function fmtTime(ts){
  if (!ts) return '';
  return new Date(ts * 1000).toLocaleString();
}

async function loadFiles(){
  const libEl = document.getElementById('library-files');
  const sessEl = document.getElementById('session-files');
  const libCount = document.getElementById('library-count');
  const sessCount = document.getElementById('session-count');
  try{
    const res = await fetch('/api/files');
    const data = await res.json();

    libCount.textContent = data.library ? data.library.length : 0;
    sessCount.textContent = data.session_files ? data.session_files.length : 0;

    libEl.innerHTML = data.library && data.library.length
      ? data.library.map(name => `<div class="file-row"><span class="name">${escapeHtml(name)}</span></div>`).join('')
      : '<div class="file-empty">Nothing ingested yet — run ingest.py to build the knowledge base.</div>';

    sessEl.innerHTML = data.session_files && data.session_files.length
      ? data.session_files.map(f => `<div class="file-row"><span class="name">${escapeHtml(f.filename)}</span><span class="meta">${f.chunks != null ? f.chunks + ' chunks · ' : ''}${fmtTime(f.uploaded_at)}</span></div>`).join('')
      : '<div class="file-empty">Nothing uploaded this session yet.</div>';
  }catch(e){
    libEl.innerHTML = '<div class="file-empty">Couldn\'t load files.</div>';
    sessEl.innerHTML = '';
  }
}

loadFiles();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    # Local dev only. For a publicly viewable deploy, run behind gunicorn:
    #   gunicorn -w 1 --threads 4 -b 0.0.0.0:8080 app:app
    # Keep -w 1 (one worker) unless you move rate limiting/session state
    # to something shared like Redis — separate workers don't share the
    # in-memory dicts above. threads>1 is what lets a slow streaming
    # response not block other visitors' requests.
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)