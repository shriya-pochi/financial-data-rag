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

from flask import Flask, request, jsonify, session, render_template_string

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

    if not context.strip():
        answer = (
            "There's nothing in the knowledge base yet to answer that from. "
            "Try uploading a sales report, deck, or spreadsheet."
        )
    elif needs_clarification(message, context):
        answer = (
            "I need a bit more to go on — could you name the file, sheet, "
            "metric, or time period you mean?"
        )
    else:
        prompt = build_prompt(message, context, history)
        try:
            resp = get_client().models.generate_content(model=GEMINI_MODEL, contents=prompt)
            answer = resp.text
        except Exception as e:
            return jsonify({"error": f"Model call failed: {e}"}), 502

    history.append({"user": message, "assistant": answer})
    return jsonify({"answer": answer})


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
    return jsonify({"status": "ok"})


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
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root{
    --ink:#0E1421;
    --panel:#161D2E;
    --panel-2:#1E2740;
    --brass:#C9A227;
    --teal:#3FA796;
    --text:#EDEFF4;
    --muted:#8B93A7;
    --line:#2B3550;
    --radius:14px;
  }
  *{box-sizing:border-box;}
  html,body{height:100%;}
  body{
    margin:0;
    background:
      radial-gradient(1200px 600px at 15% -10%, rgba(201,162,39,0.08), transparent 60%),
      radial-gradient(900px 500px at 100% 0%, rgba(63,167,150,0.07), transparent 55%),
      var(--ink);
    color:var(--text);
    font-family:'Inter',sans-serif;
    display:flex;
    flex-direction:column;
    min-height:100vh;
  }
  header{
    padding:28px clamp(16px,4vw,48px) 10px;
    display:flex;
    align-items:baseline;
    justify-content:space-between;
    border-bottom:1px solid var(--line);
  }
  .brand{display:flex;align-items:baseline;gap:10px;}
  .brand .mark{
    font-family:'IBM Plex Mono',monospace;
    color:var(--brass);
    font-size:13px;
    letter-spacing:.14em;
    border:1px solid var(--brass);
    border-radius:4px;
    padding:2px 6px;
  }
  h1{
    font-family:'Fraunces',serif;
    font-weight:600;
    font-size:clamp(20px,2.6vw,28px);
    margin:0;
    letter-spacing:.01em;
  }
  .tag{
    font-family:'IBM Plex Mono',monospace;
    color:var(--muted);
    font-size:12px;
  }
  main{
    flex:1;
    width:100%;
    max-width:860px;
    margin:0 auto;
    padding:24px clamp(12px,4vw,24px) 140px;
    display:flex;
    flex-direction:column;
    gap:18px;
  }
  .empty{
    margin-top:10vh;
    text-align:center;
    color:var(--muted);
  }
  .empty .big{
    font-family:'Fraunces',serif;
    font-size:clamp(22px,3.4vw,34px);
    color:var(--text);
    margin-bottom:10px;
  }
  .msg{
    display:flex;
    flex-direction:column;
    gap:8px;
    max-width:82%;
    animation:rise .35s ease both;
  }
  @keyframes rise{from{opacity:0;transform:translateY(6px);}to{opacity:1;transform:translateY(0);}}
  .msg.user{align-self:flex-end;align-items:flex-end;}
  .msg.bot{align-self:flex-start;align-items:flex-start;}
  .bubble{
    padding:13px 16px;
    border-radius:var(--radius);
    line-height:1.5;
    font-size:15px;
    white-space:pre-wrap;
  }
  .msg.user .bubble{
    background:var(--panel-2);
    border:1px solid var(--line);
    border-bottom-right-radius:4px;
  }
  .msg.bot .bubble{
    background:linear-gradient(180deg,var(--panel),var(--panel-2));
    border:1px solid var(--line);
    border-bottom-left-radius:4px;
  }
  /* signature element: the ledger receipt strip under bot answers,
     torn/dashed like a paper tape printout of the sources cited */
  .receipt{
    font-family:'IBM Plex Mono',monospace;
    font-size:11.5px;
    color:var(--teal);
    background:rgba(63,167,150,0.06);
    border:1px dashed rgba(63,167,150,0.45);
    border-radius:8px;
    padding:8px 12px;
    max-width:100%;
  }
  .receipt .lbl{
    color:var(--brass);
    letter-spacing:.12em;
    margin-right:6px;
  }
  .typing{
    display:flex;gap:4px;padding:14px 16px;
  }
  .typing span{
    width:6px;height:6px;border-radius:50%;
    background:var(--muted);
    animation:blink 1.2s infinite ease-in-out;
  }
  .typing span:nth-child(2){animation-delay:.15s;}
  .typing span:nth-child(3){animation-delay:.3s;}
  @keyframes blink{0%,80%,100%{opacity:.25;}40%{opacity:1;}}

  footer{
    position:fixed;bottom:0;left:0;right:0;
    background:linear-gradient(180deg, transparent, var(--ink) 30%);
    padding:18px clamp(12px,4vw,24px) 22px;
  }
  .composer{
    max-width:860px;margin:0 auto;
    background:var(--panel);
    border:1px solid var(--line);
    border-radius:16px;
    padding:8px 8px 8px 16px;
    display:flex;
    align-items:center;
    gap:8px;
    box-shadow:0 12px 30px rgba(0,0,0,0.35);
  }
  .composer input[type=text]{
    flex:1;
    background:transparent;
    border:none;
    outline:none;
    color:var(--text);
    font-family:'Inter',sans-serif;
    font-size:15px;
    padding:10px 0;
  }
  .composer input[type=text]::placeholder{color:var(--muted);}
  .icon-btn{
    width:38px;height:38px;
    border-radius:10px;
    border:1px solid var(--line);
    background:var(--panel-2);
    color:var(--muted);
    display:flex;align-items:center;justify-content:center;
    cursor:pointer;
    transition:.15s;
    flex-shrink:0;
  }
  .icon-btn:hover{border-color:var(--brass);color:var(--brass);}
  .send-btn{
    background:var(--brass);
    color:#1a1405;
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
  .send-btn:hover{filter:brightness(1.08);}
  .send-btn:disabled{opacity:.5;cursor:default;}
  #file-input{display:none;}
  .upload-note{
    max-width:860px;margin:8px auto 0;
    font-family:'IBM Plex Mono',monospace;
    font-size:11px;color:var(--muted);
    text-align:center;
  }
  .toast{
    position:fixed;top:18px;left:50%;transform:translateX(-50%);
    background:var(--panel-2);border:1px solid var(--line);
    padding:10px 16px;border-radius:10px;font-size:13px;
    color:var(--text);opacity:0;transition:opacity .25s;
    z-index:10;pointer-events:none;
  }
  .toast.show{opacity:1;}
  @media (prefers-reduced-motion: reduce){
    .msg,.typing span{animation:none;}
  }
  .tabs{display:flex;gap:6px;}
  .tab{
    font-family:'IBM Plex Mono',monospace;
    font-size:12px;
    letter-spacing:.06em;
    background:transparent;
    border:1px solid var(--line);
    color:var(--muted);
    padding:7px 14px;
    border-radius:8px;
    cursor:pointer;
    transition:.15s;
  }
  .tab:hover{color:var(--text);}
  .tab.active{
    color:var(--brass);
    border-color:var(--brass);
    background:rgba(201,162,39,0.08);
  }
  #files-view{display:none;}
  .file-section{margin-bottom:28px;}
  .file-section h2{
    font-family:'Fraunces',serif;
    font-weight:600;
    font-size:16px;
    margin:0 0 4px;
  }
  .file-section .sub{
    font-family:'IBM Plex Mono',monospace;
    font-size:11.5px;
    color:var(--muted);
    margin-bottom:12px;
  }
  .file-row{
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:12px;
    padding:12px 14px;
    background:var(--panel);
    border:1px solid var(--line);
    border-radius:10px;
    margin-bottom:8px;
  }
  .file-row .name{font-size:14px;}
  .file-row .meta{
    font-family:'IBM Plex Mono',monospace;
    font-size:11px;
    color:var(--teal);
    white-space:nowrap;
  }
  .file-empty{
    color:var(--muted);
    font-size:13px;
    padding:20px 4px;
  }
</style>
</head>
<body>

<header>
  <div class="brand">
    <span class="mark">RD</span>
    <div>
      <h1>Revenue Desk</h1>
      <div class="tag">sales &amp; revenue Q&amp;A, answers cited to the row</div>
    </div>
  </div>
  <div class="tabs">
    <button class="tab active" id="tab-chat" data-view="chat">Chat</button>
    <button class="tab" id="tab-files" data-view="files">Files</button>
  </div>
</header>

<main>
<div id="chat-view">
<div id="thread">
  <div class="empty" id="empty-state">
    <div class="big">Ask it about the numbers.</div>
    <div>Try &ldquo;what was Q3 revenue by region?&rdquo; or upload a report below.</div>
  </div>
</div>
</div>

<div id="files-view">
  <div class="file-section">
    <h2>Knowledge base</h2>
    <div class="sub">Permanently ingested, shared with every visitor</div>
    <div id="library-files"><div class="file-empty">Loading…</div></div>
  </div>
  <div class="file-section">
    <h2>This session</h2>
    <div class="sub">Only visible to you, cleared when your session ends</div>
    <div id="session-files"><div class="file-empty">Loading…</div></div>
  </div>
</div>
</main>

<div class="toast" id="toast"></div>

<footer>
  <div class="composer">
    <div class="icon-btn" id="upload-btn" title="Upload a PDF, deck, or spreadsheet">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12"/><path d="m7 8 5-5 5 5"/><path d="M5 21h14"/></svg>
    </div>
    <input type="file" id="file-input" accept=".pdf,.ppt,.pptx,.xlsx,.xls,.csv">
    <input type="text" id="msg-input" placeholder="Ask about revenue, pipeline, quota…" autocomplete="off">
    <button class="send-btn" id="send-btn">Ask</button>
  </div>
  <div class="upload-note">Uploads are private to this session and never saved permanently.</div>
</footer>

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
  window.scrollTo({top: document.body.scrollHeight, behavior:'smooth'});
  return wrap;
}

function addTyping(){
  const wrap = document.createElement('div');
  wrap.className = 'msg bot';
  wrap.id = 'typing-indicator';
  wrap.innerHTML = '<div class="bubble typing"><span></span><span></span><span></span></div>';
  thread.appendChild(wrap);
  window.scrollTo({top: document.body.scrollHeight, behavior:'smooth'});
}

function removeTyping(){
  const el = document.getElementById('typing-indicator');
  if (el) el.remove();
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
  addTyping();

  try{
    const res = await fetch('/api/chat', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({message: text})
    });
    const data = await res.json();
    removeTyping();
    if (!res.ok){
      addMessage('bot', data.error || 'Something went wrong.');
    } else {
      const {body, sources} = splitSources(data.answer || '');
      addMessage('bot', body, sources);
    }
  }catch(e){
    removeTyping();
    addMessage('bot', 'Network error — try again.');
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
      if (document.getElementById('tab-files').classList.contains('active')) loadFiles();
    }
  }catch(e){
    showToast('Upload failed — could not reach the server.');
  }
  fileInput.value = '';
});

// ---- Files tab ----
const tabChat = document.getElementById('tab-chat');
const tabFiles = document.getElementById('tab-files');
const chatView = document.getElementById('chat-view');
const filesView = document.getElementById('files-view');
const footerEl = document.querySelector('footer');

function switchTab(view){
  const toFiles = view === 'files';
  tabFiles.classList.toggle('active', toFiles);
  tabChat.classList.toggle('active', !toFiles);
  filesView.style.display = toFiles ? 'block' : 'none';
  chatView.style.display = toFiles ? 'none' : 'block';
  footerEl.style.display = toFiles ? 'none' : 'block';
  if (toFiles) loadFiles();
}
tabChat.addEventListener('click', () => switchTab('chat'));
tabFiles.addEventListener('click', () => switchTab('files'));

function fmtTime(ts){
  if (!ts) return '';
  return new Date(ts * 1000).toLocaleString();
}

async function loadFiles(){
  const libEl = document.getElementById('library-files');
  const sessEl = document.getElementById('session-files');
  libEl.innerHTML = '<div class="file-empty">Loading…</div>';
  sessEl.innerHTML = '<div class="file-empty">Loading…</div>';
  try{
    const res = await fetch('/api/files');
    const data = await res.json();

    libEl.innerHTML = data.library && data.library.length
      ? data.library.map(name => `<div class="file-row"><span class="name">${escapeHtml(name)}</span></div>`).join('')
      : '<div class="file-empty">Nothing ingested yet — run ingest.py to build the knowledge base.</div>';

    sessEl.innerHTML = data.session_files && data.session_files.length
      ? data.session_files.map(f => `<div class="file-row"><span class="name">${escapeHtml(f.filename)}</span><span class="meta">${f.chunks != null ? f.chunks + ' chunks · ' : ''}${fmtTime(f.uploaded_at)}</span></div>`).join('')
      : '<div class="file-empty">You haven\'t uploaded anything this session.</div>';
  }catch(e){
    libEl.innerHTML = '<div class="file-empty">Couldn\'t load files.</div>';
    sessEl.innerHTML = '';
  }
}

function escapeHtml(s){
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    # Local dev only. For a publicly viewable deploy, run behind gunicorn:
    #   gunicorn -w 1 -b 0.0.0.0:8080 app:app
    # Keep -w 1 (one worker) unless you move rate limiting/session state
    # to something shared like Redis — separate workers don't share the
    # in-memory dicts above.
    app.run(host="0.0.0.0", port=8080, debug=False)