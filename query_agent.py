"""
query_agent.py — Sales & Revenue Q&A agent.

Retrieves relevant chunks/tables from a Chroma collection (built by
ingest.py) and asks Gemini, via Vertex AI, to answer with exact numbers
and cited sources (file, sheet, table).

Auth: intentionally NOT handled here beyond the two lines below. This
uses Application Default Credentials through Vertex AI — no API key, no
service-account JSON in this codebase. On your server, run:

    gcloud auth application-default login \
        --impersonate-service-account=YOUR_SA@sensel-project.iam.gserviceaccount.com

or set GOOGLE_APPLICATION_CREDENTIALS to a key file if you're not using
impersonation. Either way, the credentials never leave the server —
visitors hitting the chatbot never touch Google auth themselves.
"""

import os
import re
from google import genai

GCP_PROJECT = os.environ.get("GCP_PROJECT", "sensel-project")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
GEMINI_MODEL = "gemini-2.5-flash"

N_FILE_META = 3
N_EXCEL = 6
N_TEXT = 8
HISTORY_TURNS = 4

# Hard ceiling on how much retrieved context we'll ever send to the model,
# regardless of how much comes back from the vector store. ~4 chars/token
# is a safe rough estimate, so this stays well under Gemini's context
# window even for a huge single retrieved chunk.
MAX_CONTEXT_CHARS = 400_000

_client = None


def get_client():
    global _client
    if _client is None:
        _client = genai.Client(
            vertexai=True,
            project=GCP_PROJECT,
            location=GCP_LOCATION,
        )
    return _client


# =========================
# RETRIEVAL HELPERS
# =========================

def get_known_filenames(collection):
    try:
        res = collection.get(where={"type": "file_meta"}, include=["metadatas"])
        return list({m["filename"] for m in res.get("metadatas", []) if m.get("filename")})
    except Exception:
        return []


def extract_filename_hint(query, known_filenames):
    q_lower = query.lower()
    tokens = re.findall(r"[A-Za-z0-9_\-\.]+", q_lower)

    for fn in known_filenames:
        fn_noext = os.path.splitext(fn.lower())[0]
        if fn.lower() in tokens or fn_noext in tokens:
            return fn

    for fn in known_filenames:
        fn_noext = os.path.splitext(fn.lower())[0]
        for tok in tokens:
            if len(tok) > 2 and fn_noext.startswith(tok):
                return fn

    return None


def query_collection(collection, embedding, n, ftype=None, filename=None):
    kwargs = {"query_embeddings": embedding, "n_results": n}

    if ftype and filename:
        kwargs["where"] = {
            "$and": [
                {"type": ftype},
                {"$or": [{"filename": filename}, {"filename_noext": filename}]},
            ]
        }
    elif ftype:
        kwargs["where"] = {"type": ftype}
    elif filename:
        kwargs["where"] = {"$or": [{"filename": filename}, {"filename_noext": filename}]}

    try:
        # n_results can't exceed the collection size
        count = collection.count()
        if count == 0:
            return []
        kwargs["n_results"] = min(n, count)
        res = collection.query(**kwargs)
        return res.get("documents", [[]])[0]
    except Exception as e:
        print(f"  [warn] retrieval error: {e}")
        return []


def _dedupe(docs):
    seen, out = set(), []
    for d in docs:
        key = d[:200]
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def retrieve_context(query, collection, embedder):
    emb = embedder.encode([query]).tolist()
    known_filenames = get_known_filenames(collection)
    filename = extract_filename_hint(query, known_filenames)

    file_meta = query_collection(collection, emb, N_FILE_META, "file_meta", filename)
    excel_docs = query_collection(collection, emb, N_EXCEL, "excel_struct", filename)
    text_docs = query_collection(collection, emb, N_TEXT, "text", filename)

    if not file_meta and not excel_docs and not text_docs:
        file_meta = query_collection(collection, emb, N_FILE_META, "file_meta")
        excel_docs = query_collection(collection, emb, N_EXCEL, "excel_struct")
        text_docs = query_collection(collection, emb, N_TEXT, "text")

    combined = "\n\n---\n\n".join(_dedupe(file_meta + excel_docs + text_docs))
    return _truncate_context(combined)


def _truncate_context(context):
    if len(context) <= MAX_CONTEXT_CHARS:
        return context
    return (
        context[:MAX_CONTEXT_CHARS]
        + "\n\n[...context truncated — it exceeded the size the model can accept in one request...]"
    )


# =========================
# CLARIFICATION HEURISTIC
# =========================

VAGUE_TERMS = {"this", "that", "those", "it", "total", "value", "number", "which one"}


def needs_clarification(query, context):
    words = query.lower().split()
    too_short = len(words) < 3
    only_vague = bool(words) and all(w.strip("?.,!") in VAGUE_TERMS for w in words)
    no_context = not context.strip()
    return no_context and (too_short or only_vague)


# =========================
# PROMPT
# =========================

SYSTEM_PROMPT = """
You are a sales and revenue analyst answering questions about a company's
spreadsheets, decks, and reports.

Rules:
1. Use exact numbers from the context. Never estimate, round unnecessarily,
   or invent a figure that isn't present.
2. If a row/table already contains a "Total" or subtotal, use it directly —
   do not re-sum the underlying rows yourself.
3. Every number you give must be followed by its source, formatted as:
   Source: file=<file name>, sheet=<sheet name if applicable>, table=<table number if applicable>
4. If the user refers to "that", "it", "the previous table", etc., resolve
   it using the conversation history below.
5. If the context doesn't contain enough information to answer reliably,
   say so plainly and ask what would help (file name, sheet, time period,
   or metric) instead of guessing.
6. If asked for a breakdown, list every relevant line item you can find,
   not just the top one.
""".strip()


def _format_history(history):
    if not history:
        return "(none yet)"
    recent = history[-HISTORY_TURNS:]
    lines = []
    for turn in recent:
        lines.append(f"User: {turn.get('user', '')}")
        lines.append(f"Assistant: {turn.get('assistant', '')}")
    return "\n".join(lines)


def build_prompt(query, context, history):
    return f"""{SYSTEM_PROMPT}

Conversation history:
{_format_history(history)}

Context retrieved from the knowledge base:
{context or "(no matching content found)"}

Question:
{query}

Answer:"""


# =========================
# MAIN ENTRY POINT
# =========================

def answer_query(query, collection, embedder, history=None):
    """
    query: user's question (str)
    collection: a Chroma collection object (main or per-session)
    embedder: a loaded SentenceTransformer
    history: list of {"user": ..., "assistant": ...} dicts, most recent last
    Returns: answer text (str)
    """
    if collection.count() == 0:
        return (
            "There's nothing in the knowledge base yet. Upload a sales "
            "report, deck, or spreadsheet to get started."
        )

    context = retrieve_context(query, collection, embedder)

    if needs_clarification(query, context):
        return (
            "I need a bit more to go on. Could you specify the file name, "
            "sheet, metric, or time period you're asking about?"
        )

    prompt = build_prompt(query, context, history)

    try:
        response = get_client().models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        answer = response.text
    except Exception as e:
        return f"Sorry — the model call failed: {e}"

    if history is not None:
        history.append({"user": query, "assistant": answer})

    return answer


if __name__ == "__main__":
    import sys
    import chromadb
    from ingest import get_embedder, get_persistent_client, get_or_create_collection

    client = get_persistent_client()
    col = get_or_create_collection(client)
    embedder = get_embedder()

    if len(sys.argv) > 1:
        print(answer_query(" ".join(sys.argv[1:]), col, embedder))
    else:
        print("Sales Q&A (type 'exit' to quit)\n")
        hist = []
        while True:
            try:
                q = input("Question: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not q or q.lower() in {"exit", "quit"}:
                break
            print("\n" + answer_query(q, col, embedder, hist) + "\n")