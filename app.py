# cleaned_buddy_vanilla.py
# ──────────────────────────────────────────────────────────────────────────────
# A simplified, "vanilla RAG" version of BITS Buddy.
# Replaces the multi-stage thinking/critic/final pipeline with a single-pass RAG
# call: (retrieve → build prompt → call model → answer). Keeps Firebase auth,
# chat history, FAISS retriever, and lightweight caching.
# ──────────────────────────────────────────────────────────────────────────────

import os
import time
import hashlib
import json
import sqlite3
import fitz
import requests
from PIL import Image  # noqa: F401
import streamlit as st
from typing import List, Dict, Any, Optional

# LangChain/FAISS imports
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

import firebase_admin
from firebase_admin import credentials, auth, db

# ========== CONFIG ==========
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or "YOUR_API_KEY"
MODEL_MAIN = os.getenv("MODEL_MAIN") or "openai/gpt-oss-20b:free"
MODEL_FALLBACKS = [os.getenv("MODEL_FALLBACK") or "deepseek/deepseek-chat-v3-0324:free"]

EMBED_MODEL = os.getenv("EMBED_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
K_VAL = int(os.getenv("K_VAL") or 4)

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH") or "./llm_cache.db"
ENABLE_PERSISTENT_CACHE = True

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS_BASE = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}

# ----------------- Firebase Init -----------------
if not firebase_admin._apps:
    try:
        firebase_config = dict(st.secrets["firebase"])  # pragma: no cover
        firebase_config["private_key"] = firebase_config["private_key"].replace("\\n", "\n")
        database_url = st.secrets["firebase"]["database_url"]
        cred = credentials.Certificate(firebase_config)
        firebase_admin.initialize_app(cred, {"databaseURL": database_url})
    except Exception as e:  # pragma: no cover
        st.error(f"Firebase initialization failed: {e}")
        st.stop()
else:
    firebase_admin.get_app()

# Convenience reference (not strictly needed here but keeps parity)
realtime_db = db.reference('/')

# ----------------- Streamlit Page -----------------
st.set_page_config(page_title="BITS Buddy", layout="wide")
st.title("🎓 BITS Buddy")
st.markdown("Ask me anything about BITS Pilani")

# ----------------- Sidebar -----------------
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🔁 Start New Chat"):
        uid = st.session_state.get("user_uid")
        if uid:
            try:
                db.reference(f"user_chats/{uid}").delete()
            except Exception as e:
                st.warning(f"Failed to clear history: {e}")
        st.session_state.chat_history = []
        st.rerun()

    language = st.selectbox("🌐 Response Language", [
        "English", "Hindi", "Telugu", "Tamil", "Marathi", "Bengali"
    ])
    st.markdown("---")
    st.checkbox("Enable Persistent SQLite Cache", value=ENABLE_PERSISTENT_CACHE, key="enable_sqlite")

# ----------------- SQLite Cache -----------------
def init_sqlite(db_path: str = SQLITE_DB_PATH):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            model TEXT,
            messages_json TEXT,
            response TEXT,
            ts REAL
        )
        """
    )
    conn.commit()
    return conn

_sql_conn: Optional[sqlite3.Connection] = None
if ENABLE_PERSISTENT_CACHE:
    try:
        _sql_conn = init_sqlite(SQLITE_DB_PATH)
    except Exception as e:
        st.warning(f"Could not initialize SQLite cache: {e}")
        _sql_conn = None


def sql_get(key: str) -> Optional[str]:
    if not _sql_conn:
        return None
    cur = _sql_conn.execute("SELECT response FROM cache WHERE key=?", (key,))
    row = cur.fetchone()
    return row[0] if row else None


def sql_set(key: str, model: str, messages: List[Dict[str, str]], response: str):
    if not _sql_conn:
        return
    _sql_conn.execute(
        "INSERT OR REPLACE INTO cache (key, model, messages_json, response, ts) VALUES (?, ?, ?, ?, ?)",
        (key, model, json.dumps(messages, ensure_ascii=False), response, time.time()),
    )
    _sql_conn.commit()

# ----------------- In-memory Cache -----------------
if "prompt_cache" not in st.session_state:
    st.session_state.prompt_cache = {}

CACHE_MAX_ENTRIES = 4000


def _cache_set(key: str, value: str):
    pc = st.session_state.prompt_cache
    if len(pc) >= CACHE_MAX_ENTRIES:
        oldest = min(pc.items(), key=lambda kv: kv[1]["ts"])[0]
        pc.pop(oldest, None)
    pc[key] = {"response": value, "ts": time.time()}


def _cache_get(key: str) -> Optional[str]:
    v = st.session_state.prompt_cache.get(key)
    return v["response"] if v else None


def make_cache_key(model: str, messages: List[Dict[str, str]]):
    digest = hashlib.sha256()
    digest.update(model.encode("utf-8"))
    digest.update(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()

# ----------------- Vector DB (PDFs in current folder) -----------------
@st.cache_resource(show_spinner=False)
def load_vector_db(folder: str = "."):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)

    for file in os.listdir(folder):
        if file.lower().endswith(".pdf"):
            path = os.path.join(folder, file)
            try:
                with fitz.open(path) as doc:
                    full_text = "\n".join(page.get_text() for page in doc)
                for chunk in splitter.split_text(full_text):
                    docs.append(Document(page_content=chunk, metadata={"source": file}))
            except Exception as e:
                st.warning(f"Could not read {file}: {e}")

    if not docs:
        class EmptyRetriever:
            def get_relevant_documents(self, q):
                return []
        return EmptyRetriever()

    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = FAISS.from_documents(docs, embedder)
    return vectordb.as_retriever(search_type="similarity", k=K_VAL)

retriever = load_vector_db()

# ----------------- OpenRouter (single-pass) -----------------

def call_openrouter(model: str, messages: List[Dict[str, str]], timeout: int = 30) -> str:
    key = make_cache_key(model, messages)
    cached = _cache_get(key)
    if cached:
        return cached
    if st.session_state.get("enable_sqlite", ENABLE_PERSISTENT_CACHE) and _sql_conn:
        csql = sql_get(key)
        if csql:
            _cache_set(key, csql)
            return csql

    payload = {"model": model, "messages": messages}
    try:
        r = requests.post(OPENROUTER_URL, headers=HEADERS_BASE, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        content = None
        if isinstance(data.get("choices"), list) and data["choices"]:
            msg = data["choices"][0].get("message") or {}
            content = msg.get("content")
        if not content:
            content = data.get("text") or json.dumps(data)
        _cache_set(key, content)
        if st.session_state.get("enable_sqlite", ENABLE_PERSISTENT_CACHE) and _sql_conn:
            sql_set(key, model, messages, content)
        return content
    except Exception as e:
        raise RuntimeError(f"OpenRouter call failed: {e}")


def call_with_fallbacks(messages: List[Dict[str, str]]) -> str:
    last_err = None
    for m in [MODEL_MAIN] + MODEL_FALLBACKS:
        try:
            return call_openrouter(m, messages)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"All models failed. Last error: {last_err}")

# ----------------- Vanilla RAG Prompting -----------------

def build_vanilla_prompt(question: str, context: str, lang: str) -> List[Dict[str, str]]:
    """Single, compact instruction + question + context. No chain-of-thought, no critique stage."""
    sys = (
        f"You are BitsBuddy, a helpful BITS Pilani senior. Answer strictly in {lang}. "
        "If the context is insufficient, say so briefly and suggest where to find it on campus/official sites. "
        "Keep replies concise, add bullet points for steps, and cite inline which PDF/source chunk you used if applicable (e.g., [source: filename.pdf])."
    )
    user = (
        f"Question:\n{question}\n\n"
        f"Context (top-{K_VAL} chunks):\n{context[:4000]}"
    )
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]


def vanilla_rag_answer(query: str, lang: str = "English") -> Dict[str, Any]:
    # 1) Retrieve
    try:
        docs = retriever.get_relevant_documents(query)
    except Exception as e:
        docs = []
        st.warning(f"Retriever failed: {e}")

    # 2) Build context text with simple in-line source tags
    context_parts = []
    for d in docs:
        src = d.metadata.get("source", "")
        context_parts.append(f"[{src}]\n{d.page_content}")
    context_text = "\n\n".join(context_parts) if context_parts else ""

    # 3) Build prompt → single call
    messages = build_vanilla_prompt(query, context_text, lang)
    answer = call_with_fallbacks(messages)

    # 4) Return both answer and the sources we used
    sources = sorted({d.metadata.get("source", "unknown") for d in docs if d})
    return {
        "answer": answer,
        "sources": sources,
        "chunks": [d.page_content for d in docs],
    }

# ----------------- Firebase chat history helpers -----------------

def load_user_chat_history(uid: str) -> List[Dict[str, Any]]:
    try:
        snap = db.reference(f"user_chats/{uid}").get()
        if not snap:
            return []
        chat = snap.get("chat")
        return chat if isinstance(chat, list) else []
    except Exception as e:  # pragma: no cover
        st.error(f"Failed to load chat history for UID {uid}: {e}")
        return []


def save_user_chat_history(uid: str, chat: List[Dict[str, Any]]) -> bool:
    try:
        db.reference(f"user_chats/{uid}").set({"chat": chat})
        return True
    except Exception as e:  # pragma: no cover
        st.error(f"Failed to save chat history for UID {uid}: {e}")
        return False

# ----------------- Session / Auth -----------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    # Minimal login screen
    st.markdown("---")
    st.title("🔐 BITS Buddy Login")
    st.markdown("Please log in to continue")
    name = st.text_input("Full Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login / Sign Up"):
        if not name or not email or not password:
            st.error("Please fill in all fields.")
        else:
            try:
                email_norm = email.strip().lower()
                try:
                    user = auth.get_user_by_email(email_norm)
                    st.success(f"Welcome back, {user.display_name or name}!")
                except auth.UserNotFoundError:
                    user = auth.create_user(email=email_norm, password=password, display_name=name)
                    st.success(f"Account created! Welcome, {name}!")
                # Persist session
                st.session_state["user_uid"] = user.uid
                st.session_state["user_name"] = name
                st.session_state["authenticated"] = True
                # Load history
                st.session_state["chat_history"] = load_user_chat_history(user.uid)
                st.rerun()
            except Exception as e:
                st.error(f"Authentication failed: {e}")
    st.stop()

# If authenticated, ensure chat history is present
if "chat_history" not in st.session_state:
    uid = st.session_state.get("user_uid")
    st.session_state["chat_history"] = load_user_chat_history(uid) if uid else []

# ----------------- Main Chat UI -----------------
st.title(f"Welcome {st.session_state.get('user_name', 'User')} 👋")

user_query = st.chat_input("Ask me about BITS Pilani anything")
if user_query:
    query = user_query.strip()
    if not query:
        st.warning("Please type a question.")
    else:
        # Show the user message and store in history
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Vanilla RAG: single pass
        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                result = vanilla_rag_answer(query, lang=language)
                final_answer = result.get("answer", "Sorry, I couldn't generate an answer.")

                # (Optional tiny typewriter effect)
                buf = ""
                for ch in final_answer:
                    buf += ch
                    placeholder.markdown(buf + "|")
                    time.sleep(0.003)
                placeholder.markdown(buf)

                # Append assistant response
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": final_answer,
                    "sources": result.get("sources", []),
                })

                # Persist to Firebase
                uid = st.session_state.get("user_uid")
                if uid:
                    save_user_chat_history(uid, st.session_state.chat_history)

            except Exception as e:
                placeholder.markdown(f"❌ Error: {e}")
                st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {e}"})

# ----------------- Display prior (non-latest) messages -----------------
# We already rendered the latest pair above; re-render the history for continuity.
for msg in st.session_state.chat_history[:-0]:
    pass  # Intentionally skip re-render; Streamlit chat UI holds state.

# ----------------- Sidebar History Preview (pairs) -----------------
with st.sidebar:
    st.subheader("📂 Chat History")
    # Build Q/A pairs from linear history
    pairs = []
    pending_q = None
    for m in st.session_state.chat_history:
        if m.get("role") == "user":
            pending_q = m.get("content", "")
        elif m.get("role") == "assistant" and pending_q is not None:
            pairs.append((pending_q, m.get("content", "")))
            pending_q = None

    for i, (q, a) in enumerate(reversed(pairs)):
        st.markdown(f"**Q{i}:** {q}")
        st.markdown(f"**A{i}:** {a[:150]}...")
        st.markdown("---")

# ----------------- Footer -----------------
st.markdown(
    """
    <hr style="margin-top: 40px;">
    <div style='text-align: center; color: #888; font-size: 14px;'>
        Built with ❤️ by <b>BITS Pilani</b> · Pilani Campus ·
        <br>📬 Email: <a href="mailto:f20240347@pilani.bits-pilani.ac.in">Contact us</a>
    </div>
    """,
    unsafe_allow_html=True,
)
