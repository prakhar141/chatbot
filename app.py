
import os
import time
import hashlib
import json
import sqlite3
import fitz
import requests
import logging
from math import ceil
from PIL import Image
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
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("bitsbuddy")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or "YOUR_API_KEY"
MODEL_CHEAP = os.getenv("MODEL_CHEAP") or "deepseek/deepseek-r1:free"
MODEL_MID = os.getenv("MODEL_MID") or "openai/gpt-oss-20b:free"
MODEL_HIGH = os.getenv("MODEL_HIGH") or "deepseek/deepseek-r1-0528:free"

EMBED_MODEL = os.getenv("EMBED_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
K_VAL = int(os.getenv("K_VAL") or 4)

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH") or "./llm_cache.db"
ENABLE_PERSISTENT_CACHE = True
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS") or 60 * 60 * 24)  # 1 day default
CACHE_MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES") or 4000)

FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR") or "./faiss_index"
PDF_DOCS_FOLDER = os.getenv("PDF_DOCS_FOLDER") or "."

# OpenRouter settings
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS_BASE = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}

# Per-process minimal spacing; keep but we also use per-session per-user limiter
MIN_DELAY = float(os.getenv("OPENROUTER_MIN_DELAY", "1.5"))

# ----------------- Firebase init -----------------
if not firebase_admin._apps:
    try:
        firebase_config = dict(st.secrets["firebase"])
        firebase_config["private_key"] = firebase_config["private_key"].replace("\\n", "\n")
        database_url = st.secrets["firebase"]["database_url"]
        cred = credentials.Certificate(firebase_config)
        firebase_admin.initialize_app(cred, {"databaseURL": database_url})
        logger.info("Firebase initialized.")
    except Exception as e:
        logger.exception("Firebase initialization failed.")
        st.error("Firebase initialization failed. Check server logs.")
        st.stop()
else:
    firebase_admin.get_app()

realtime_db = db.reference("/")

# ----------------- Streamlit page & sidebar -----------------
st.set_page_config(page_title="BITS Buddy", layout="wide")
col1, col2 = st.columns([1, 8])
with col1:
    # gracefully handle missing logo
    try:
        st.image("bits_logo.jpg", width=50)
    except Exception:
        pass
with col2:
    st.markdown("<h1 style='margin-top: 0;'>BITS Buddy</h1>", unsafe_allow_html=True)

st.markdown("Ask me anything about BITS Pilani")

with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🔁 Start New Chat"):
        uid = st.session_state.get("user_uid")
        if uid:
            try:
                ref = db.reference(f"user_chats/{uid}")
                ref.delete()
            except Exception as e:
                st.warning(f"Failed to clear history: {e}")
                logger.exception("Failed to delete chat history")
        st.session_state.chat_history = []
        st.session_state.just_streamed = False
        st.rerun()
    language = st.selectbox("🌐 Response Language", ["English", "Hindi", "Telugu", "Tamil", "Marathi", "Bengali"])
    st.markdown("---")
    st.checkbox("For Faster Loading", value=ENABLE_PERSISTENT_CACHE, key="enable_sqlite")

# ----------------- SQLITE CACHE (with TTL) -----------------
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
        logger.exception("Could not initialize SQLite cache")
        st.warning(f"Could not initialize SQLite cache: {e}")
        _sql_conn = None

def sql_get(key: str) -> Optional[Dict[str, Any]]:
    if not _sql_conn:
        return None
    cur = _sql_conn.execute("SELECT response, ts FROM cache WHERE key=?", (key,))
    row = cur.fetchone()
    if not row:
        return None
    response, ts = row
    if time.time() - ts > CACHE_TTL_SECONDS:
        # expired
        try:
            _sql_conn.execute("DELETE FROM cache WHERE key=?", (key,))
            _sql_conn.commit()
        except Exception:
            pass
        return None
    return {"response": response, "ts": ts}

def sql_set(key: str, model: str, messages: List[Dict[str, str]], response: str):
    if not _sql_conn:
        return
    try:
        _sql_conn.execute(
            "INSERT OR REPLACE INTO cache (key, model, messages_json, response, ts) VALUES (?, ?, ?, ?, ?)",
            (key, model, json.dumps(messages, ensure_ascii=False), response, time.time())
        )
        _sql_conn.commit()
    except Exception:
        logger.exception("Failed to write to sqlite cache")

# ----------------- in-memory cache (TTL + LRU-ish) -----------------
if "prompt_cache" not in st.session_state:
    st.session_state.prompt_cache = {}  # key -> {"response": str, "ts": float}

def _prune_in_memory_cache():
    cache = st.session_state.prompt_cache
    if len(cache) <= CACHE_MAX_ENTRIES:
        return
    # remove oldest entries until under limit
    items = sorted(cache.items(), key=lambda kv: kv[1]["ts"])
    to_remove = len(cache) - CACHE_MAX_ENTRIES
    for k, _ in items[:to_remove]:
        cache.pop(k, None)

def _cache_set(key: str, value: str):
    _prune_in_memory_cache()
    st.session_state.prompt_cache[key] = {"response": value, "ts": time.time()}

def _cache_get(key: str) -> Optional[str]:
    v = st.session_state.prompt_cache.get(key)
    if not v:
        return None
    if time.time() - v["ts"] > CACHE_TTL_SECONDS:
        st.session_state.prompt_cache.pop(key, None)
        return None
    return v["response"]

def make_cache_key(model: str, messages: List[Dict[str, str]]):
    digest = hashlib.sha256()
    digest.update(model.encode("utf-8"))
    digest.update(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()

# ----------------- Vector DB (with persistence) -----------------
@st.cache_resource
def load_vector_db(folder: str = PDF_DOCS_FOLDER):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    for file in os.listdir(folder):
        if file.lower().endswith(".pdf"):
            try:
                with fitz.open(os.path.join(folder, file)) as doc:
                    text = "\n".join(page.get_text() for page in doc)
                    chunks = splitter.split_text(text)
                    docs.extend([Document(page_content=c, metadata={"source": file}) for c in chunks])
            except Exception as e:
                logger.exception("Could not read PDF %s", file)
                st.warning(f"Could not read {file}: {e}")

    # If nothing to index, return empty retriever shim
    if not docs:
        class EmptyRetriever:
            def get_relevant_documents(self, q): return []
        return EmptyRetriever()

    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Try to load persisted FAISS if present
    try:
        if os.path.isdir(FAISS_INDEX_DIR) and os.listdir(FAISS_INDEX_DIR):
            vectordb = FAISS.load_local(FAISS_INDEX_DIR, embedder)
            logger.info("Loaded FAISS index from %s", FAISS_INDEX_DIR)
            return vectordb.as_retriever(search_type="similarity", k=K_VAL)
    except Exception:
        logger.exception("Failed to load persisted FAISS index, rebuilding.")

    # build and persist
    try:
        vectordb = FAISS.from_documents(docs, embedder)
        try:
            vectordb.save_local(FAISS_INDEX_DIR)
            logger.info("Persisted FAISS index to %s", FAISS_INDEX_DIR)
        except Exception:
            logger.exception("Failed to save FAISS index (non-fatal)")
        return vectordb.as_retriever(search_type="similarity", k=K_VAL)
    except Exception:
        logger.exception("Failed to build FAISS vectorstore")
        class EmptyRetriever:
            def get_relevant_documents(self, q): return []
        return EmptyRetriever()

retriever = load_vector_db()

# ----------------- OpenRouter helpers (Improved) -----------------
import itertools
import threading

# process-level rate limiter to avoid hammering API from same process
_last_call = 0.0
_rate_lock = threading.Lock()

def rate_limited_request():
    """Ensure a minimum delay between API calls (global per-process)."""
    global _last_call
    with _rate_lock:
        now = time.time()
        wait = MIN_DELAY - (now - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()

class RateLimitError(RuntimeError):
    pass

def query_openrouter_with_backoff(model: str, messages: List[Dict[str, str]], max_retries: int = 6, timeout: int = 30) -> str:
    """
    Backoff-based query. Respects Retry-After header. Returns assistant content or raises.
    """
    key = make_cache_key(model, messages)

    # cache lookup (in-memory then sqlite)
    cached = _cache_get(key)
    if cached:
        logger.debug("Cache hit (in-memory).")
        return cached
    if st.session_state.get("enable_sqlite", ENABLE_PERSISTENT_CACHE) and _sql_conn:
        cached_sql = sql_get(key)
        if cached_sql:
            _cache_set(key, cached_sql["response"])
            logger.debug("Cache hit (sqlite).")
            return cached_sql["response"]

    payload = {"model": model, "messages": messages}
    backoff = 1.0

    for attempt in range(max_retries):
        try:
            # apply process-level spacing
            rate_limited_request()

            r = requests.post(OPENROUTER_URL, headers=HEADERS_BASE, json=payload, timeout=timeout)

            # Handle rate limiting (429)
            if r.status_code == 429:
                retry_after_raw = r.headers.get("Retry-After")
                try:
                    retry_after = float(retry_after_raw) if retry_after_raw is not None else backoff
                except ValueError:
                    retry_after = backoff
                wait_time = max(0.0, retry_after)
                logger.warning("Received 429 from OpenRouter, attempt %s/%s, waiting %s seconds", attempt + 1, max_retries, wait_time)
                time.sleep(wait_time)
                backoff = min(backoff * 2, 60.0)
                continue

            r.raise_for_status()
            data = r.json()

            # extract content robustly
            content = None
            if isinstance(data.get("choices"), list) and data["choices"]:
                c0 = data["choices"][0]
                msg = c0.get("message") or c0.get("delta") or c0
                content = msg.get("content") if isinstance(msg, dict) else str(msg)
            elif data.get("text"):
                content = data["text"]
            else:
                content = json.dumps(data)

            # cache response
            _cache_set(key, content)
            if st.session_state.get("enable_sqlite", ENABLE_PERSISTENT_CACHE) and _sql_conn:
                try:
                    sql_set(key, model, messages, content)
                except Exception:
                    logger.exception("Failed to write to sqlite cache")
            return content

        except requests.RequestException as e:
            # on last attempt, wrap and re-raise so caller can decide user-friendly message
            logger.warning("RequestException on attempt %s: %s", attempt + 1, e)
            if attempt == max_retries - 1:
                logger.exception("OpenRouter request failed after retries")
                raise RuntimeError(f"OpenRouter request failed after retries: {e}")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)

    raise RuntimeError("Failed to get response from OpenRouter after retries")

# Round-robin cycling to spread load
_model_cycle = itertools.cycle([MODEL_MID, MODEL_CHEAP, MODEL_HIGH])

def query_models_balanced(messages: List[Dict[str, str]]) -> str:
    last_error = None
    # Try at most 3 different models (rotate)
    for _ in range(3):
        model = next(_model_cycle)
        try:
            return query_openrouter_with_backoff(model, messages)
        except Exception as e:
            last_error = e
            logger.warning("%s failed: %s", model, e)
            continue
    # After trying multiple models, raise last error
    logger.exception("All models failed.")
    raise RuntimeError(f"All models failed. Last error: {last_error}")

# ----------------- Simple RAG answer -----------------
def display_typing_animation(text: str, placeholder, chunk_size: int = 60, delay: float = 0.02):
    """
    Non-blocking-ish chunked display of text into provided placeholder.
    Note: Streamlit is single-threaded; this function yields chunks to UI to avoid printing char-by-char.
    """
    try:
        for i in range(0, len(text), chunk_size):
            placeholder.markdown(text[: i + chunk_size])
            time.sleep(delay)
        placeholder.markdown(text)
    except Exception:
        # fallback to direct write
        placeholder.markdown(text)

def friendly_error_message_for_exception(e: Exception) -> str:
    txt = str(e).lower()
    if "429" in txt or "rate" in txt or "too many" in txt or "rate-limit" in txt:
        return "⚠️ The server is busy right now (rate-limited). Please wait a few seconds and try again."
    return "⚠️ I'm having trouble connecting to the server right now. Please try again shortly."

def vanilla_rag_answer(context: str, question: str, lang: str = "English") -> str:
    """
    Compose prompt and query model. Returns friendly fallback strings on recoverable errors.
    """
    try:
        prompt = [
            {
                "role": "system",
                "content": (
                    f"You are BitsBuddy, a BITSian. Answer in the most analytical way covering all aspects and be helpful. "
                    f"Answer only when the question is about BITS; otherwise politely decline. Use relevant emojis. "
                    f"Answer in {lang}."
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
        ]
        return query_models_balanced(prompt)
    except Exception as e:
        logger.exception("vanilla_rag_answer failed")
        return friendly_error_message_for_exception(e)

# ----------------- Per-user rate limiting (session-based) -----------------
def can_user_make_request(uid: str, min_interval: float = 1.5) -> bool:
    """
    Return True if user can make a new request (based on last_call stored in session_state per uid).
    This prevents rapid-fire clicks generating multiple API calls.
    """
    key = f"last_call_{uid}"
    last = st.session_state.get(key, 0.0)
    now = time.time()
    if now - last < min_interval:
        return False
    st.session_state[key] = now
    return True

# ----------------- Session initialization -----------------
if "authenticated" in st.session_state and st.session_state["authenticated"]:
    if "chat_history" not in st.session_state:
        uid = st.session_state.get("user_uid")
        st.session_state.chat_history = load_user_chat_history(uid) if uid else []
    if "just_streamed" not in st.session_state:
        st.session_state.just_streamed = False
else:
    def login_screen():
        st.title("🔐 BITS Buddy Login")
        st.markdown("Please log in to continue")
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login / Sign Up"):
            if not name or not email or not password:
                st.error("Please fill in all fields.")
                return False
            try:
                email_norm = email.strip().lower()
                # NOTE: Using admin SDK to create/get users is server-side behavior.
                # In production, don't do client-signup using admin SDK.
                try:
                    user = auth.get_user_by_email(email_norm)
                    st.success(f"Welcome back, {user.display_name or name}!")
                    st.session_state.uid = user.uid
                    st.session_state.chat_history = load_user_chat_history(user.uid)
                except auth.UserNotFoundError:
                    user = auth.create_user(email=email_norm, password=password, display_name=name)
                    st.success(f"Account created! Welcome, {name}!")
                    st.session_state.uid = user.uid
                    st.session_state.chat_history = []
                st.session_state["user_uid"] = user.uid
                st.session_state["user_name"] = name
                st.session_state["authenticated"] = True
                st.rerun()
            except Exception as e:
                logger.exception("Authentication failed")
                st.error("Authentication failed. Check logs.")
                return False
    login_screen()
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title(f"Welcome {st.session_state.get('user_name', 'User')} 👋")

# ----------------- Main chat input handling -----------------
if user_query := st.chat_input("Ask me about BITS Pilani"):
    query = user_query.strip()
    if query:
        uid = st.session_state.get("user_uid") or "anonymous"
        # block rapid repeated requests from same user
        if not can_user_make_request(uid, min_interval=1.5):
            st.warning("You're sending requests too quickly. Please wait a second before trying again.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": query})
            try:
                # Retrieve relevant docs (safe wrapper)
                try:
                    docs = retriever.get_relevant_documents(query)
                    context = "\n".join([doc.page_content for doc in docs]) if docs else (
                        st.session_state.get("uploaded_content", "") or ""
                    )
                except Exception as e:
                    logger.exception("Retriever failed")
                    context = st.session_state.get("uploaded_content", "") or ""
                    st.warning("Retriever failed; continuing without RAG context.")

                final_answer = vanilla_rag_answer(context, query, lang=language)
                st.session_state.chat_history.append({"role": "assistant", "content": final_answer})

                if "user_uid" in st.session_state:
                    try:
                        save_user_chat_history(st.session_state.user_uid, st.session_state.chat_history)
                    except Exception:
                        logger.exception("Failed to save chat history (non-fatal)")

            except Exception as e:
                # single friendly message to user; detailed logged to server
                logger.exception("Unhandled exception during chat processing")
                msg = friendly_error_message_for_exception(e)
                st.session_state.chat_history.append({"role": "assistant", "content": msg})

# ----------------- Display chat history -----------------
for i, chat in enumerate(st.session_state.chat_history):
    with st.chat_message("user" if chat["role"] == "user" else "assistant"):
        if i == len(st.session_state.chat_history) - 1 and chat["role"] == "assistant":
            placeholder = st.empty()
            # chunked display to avoid per-character blocking and avoid double writes
            display_typing_animation(chat["content"], placeholder)
        else:
            st.markdown(chat["content"])

# ----------------- Sidebar history preview -----------------
with st.sidebar:
    st.subheader("📂 Chat History")
    preview_items = list(reversed(st.session_state.get("chat_history", [])))[:50]  # limit preview size
    for i, chat in enumerate(preview_items):
        role = chat.get("role", "user")
        content = chat.get("content", "")
        preview = content.replace("\n", " ")
        if len(preview) > 150:
            preview = preview[:150] + "..."
        if role == "user":
            st.markdown(f"**Q{i+1}:** {preview}")
        else:
            st.markdown(f"**A{i+1}:** {preview}")
        st.markdown("---")

# ----------------- Footer -----------------
st.markdown(
    """
<hr style="margin-top: 40px;">
<div style="
    text-align: center;
    color: #000;
    font-size: 14px;
    padding: 12px 0;
    background: linear-gradient(
        to right,
        red 0%, red 33.33%,
        lightblue 33.33%, lightblue 66.66%,
        yellow 66.66%, yellow 100%
    );
">
    Built with ❤️ by <b>BITS Pilani</b> · Pilani Campus · 
    <br>📬 Email: <a href="mailto:f20240347@pilani.bits-pilani.ac.in" style="color: #000; text-decoration: underline;">Contact us</a>
</div>
""",
    unsafe_allow_html=True,
)
