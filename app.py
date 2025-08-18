# cleaned_buddy_simple.py
import os
import time
import hashlib
import json
import sqlite3
import fitz
import requests
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
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or "YOUR_API_KEY"
MODEL_CHEAP = os.getenv("MODEL_CHEAP") or "deepseek/deepseek-r1:free"
MODEL_MID = os.getenv("MODEL_MID") or "openai/gpt-oss-20b:free"
MODEL_HIGH = os.getenv("MODEL_HIGH") or "deepseek/deepseek-r1-0528:free"

EMBED_MODEL = os.getenv("EMBED_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
K_VAL = int(os.getenv("K_VAL") or 4)

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH") or "./llm_cache.db"
ENABLE_PERSISTENT_CACHE = True

# ----------------- Firebase chat history -----------------
def load_user_chat_history(uid: str) -> List[Dict[str, Any]]:
    try:
        ref = db.reference(f"user_chats/{uid}")
        snapshot = ref.get()
        if not snapshot:
            return []
        chat_data = snapshot.get("chat")
        if isinstance(chat_data, list):
            return chat_data
        st.warning(f"Unexpected chat format for UID {uid}, resetting history.")
        return []
    except Exception as e:
        st.error(f"Failed to load chat history for UID {uid}: {e}")
        return []

def save_user_chat_history(uid: str, chat: List[Dict[str, Any]]) -> bool:
    try:
        ref = db.reference(f"user_chats/{uid}")
        ref.set({"chat": chat})
        return True
    except Exception as e:
        st.error(f"Failed to save chat history for UID {uid}: {e}")
        return False

# ----------------- FIREBASE INIT -----------------
if not firebase_admin._apps:
    try:
        firebase_config = dict(st.secrets["firebase"])
        firebase_config["private_key"] = firebase_config["private_key"].replace("\\n", "\n")
        database_url = st.secrets["firebase"]["database_url"]
        cred = credentials.Certificate(firebase_config)
        firebase_admin.initialize_app(cred, {"databaseURL": database_url})
    except Exception as e:
        st.error(f"Firebase initialization failed: {e}")
        st.stop()
else:
    firebase_admin.get_app()

realtime_db = db.reference('/')

# ----------------- Streamlit page & sidebar -----------------
st.set_page_config(page_title="BITS Buddy", layout="wide")
col1, col2 = st.columns([1, 8])

with col1:
    st.image("bits_logo.jpg", width=50)

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
        st.session_state.chat_history = []
        st.session_state.just_streamed = False
        st.rerun()

    language = st.selectbox("🌐 Response Language", ["English", "Hindi", "Telugu", "Tamil", "Marathi", "Bengali"])
    st.markdown("---")
    st.checkbox("For Faster Loading", value=ENABLE_PERSISTENT_CACHE, key="enable_sqlite")

# ----------------- SQLITE CACHE -----------------
def init_sqlite(db_path: str = SQLITE_DB_PATH):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            model TEXT,
            messages_json TEXT,
            response TEXT,
            ts REAL
        )
    """)
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
        (key, model, json.dumps(messages, ensure_ascii=False), response, time.time())
    )
    _sql_conn.commit()

# ----------------- in-memory cache -----------------
if "prompt_cache" not in st.session_state:
    st.session_state.prompt_cache = {}

CACHE_MAX_ENTRIES = 4000

def _cache_set(key: str, value: str):
    if len(st.session_state.prompt_cache) >= CACHE_MAX_ENTRIES:
        oldest = min(st.session_state.prompt_cache.items(), key=lambda kv: kv[1]["ts"])[0]
        st.session_state.prompt_cache.pop(oldest, None)
    st.session_state.prompt_cache[key] = {"response": value, "ts": time.time()}

def _cache_get(key: str) -> Optional[str]:
    v = st.session_state.prompt_cache.get(key)
    return v["response"] if v else None

def make_cache_key(model: str, messages: List[Dict[str, str]]):
    digest = hashlib.sha256()
    digest.update(model.encode("utf-8"))
    digest.update(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()

# ----------------- Vector DB -----------------
@st.cache_resource
def load_vector_db(folder="."):
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
                st.warning(f"Could not read {file}: {e}")

    if not docs:
        class EmptyRetriever:
            def get_relevant_documents(self, q): return []
        return EmptyRetriever()

    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = FAISS.from_documents(docs, embedder)
    return vectordb.as_retriever(search_type="similarity", k=K_VAL)

retriever = load_vector_db()

# ----------------- OpenRouter helpers (Improved) -----------------
import itertools
import threading

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS_BASE = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}

# 🔹 Local rate limiter (per process)
_last_call = 0.0
_rate_lock = threading.Lock()
# Make this configurable via env; default 1.5s between calls
MIN_DELAY = float(os.getenv("OPENROUTER_MIN_DELAY", "1.5"))

def rate_limited_request():
    """Ensure a minimum delay between API calls (global lock)."""
    global _last_call
    with _rate_lock:
        now = time.time()
        wait = MIN_DELAY - (now - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()

def query_openrouter_with_backoff(model: str, messages: List[Dict[str, str]], max_retries: int = 6, timeout: int = 30) -> str:
    """Query OpenRouter with Retry-After support, exponential backoff, and caching."""
    key = make_cache_key(model, messages)

    # 🔹 First check cache
    cached = _cache_get(key)
    if cached:
        return cached
    if st.session_state.get("enable_sqlite", ENABLE_PERSISTENT_CACHE) and _sql_conn:
        cached_sql = sql_get(key)
        if cached_sql:
            _cache_set(key, cached_sql)
            return cached_sql

    payload = {"model": model, "messages": messages}
    backoff = 1.0

    for attempt in range(max_retries):
        try:
            # 🔹 Apply rate limiting
            rate_limited_request()

            r = requests.post(OPENROUTER_URL, headers=HEADERS_BASE, json=payload, timeout=timeout)

            # 🔹 Handle rate limiting errors
            if r.status_code == 429:
                # Respect server guidance if available
                retry_after_raw = r.headers.get("Retry-After")
                try:
                    retry_after = float(retry_after_raw) if retry_after_raw is not None else backoff
                except ValueError:
                    retry_after = backoff
                time.sleep(max(0.0, retry_after))
                backoff = min(backoff * 2, 60.0)  # cap backoff
                continue

            r.raise_for_status()
            data = r.json()

            # 🔹 Extract assistant text
            content = None
            if isinstance(data.get("choices"), list) and data["choices"]:
                c0 = data["choices"][0]
                msg = c0.get("message") or c0.get("delta") or c0
                content = msg.get("content") if isinstance(msg, dict) else str(msg)
            elif data.get("text"):
                content = data["text"]
            else:
                content = json.dumps(data)

            # 🔹 Cache response
            _cache_set(key, content)
            if st.session_state.get("enable_sqlite", ENABLE_PERSISTENT_CACHE) and _sql_conn:
                try:
                    sql_set(key, model, messages, content)
                except Exception:
                    pass

            return content

        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"OpenRouter request failed after retries: {e}")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)

    raise RuntimeError("Failed to get response from OpenRouter after retries")

# 🔹 Round-robin model cycling (load balancing)
_model_cycle = itertools.cycle([MODEL_MID, MODEL_CHEAP, MODEL_HIGH])

def query_models_balanced(messages: List[Dict[str, str]]) -> str:
    """Rotate between models to balance load, retry on failure."""
    last_error = None
    # Try up to len(unique models) times
    for _ in range(3):
        model = next(_model_cycle)
        try:
            return query_openrouter_with_backoff(model, messages)
        except Exception as e:
            st.warning(f"{model} failed: {e}")
            last_error = e
            continue
    raise RuntimeError(f"All models failed. Last error: {last_error}")

# ----------------- Simple RAG answer -----------------
def display_typing_animation(text: str, delay: float = 0.003):
    """Displays text with a typing effect inside Streamlit."""
    placeholder = st.empty()
    displayed = ""
    for char in text:
        displayed += char
        placeholder.markdown(displayed)
        time.sleep(delay)
    placeholder.markdown(displayed)

def vanilla_rag_answer(context: str, question: str, lang: str = "English") -> str:
    try:
        prompt = [
            {"role": "system", "content": (
                f"You are BitsBuddy, a BITSian. Answer in the most analytical way covering all aspects and be helpful. "
                f"Answer only when the question is about BITS; otherwise politely decline. Use relevant emojis. "
                f"Answer in {lang}."
            )},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
        ]
        # ✅ Use balanced rotation instead of single-model fallbacks
        return query_models_balanced(prompt)
    except Exception as e:
        return f"Error in Vanilla RAG: {e}"

# ----------------- Session init -----------------
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
                st.error(f"Authentication failed: {e}")
                return False

    login_screen()
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title(f"Welcome {st.session_state.get('user_name', 'User')} 👋")

if user_query := st.chat_input("Ask me about BITS Pilani"):
    query = user_query.strip()
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})

        try:
            docs = retriever.get_relevant_documents(query)
            context = "\n".join([doc.page_content for doc in docs]) if docs else (
                st.session_state.get("uploaded_content", "") or ""
            )
        except Exception as e:
            context = st.session_state.get("uploaded_content", "") or ""
            st.warning(f"Retriever failed: {e}")

        try:
            final_answer = vanilla_rag_answer(context, query, lang=language)
            st.session_state.chat_history.append({"role": "assistant", "content": final_answer})

            if "uid" in st.session_state:
                save_user_chat_history(st.session_state.uid, st.session_state.chat_history)

        except Exception as e:
            st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {e}"})

# ----------------- Display chat history -----------------
for i, chat in enumerate(st.session_state.chat_history):
    with st.chat_message("user" if chat["role"] == "user" else "assistant"):
        if i == len(st.session_state.chat_history) - 1 and chat["role"] == "assistant":
            display_typing_animation(chat["content"])
        else:
            st.markdown(chat["content"])

# ----------------- Sidebar history preview -----------------
with st.sidebar:
    st.subheader("📂 Chat History")
    for i, chat in enumerate(reversed(st.session_state.get("chat_history", []))):
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
st.markdown("""
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
""", unsafe_allow_html=True)
