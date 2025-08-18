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
MODEL_DEFAULT = os.getenv("MODEL_DEFAULT") or "deepseek/deepseek-chat-v3-0324:free"
EMBED_MODEL = os.getenv("EMBED_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
K_VAL = int(os.getenv("K_VAL") or 4)

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH") or "./llm_cache.db"
ENABLE_PERSISTENT_CACHE = True

# ----------------- Firebase utils -----------------
def load_user_chat_history(uid: str) -> List[Dict[str, Any]]:
    try:
        ref = db.reference(f"user_chats/{uid}")
        snapshot = ref.get()
        if not snapshot:
            return []
        chat_data = snapshot.get("chat")
        return chat_data if isinstance(chat_data, list) else []
    except Exception:
        return []

def save_user_chat_history(uid: str, chat: List[Dict[str, Any]]):
    try:
        ref = db.reference(f"user_chats/{uid}")
        ref.set({"chat": chat})
    except Exception:
        pass

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

# ----------------- Streamlit page -----------------
st.set_page_config(page_title="BITS Buddy", layout="wide")

col1, col2 = st.columns([1, 6])  # adjust ratio as needed

with col1:
    st.image("bits_logo.jpg", width=60)  # smaller logo size

with col2:
    st.title(" BITS Buddy")

st.markdown("Ask me anything about BITS Pilani")

with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🔁 Start New Chat"):
        uid = st.session_state.get("user_uid")
        if uid:
            try:
                ref = db.reference(f"user_chats/{uid}")
                ref.delete()
            except Exception:
                pass
        st.session_state.chat_history = []
        st.rerun()

    uploaded_file = st.file_uploader("📄 Upload PDF or image", type=["pdf", "png", "jpg", "jpeg"])
    language = st.selectbox("🌐 Response Language", ["English", "Hindi", "Telugu", "Tamil", "Marathi", "Bengali"])
    st.markdown("---")
    st.checkbox("Enable Persistent SQLite Cache", value=ENABLE_PERSISTENT_CACHE, key="enable_sqlite")

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
    except Exception:
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

# ----------------- Cache helpers -----------------
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
            except Exception:
                pass

    if not docs:
        class EmptyRetriever:
            def get_relevant_documents(self, q): return []
        return EmptyRetriever()

    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = FAISS.from_documents(docs, embedder)
    return vectordb.as_retriever(search_type="similarity", k=K_VAL)

retriever = load_vector_db()

# ----------------- OpenRouter -----------------
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS_BASE = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}

def query_openrouter(model: str, messages: List[Dict[str, str]]) -> str:
    key = make_cache_key(model, messages)
    if _sql_conn:
        cached = sql_get(key)
        if cached:
            return cached

    r = requests.post(OPENROUTER_URL, headers=HEADERS_BASE, json={"model": model, "messages": messages}, timeout=30)
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]

    if _sql_conn:
        sql_set(key, model, messages, content)
    return content

# ----------------- Prompt builder -----------------
def build_prompt(context: str, question: str, lang: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": f"You are BitsBuddy, a helpful BITS Pilani assistant. Answer in {lang}."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]

# ----------------- Session -----------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = True  # skip login for now
    st.session_state["user_name"] = "BITSian"
    st.session_state["user_uid"] = "local_user"
    st.session_state["chat_history"] = []

st.title(f"Welcome {st.session_state.get('user_name', 'User')} 👋")

# ----------------- Main Chat -----------------
if user_query := st.chat_input("Ask me about BITS Pilani anything"):
    query = user_query.strip()
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Retrieve context
        docs = retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in docs])

        # Build + Query
        with st.chat_message("assistant"):
            try:
                messages = build_prompt(context, query, language)
                answer = query_openrouter(MODEL_DEFAULT, messages)
                animated = ""
                placeholder = st.empty()
                for c in answer:
                    animated += c
                    placeholder.markdown(animated + "|")
                    time.sleep(0.004)
                placeholder.markdown(animated)

                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                save_user_chat_history(st.session_state["user_uid"], st.session_state.chat_history)
            except Exception as e:
                st.error(f"Error: {e}")

# ----------------- Display history -----------------
for chat in st.session_state.chat_history:
    with st.chat_message("user" if chat["role"] == "user" else "assistant"):
        st.markdown(chat["content"])

# ----------------- Sidebar history -----------------
with st.sidebar:
    st.subheader("📂 Chat History")
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        st.markdown(f"**{chat['role'].title()} {i+1}:** {chat['content'][:120]}...")
        st.markdown("---")

st.markdown("""
<hr style="margin-top: 40px;">
<div style='
    text-align: center; 
    color: #fff; 
    font-size: 14px; 
    padding: 12px; 
    border-radius: 10px;
    background: linear-gradient(90deg, red, lightblue, yellow);
'>
    Built with ❤️ by <b>BITS Pilani</b> · Pilani Campus · <br>
    📬 Email: <a href="mailto:f20240347@pilani.bits-pilani.ac.in" style="color:#fff; text-decoration:underline;">Contact us</a>
</div>
""", unsafe_allow_html=True)
