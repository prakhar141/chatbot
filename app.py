# cleaned_buddy_vanilla.py
import os
import json
import sqlite3
import time
import hashlib
import fitz
import requests
import streamlit as st
from typing import List, Dict, Optional

# LangChain / FAISS imports
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

import firebase_admin
from firebase_admin import credentials, auth, db

# ================= CONFIG =================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or "YOUR_API_KEY"
MODEL_MID = os.getenv("MODEL_MID") or "openai/gpt-oss-20b:free"
EMBED_MODEL = os.getenv("EMBED_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
K_VAL = int(os.getenv("K_VAL") or 4)
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH") or "./llm_cache.db"
ENABLE_PERSISTENT_CACHE = True

# ================= FIREBASE =================
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

# ================= AUTHENTICATION (Passwordless) =================
if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    st.title("🔐 BITS Buddy Login")
    st.markdown("Enter your name and email to continue")

    name = st.text_input("Full Name")
    email = st.text_input("Email")

    if st.button("Login / Sign Up"):
        if not name or not email:
            st.error("Please fill in all fields.")
        else:
            email_norm = email.strip().lower()
            try:
                try:
                    user = auth.get_user_by_email(email_norm)
                    st.success(f"Welcome back, {user.display_name or name}!")
                    st.session_state.chat_history = []
                except auth.UserNotFoundError:
                    import secrets
                    random_password = secrets.token_urlsafe(16)
                    user = auth.create_user(email=email_norm, password=random_password, display_name=name)
                    st.success(f"Account created! Welcome, {name}!")
                    st.session_state.chat_history = []

                st.session_state["user_uid"] = user.uid
                st.session_state["user_name"] = name
                st.session_state["authenticated"] = True
                st.rerun()
            except Exception as e:
                st.error(f"Authentication failed: {e}")
    st.stop()

# ================= CHAT APP STARTS HERE =================
st.set_page_config(page_title="BITS Buddy", layout="wide")
# ================= PAGE HEADER =================

st.markdown(
    """
    <div style="display: flex; border-radius: 10px; overflow: hidden; box-shadow: 2px 2px 10px rgba(0,0,0,0.3);">
        <div style="background-color: #FFA500; flex: 1; padding: 20px; text-align: center; color: white; font-size: 32px; font-weight: bold;">
            🎓 BITS Buddy
        </div>
        <div style="background-color: #87CEEB; flex: 1; padding: 20px; text-align: center; color: white; font-size: 32px; font-weight: bold;">
            Buddy
        </div>
        <div style="background-color: #FF0000; flex: 1; padding: 20px; text-align: center; color: white; font-size: 32px; font-weight: bold;">
            
        </div>
   
    """,
    unsafe_allow_html=True
)
st.write("")  # small space after header

st.title(f"Welcome {st.session_state.get('user_name', 'User')} 👋")

# ================= SQLITE CACHE =================
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

# ================= IN-MEMORY CACHE =================
if "prompt_cache" not in st.session_state:
    st.session_state.prompt_cache = {}

def _cache_set(key: str, value: str):
    st.session_state.prompt_cache[key] = {"response": value, "ts": time.time()}

def _cache_get(key: str) -> Optional[str]:
    v = st.session_state.prompt_cache.get(key)
    return v["response"] if v else None

def make_cache_key(model: str, messages: List[Dict[str, str]]):
    digest = hashlib.sha256()
    digest.update(model.encode("utf-8"))
    digest.update(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()

# ================= VECTOR DB =================
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

# ================= OPENROUTER QUERY =================
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS_BASE = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}

def query_openrouter(model: str, messages: List[Dict[str, str]]) -> str:
    key = make_cache_key(model, messages)
    cached = _cache_get(key)
    if cached:
        return cached
    if ENABLE_PERSISTENT_CACHE and _sql_conn:
        cached_sql = sql_get(key)
        if cached_sql:
            _cache_set(key, cached_sql)
            return cached_sql

    payload = {"model": model, "messages": messages}
    r = requests.post(OPENROUTER_URL, headers=HEADERS_BASE, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content") or data.get("text") or str(data)

    _cache_set(key, content)
    if ENABLE_PERSISTENT_CACHE and _sql_conn:
        try:
            sql_set(key, model, messages, content)
        except:
            pass
    return content

# ================= PROMPTS =================
def build_primary_prompt(context: str, question: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are BitsBuddy, a BITSian senior. Answer concisely and helpfully with emojis."},
        {"role": "user", "content": f"Question: {question}\nContext:\n{context}"}
    ]

# ================= AUTHENTICATION CHECK =================
if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    st.warning("Authentication required.")
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ================= CHAT HANDLER WITH TYPING ANIMATION =================
user_query = st.text_input("", key="chat_input", placeholder="Type your question here...").strip()
if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    assistant_placeholder = st.empty()
    assistant_placeholder.markdown("⏳ I am preparing your answer...")

    try:
        docs = retriever.get_relevant_documents(user_query)
        context = "\n".join([doc.page_content for doc in docs]) if docs else ""
    except:
        context = ""

    prompt = build_primary_prompt(context, user_query)

    try:
        answer = query_openrouter(MODEL_MID, prompt)
    except Exception as e:
        answer = f"❌ Error generating answer: {e}"

    # Typing animation
    animated_text = ""
    for c in answer:
        animated_text += c
        assistant_placeholder.markdown(animated_text + "▌")
        time.sleep(0.02)
    assistant_placeholder.markdown(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})

# ================= DISPLAY CHAT HISTORY =================
for chat in st.session_state.chat_history:
    with st.chat_message("user" if chat["role"] == "user" else "assistant"):
        st.markdown(chat["content"])
# ================= PAGE FOOTER =================
st.markdown(
    """
    <div style="
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        padding: 10px 0;
        text-align: center;
        font-size: 14px;
        color: #555;
        border-top: 1px solid #ccc;
    ">
        Developed by BITS Pilani, Pilani Campus | &copy; 2025 Developer Tool
    </div>
    """,
    unsafe_allow_html=True
)
