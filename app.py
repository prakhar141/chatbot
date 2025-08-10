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
from firebase_admin import credentials, auth
import streamlit as st
import os
#from firebase_admin import firestore
from firebase_admin import credentials, initialize_app, db
# ========== CONFIG (tweak these models per your OpenRouter access) ==========
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or "YOUR_API_KEY"
# Tiered models (prefer models you actually can access via OpenRouter)
MODEL_CHEAP = os.getenv("MODEL_CHEAP") or "deepseek/deepseek-chat-v3-0324:free"        # cheap, good for short tasks
MODEL_MID = os.getenv("MODEL_MID") or "openai/gpt-oss-20b:free"          # mid-tier for primary/drafts
MODEL_HIGH = os.getenv("MODEL_HIGH") or "deepseek/deepseek-r1-0528:free"              # high-quality for final answer (may be paid)
# fallback list in order if model hits quota or returns 429
MODEL_FALLBACKS = [MODEL_MID, MODEL_CHEAP]

# Embedding model for retriever
EMBED_MODEL = os.getenv("EMBED_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
K_VAL = int(os.getenv("K_VAL") or 4)

# Persistent cache DB file
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH") or "./llm_cache.db"
# Toggle persistent cache
ENABLE_PERSISTENT_CACHE = True
def load_user_chat_history(uid: str) -> List[Dict[str, Any]]:
    try:
        doc_ref = db.collection("user_chats").document(uid)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict().get("chat", [])
        else:
            return []
    except Exception as e:
        st.warning(f"Failed to load chat history: {e}")
        return []

def save_user_chat_history(uid: str, chat: List[Dict[str, Any]]):
    try:
        doc_ref = db.collection("user_chats").document(uid)
        doc_ref.set({"chat": chat})
    except Exception as e:
        st.warning(f"Failed to save chat history: {e}")

# ====== FIREBASE INIT ======
if not firebase_admin._apps:
    firebase_config = dict(st.secrets["firebase"])
    firebase_config["private_key"] = firebase_config["private_key"].replace("\\n", "\n")
    cred = credentials.Certificate(firebase_config)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://fir-d2037-default-rtdb.firebaseio.com/'  # your Realtime Database URL here
    })
else:
    firebase_admin.get_app()

# ====== LOGIN SCREEN ======
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
            # Try to get user
            try:
                user = auth.get_user_by_email(email)
                st.success(f"Welcome back, {name}!")
            except auth.UserNotFoundError:
                user = auth.create_user(email=email, password=password, display_name=name)
                st.success(f"Account created! Welcome, {name}!")

            # Store uid in session state
            st.session_state["user_uid"] = user.uid

            st.session_state["user_name"] = name
            st.session_state["authenticated"] = True
            st.rerun()

        except Exception as e:
            st.error(f"Authentication failed: {e}")
            return False

# ====== CHECK AUTH BEFORE LOADING APP ======
if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    login_screen()
    st.stop()  # Stop execution until login happens

# App UI settings
st.set_page_config(page_title="BITS Buddy", layout="wide")
st.title("🎓 BITS Buddy")
st.markdown("Ask me anything about BITS Pilani")

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🔁 Start New Chat"):
       uid = st.session_state.get("user_uid")
       if uid:
        db.collection("user_chats").document(uid).delete()
       st.session_state.clear()
       st.rerun()

    uploaded_file = st.file_uploader("📄 Upload PDF or image", type=["pdf", "png", "jpg", "jpeg"])
    language = st.selectbox("🌐 Response Language", ["English", "Hindi", "Telugu", "Tamil", "Marathi", "Bengali"])
    st.markdown("---")
    st.checkbox("Enable Persistent SQLite Cache", value=ENABLE_PERSISTENT_CACHE, key="enable_sqlite")

# ========== SQLITE PERSISTENT CACHE SETUP ==========
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

# ========== UTIL: in-memory prompt cache (session persistent)
if "prompt_cache" not in st.session_state:
    st.session_state.prompt_cache = {}

CACHE_MAX_ENTRIES = 4000

def _cache_set(key: str, value: str):
    if len(st.session_state.prompt_cache) >= CACHE_MAX_ENTRIES:
        # simple eviction: remove oldest
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

# ========== VECTOR DB (cached resource) ==========
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
                    docs.extend([
                        Document(page_content=c, metadata={"source": file})
                        for c in chunks
                    ])
            except Exception as e:
                st.warning(f"Could not read {file}: {e}")

    if not docs:
        class EmptyRetriever:
            def get_relevant_documents(self, q): return []
        return EmptyRetriever()

    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = FAISS.from_documents(docs, embedder)
    return vectordb.as_retriever(search_type="similarity", k=K_VAL)
# Load retriever once (indexes files in current folder)
retriever = load_vector_db()

# ========== FILE PROCESSING (when user uploads) ==========
uploaded_content = ""
if uploaded_file:
    file_type = uploaded_file.type
    if file_type == "application/pdf":
        try:
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                uploaded_content = "\n".join(page.get_text() for page in doc)
        except Exception as e:
            st.warning(f"PDF read error: {e}")
    elif "image" in file_type:
        try:
            img = Image.open(uploaded_file)
            uploaded_content = "[Image content; enable OCR to extract text]"
        except Exception as e:
            st.warning(f"Image read error: {e}")

    if uploaded_content.strip():
        st.success("✅ Extracted content from file.")
        st.text_area("📄 Preview (first 1000 chars)", uploaded_content[:1000], height=200)
    else:
        st.warning("⚠️ Couldn't extract readable text from the file.")

# ========== OPENROUTER QUERY WITH BACKOFF + FALLBACKS + PERSISTENT CACHE ==========
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS_BASE = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}


def query_openrouter_with_backoff(model: str, messages: List[Dict[str, str]], max_retries: int = 4, timeout: int = 30) -> str:
    """Queries OpenRouter for a single model with exponential backoff.
    This function uses in-memory cache and (optionally) SQLite persistent cache.
    """
    key = make_cache_key(model, messages)

    # 1) Try in-memory cache
    cached = _cache_get(key)
    if cached:
        return cached

    # 2) Try persistent cache
    if st.session_state.get("enable_sqlite", ENABLE_PERSISTENT_CACHE) and _sql_conn:
        cached_sql = sql_get(key)
        if cached_sql:
            # populate in-memory cache for faster next calls
            _cache_set(key, cached_sql)
            return cached_sql

    payload = {"model": model, "messages": messages}
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            r = requests.post(OPENROUTER_URL, headers=HEADERS_BASE, json=payload, timeout=timeout)
            if r.status_code == 429:
                # signal caller to try fallback model
                raise requests.HTTPError("429")
            r.raise_for_status()
            data = r.json()
            content = None
            if isinstance(data.get("choices"), list) and data["choices"]:
                c = data["choices"][0]
                msg = c.get("message") or c.get("delta") or c
                content = msg.get("content") if isinstance(msg, dict) else str(msg)
            elif data.get("text"):
                content = data.get("text")
            else:
                content = json.dumps(data)

            # cache results
            _cache_set(key, content)
            if st.session_state.get("enable_sqlite", ENABLE_PERSISTENT_CACHE) and _sql_conn:
                try:
                    sql_set(key, model, messages, content)
                except Exception:
                    pass
            return content

        except requests.HTTPError as e:
            if str(e).find("429") >= 0:
                # propagate to let higher-level function try fallbacks
                raise
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff)
            backoff *= 2
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("Failed to get response from OpenRouter after retries")


def query_models_with_fallbacks(models: List[str], messages: List[Dict[str, str]]) -> str:
    """Try models in order; on 429/quota try next. Uses persistent + in-memory cache.
    Returns first successful response or raises RuntimeError.
    """
    last_error = None
    for m in models:
        try:
            return query_openrouter_with_backoff(m, messages)
        except requests.HTTPError as e:
            # Likely quota/429 — try next model
            last_error = e
            continue
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"All models failed. Last error: {last_error}")

# ========== PROMPT BUILDERS (concise prompts to save tokens) ==========
def scratchpad_reasoning(context: str, question: str) -> str:
    return (
        f"Let's think step-by-step.\n\n"
        f"Context (shortened):\n"
        f"{(context[:2000] + '...') if len(context) > 2000 else context}\n\n"
        f"Question:\n"
        f"{question}"
    )


def build_thinking_prompt(question: str, context: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an assistant that narrates a concise, casual internal monologue "
                "before answering. Keep it 2-4 short sentences, conversational, use 'Hmm...', "
                "'Oh I see...', 'Wait...' and DO NOT give the final answer — only describe what "
                "you are thinking and what you plan to do next."
            )
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Relevant context:\n"
                f"{(context[:1500] + '...') if len(context) > 1500 else context}"
            )
        }
    ]


def build_primary_prompt(context: str, question: str, lang: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                f"You are BitsBuddy, a BITSian senior. Answer in {lang}. "
                f"Use emojis, be concise and helpful. Provide actionable steps if relevant."
            )
        },
        {
            "role": "user",
            "content": scratchpad_reasoning(context, question)
        }
    ]


def build_critic_prompt(context: str, question: str, answer: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an honest critic checking the assistant’s answer for factual errors, "
                "incompleteness, or hallucinations. Keep critiques short and list any unsupported "
                "claims with reasons."
            )
        },
        {
            "role": "user",
            "content": (
                f"Context:\n"
                f"{(context[:1500] + '...') if len(context) > 1500 else context}\n\n"
                f"Question:\n"
                f"{question}\n\n"
                f"Answer:\n"
                f"{answer}\n\n"
                f"Critique and list corrections:"
            )
        }
    ]


def build_final_prompt(context: str, question: str, answer: str, critique: str, lang: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                f"You are BitsBuddy+ with self-evaluation enabled. Based on critique, "
                f"revise your original answer. Be clear and concise in {lang}."
            )
        },
        {
            "role": "user",
            "content": (
                f"Original Answer:\n"
                f"{answer}\n\n"
                f"Critique:\n"
                f"{critique}\n\n"
                f"Now improve the answer accordingly."
            )
        }
    ]
# ========== MODULAR RAG PIPELINE (uses tiering + caching + fallbacks) ==========

def modular_rag_smart_answer(context: str, question: str, lang: str = "English") -> Dict[str, Any]:
    """Run stages: thinking (cheap), primary (mid), critique (cheap), final (high)
    All stages use caching and fallbacks to avoid exhausting quota.
    """
    result = {}
    try:
        # Thinking (cheap)
        thinking_msgs = build_thinking_prompt(question, context)
        thinking = query_models_with_fallbacks([MODEL_CHEAP] + MODEL_FALLBACKS, thinking_msgs)
        result["thinking"] = thinking

        # Primary draft (mid-tier)
        primary_msgs = build_primary_prompt(context, question, lang)
        primary = query_models_with_fallbacks([MODEL_MID] + MODEL_FALLBACKS, primary_msgs)
        result["primary"] = primary

        # Critique (cheap)
        critique_msgs = build_critic_prompt(context, question, primary)
        critique = query_models_with_fallbacks([MODEL_CHEAP] + MODEL_FALLBACKS, critique_msgs)
        result["critique"] = critique

        # Final revise (attempt high quality model first, then fallbacks)
        final_msgs = build_final_prompt(context, question, primary, critique, lang)
        final = query_models_with_fallbacks([MODEL_HIGH] + MODEL_FALLBACKS, final_msgs)
        result["final"] = final

        return result
    except Exception as e:
        return {"error": str(e)}

# ========== SESSION INIT ==========
if "authenticated" in st.session_state and st.session_state["authenticated"]:
    if "chat" not in st.session_state:
        uid = st.session_state.get("user_uid")
        st.session_state.chat = load_user_chat_history(uid) if uid else []
    
    # Initialize just_streamed here
    if "just_streamed" not in st.session_state:
        st.session_state.just_streamed = False
# ========== CHAT HANDLER (UI) ==========
query = st.chat_input("💬 Ask anything about BITS Pilani...")
if query:
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        try:
            docs = retriever.get_relevant_documents(query)
            context = "\n".join([doc.page_content for doc in docs]) if docs else (uploaded_content or "")

            # 0) Thinking monologue (stream-like animation locally)
            thinking_prompt = build_thinking_prompt(query, context)
            thinking_text = query_models_with_fallbacks([MODEL_CHEAP] + MODEL_FALLBACKS, thinking_prompt)
            animated = ""
            for c in thinking_text:
                animated += c
                thinking_placeholder.markdown(f"**Thinking:** {animated}|")
                time.sleep(0.01)
            thinking_placeholder.markdown(f"**Thinking:** {animated}")

            time.sleep(0.25)
            thinking_placeholder.markdown("🔁 Reasoning...\n\n• ✏️ Drafting initial answer...")

            rag_result = modular_rag_smart_answer(context, query, lang=language)

            chat_record = {
                "question": query,
                "thinking": rag_result.get("thinking", ""),
                "primary": rag_result.get("primary", ""),
                "critique": rag_result.get("critique", ""),
                "final": rag_result.get("final", rag_result.get("error", "Sorry — something went wrong.")),
                "language": language
            }

            if "error" in rag_result:
                thinking_placeholder.markdown(f"❌ Error while generating answer: {rag_result['error']}")
            else:
                # Stream final answer text
                final_answer = chat_record["final"]
                animated = ""
                for c in final_answer:
                    animated += c
                    thinking_placeholder.markdown(animated + "|")
                    time.sleep(0.004)
                thinking_placeholder.markdown(animated)

            st.session_state.chat.append(chat_record)
            st.session_state.just_streamed = True

        except Exception as e:
            thinking_placeholder.markdown(f"❌ Error: {e}")
            st.session_state.chat.append({
                "question": query,
                "thinking": f"Error generating thinking: {e}",
                "primary": "",
                "critique": "",
                "final": f"Error: {e}",
                "language": language
            })
            st.session_state.just_streamed = True

# ========== DISPLAY CHAT HISTORY ==========
if st.session_state.just_streamed and len(st.session_state.chat) > 0:
    history_to_show = st.session_state.chat[:-1]
else:
    history_to_show = st.session_state.chat

for chat in reversed(history_to_show):
    with st.chat_message("user"):
        st.markdown(chat["question"])
    with st.chat_message("assistant"):
        st.markdown(chat["final"])

if st.session_state.just_streamed:
    st.session_state.just_streamed = False

# ========== SIDEBAR HISTORY ==========
with st.sidebar:
    st.subheader("📂 Chat History")
    for i, chat in enumerate(reversed(st.session_state.chat)):
        st.markdown(f"**Q{i+1}:** {chat['question']}")
        st.markdown(f"**A{i+1}:** {chat['final'][:150]}...")
        st.markdown("---")

# ========== FOOTER ==========
st.markdown("""
<hr style="margin-top: 40px;">
<div style='text-align: center; color: #888; font-size: 14px;'>
    Built with ❤️ by <b>Prakhar Mathur</b> · BITS Pilani · 
    <br>📬 Email: <a href="mailto:f20240347@pilani.bits-pilani.ac.in">Contact Prakhar</a>
</div>
""", unsafe_allow_html=True)  
