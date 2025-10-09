# cleaned_buddy.py
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
import re
# LangChain/FAISS imports
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

import firebase_admin
from firebase_admin import credentials, auth, db
import time
import streamlit as st
from sentence_transformers import SentenceTransformer, util

# ========== CONFIG (tweak these models per your OpenRouter access) ==========
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or "YOUR_API_KEY"
MODEL_CHEAP = os.getenv("MODEL_CHEAP") or "deepseek/deepseek-chat-v3-0324:free"
MODEL_MID = os.getenv("MODEL_MID") or "openai/gpt-oss-20b:free"
MODEL_HIGH = os.getenv("MODEL_HIGH") or "deepseek/deepseek-r1-0528:free"
MODEL_FALLBACKS = [MODEL_MID, MODEL_CHEAP]

EMBED_MODEL = os.getenv("EMBED_MODEL") or "multi-qa-mpnet-base-dot-v1"
K_VAL = int(os.getenv("K_VAL") or 4)

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH") or "./llm_cache.db"
ENABLE_PERSISTENT_CACHE = True

# ----------------- utilities for firebase chat history -----------------
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
col1, col2 = st.columns([1, 5])

with col1:
    st.image("bits_logo.jpg", width=60)

with col2:
    st.markdown("<h1 style='margin-top: 10px;'>BITS Buddy</h1>", unsafe_allow_html=True)

st.markdown("Ask me anything about BITS Pilani Admission")

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

    language = st.selectbox("🌐 Response Language", ["English"])
    #st.checkbox("🧠Deep Think", value=False, key="use_smart_llm") 
    st.markdown("---")
    #st.checkbox("For fast loading", value=ENABLE_PERSISTENT_CACHE, key="enable_sqlite")

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


# ---------------- CONFIG ----------------
EMBED_MODEL = "multi-qa-mpnet-base-dot-v1"
K_VAL = 4

# Hugging Face URLs for prebuilt FAISS index
FAISS_INDEX_URL = "https://huggingface.co/datasets/prakhar146/chatbot/resolve/main/index.faiss"
FAISS_PKL_URL = "https://huggingface.co/datasets/prakhar146/chatbot/resolve/main/index.pkl"

# Local directory to store downloaded files
LOCAL_FAISS_DIR = "./faiss_store"
os.makedirs(LOCAL_FAISS_DIR, exist_ok=True)
LOCAL_INDEX_FILE = os.path.join(LOCAL_FAISS_DIR, "index.faiss")
LOCAL_PKL_FILE = os.path.join(LOCAL_FAISS_DIR, "index.pkl")

# ---------------- HELPER TO DOWNLOAD FILES ----------------
def download_if_not_exists(url: str, local_path: str):
    if not os.path.exists(local_path):
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

# ---------------- LOAD VECTOR DB ----------------
@st.cache_resource
def load_vector_db_from_hf():
    try:
        # Download files if missing
        download_if_not_exists(FAISS_INDEX_URL, LOCAL_INDEX_FILE)
        download_if_not_exists(FAISS_PKL_URL, LOCAL_PKL_FILE)

        # Load the FAISS index with safe override
        embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectordb = FAISS.load_local(
            LOCAL_FAISS_DIR,
            embedder,
            allow_dangerous_deserialization=True  # ✅ required to read index.pkl
        )

        return vectordb.as_retriever(search_type="similarity", k=K_VAL)

    except Exception as e:
        st.warning(f"Failed to load vector DB: {e}")

        class EmptyRetriever:
            def get_relevant_documents(self, query):
                return []

        return EmptyRetriever()

# ---------------- LOAD RETRIEVER ----------------
retriever = load_vector_db_from_hf()
from bs4 import BeautifulSoup

# ---------------------- 0️⃣ Real-time BITSAdmission content ----------------------
BITSADMISSION_URLS = [
    "https://www.bitsadmission.com/index.html",
    "https://www.bitsadmission.com/FD/FD.html"
]

def fetch_bitsadmission_content() -> str:
    """Fetch and combine all relevant BITSAdmission pages in plain text."""
    combined_text = ""
    for url in BITSADMISSION_URLS:
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator="\n")
            text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
            combined_text += f"\n\n=== Content from {url} ===\n{text}"
        except Exception as e:
            combined_text += f"\n\n⚠️ Failed to fetch {url}: {e}"
    return combined_text

# Cache website content in session to avoid multiple requests
if "bitsadmission_content" not in st.session_state:
    st.session_state.bitsadmission_content = fetch_bitsadmission_content()

# ----------------- OpenRouter helpers (unchanged) -----------------
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS_BASE = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}

def query_openrouter_with_backoff(model: str, messages: List[Dict[str, str]], max_retries: int = 4, timeout: int = 30) -> str:
    key = make_cache_key(model, messages)
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
            r = requests.post(OPENROUTER_URL, headers=HEADERS_BASE, json=payload, timeout=timeout)
            if r.status_code == 429:
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

            _cache_set(key, content)
            if st.session_state.get("enable_sqlite", ENABLE_PERSISTENT_CACHE) and _sql_conn:
                try:
                    sql_set(key, model, messages, content)
                except Exception:
                    pass
            return content
        except requests.HTTPError as e:
            if "429" in str(e):
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
    last_error = None
    for m in models:
        try:
            return query_openrouter_with_backoff(m, messages)
        except requests.HTTPError as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"All models failed. Last error: {last_error}")

# ----------------- Prompts and RAG pipeline (unchanged structure) -----------------
def scratchpad_reasoning(context: str, question: str) -> str:
    return (
        f"Let's think step-by-step.\n\nContext (shortened):\n"
        f"{(context[:2000] + '...') if len(context) > 2000 else context}\n\nQuestion:\n{question}"
    )

def build_thinking_prompt(question: str, context: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": ("You are an assistant that narrates a concise, casual internal monologue "
                                      "before answering. Keep it 2-4 short sentences, conversational, use 'Hmm...', "
                                      "'Oh I see...', 'Wait...' and DO NOT give the final answer — only describe what "
                                      "you are thinking and what you plan to do next.")},
        {"role": "user", "content": (f"Question: {question}\n\nRelevant context:\n"
                                     f"{(context[:1500] + '...') if len(context) > 1500 else context}")}
    ]

def build_primary_prompt(context: str, question: str, lang: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": (f"You are BitsBuddy, a BITSian Assistant. Answer in {lang}. "
                                       "Use emojis, be concise and helpful.always satisfy user'egoAnswer questions which are relevanto bits only.otherwise politely tell ur capabilities")},
        {"role": "user", "content": scratchpad_reasoning(context, question)}
    ]

def build_critic_prompt(context: str, question: str, answer: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": ("You are an honest critic checking the assistant’s answer for factual errors, "
                                       "incompleteness, or hallucinations. Keep critiques short and list any unsupported "
                                       "claims with reasons.")},
        {"role": "user", "content": (f"Context:\n{(context[:1500] + '...') if len(context) > 1500 else context}\n\n"
                                     f"Question:\n{question}\n\nAnswer:\n{answer}\n\nCritique and list corrections:")}
    ]

def build_final_prompt(context: str, question: str, answer: str, critique: str, lang: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": (f"You are BitsBuddy with self-evaluation enabled.Use Relevant Emojis.Based on critique, "
                                       f"always satisfy user's ego.Never invent or use outside knowledge. Stay faithful to CONTEXT only. Be clear and concise in {lang}.") },
        {"role": "user", "content": (f"Original Answer:\n{answer}\n\nCritique:\n{critique}\n\nNow improve the answer accordingly.")}
    ]

def modular_rag_smart_answer(context: str, question: str, lang: str = "English") -> Dict[str, Any]:
    result = {}
    try:
        thinking_msgs = build_thinking_prompt(question, context)
        thinking = query_models_with_fallbacks([MODEL_CHEAP] + MODEL_FALLBACKS, thinking_msgs)
        result["thinking"] = thinking

        primary_msgs = build_primary_prompt(context, question, lang)
        primary = query_models_with_fallbacks([MODEL_MID] + MODEL_FALLBACKS, primary_msgs)
        result["primary"] = primary

        critique_msgs = build_critic_prompt(context, question, primary)
        critique = query_models_with_fallbacks([MODEL_CHEAP] + MODEL_FALLBACKS, critique_msgs)
        result["critique"] = critique

        final_msgs = build_final_prompt(context, question, primary, critique, lang)
        final = query_models_with_fallbacks([MODEL_HIGH] + MODEL_FALLBACKS, final_msgs)
        result["final"] = final

        return result
    except Exception as e:
        return {"error": str(e)}
def vanilla_rag_answer(context: str, question: str, lang: str = "English") -> str:
    """Simple retriever + one model answer, no self-critique or multi-step LLM calls."""
    prompt = [
        {"role": "system", "content": f"You are BitsBuddy,.Never guess or make up facts. Answer ONLY if the question is directly related to BITS Pilani,always staisfy user's ego. Answer clearly in {lang}."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
    ]
    try:
        return query_models_with_fallbacks([MODEL_MID] + MODEL_FALLBACKS, prompt)
    except Exception as e:
        return f"⚠️ Error generating answer: {e}"
        
def build_clarification_prompt(last_answer: str, user_query: str, lang: str = "English"):
    return [
        {
            "role": "system",
            "content": (
                f"You are BitsBuddy. The user did not understand your previous answer. "
                f"Re-explain it step by step as if to a beginner "
                f"Do NOT introduce new context or use outside information. "
                f"Reanswer it from scratch.Answer in {lang}."
            )
        },
        {
            "role": "user",
            "content": f"Previous Answer:\n{last_answer}\n\nUser Query:\n{user_query}"
        }
    ]

# ----------------- Session init -----------------
# Initialize Firebase (make sure to replace this with your Firebase credentials)
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

# Custom styles for the login page with animations
st.markdown("""
    <style>
        /* Global Styles */
        body {
            font-family: 'Roboto', sans-serif;
            background-color: #f4f7fa;
            margin: 0;
            padding: 0;
            animation: fadeIn 1.5s ease-in;
        }
        .login-container {
            padding: 40px 50px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            width: 400px;
            margin: auto;
            text-align: center;
            display: flex;
            flex-direction: column;
            justify-content: center;
            opacity: 0;
            animation: fadeIn 1.5s ease-in 0.5s forwards;
        }
        @keyframes fadeIn {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }
        .login-title {
            font-size: 36px;
            color: #2f54eb;
            font-weight: 600;
            animation: slideIn 1s ease-out;
        }
        @keyframes slideIn {
            from {
                transform: translateX(-50%);
            }
            to {
                transform: translateX(0);
            }
        }
        .login-subtitle {
            font-size: 18px;
            color: #555;
            margin-bottom: 30px;
            animation: slideIn 1s ease-out 0.5s;
        }
        .form-input {
            margin-bottom: 20px;
            width: 100%;
            padding: 12px;
            font-size: 16px;
            border: 1px solid #dcdfe6;
            border-radius: 5px;
            outline: none;
            transition: all 0.3s ease-in-out;
        }
        .form-input:focus {
            border-color: #2f54eb;
            box-shadow: 0 0 5px rgba(47, 85, 235, 0.5);
        }
        .form-input:hover {
            border-color: #1d39c4;
        }
        .stButton>button {
            background-color: #2f54eb;
            color: white;
            border: none;
            padding: 14px 0;
            border-radius: 5px;
            font-size: 18px;
            width: 100%;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #1d39c4;
        }
        .error-message {
            background-color: #ff4d4f;
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            font-size: 14px;
            animation: bounce 1s ease-in-out;
        }
        @keyframes bounce {
            0% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0); }
        }
        .success-message {
            background-color: #52c41a;
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            font-size: 14px;
            animation: fadeIn 1.5s ease-in;
        }
    </style>
""", unsafe_allow_html=True)

# Check if the user is authenticated
if "authenticated" in st.session_state and st.session_state["authenticated"]:
    if "chat_history" not in st.session_state:
        uid = st.session_state.get("user_uid")
        st.session_state.chat_history = load_user_chat_history(uid) if uid else []
    if "just_streamed" not in st.session_state:
        st.session_state.just_streamed = False
else:
    # Show login screen if not authenticated
    def login_screen():
        # Ensure the form container is properly styled and no empty containers
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        # Title and Subtitle
        st.markdown('<h2 class="login-title">Welcome to BITS Buddy <span>🤖</span></h2>', unsafe_allow_html=True)
        st.markdown('<p class="login-subtitle">Please login or sign up to continue 🚀</p>', unsafe_allow_html=True)
        
        with st.form(key='login_form', clear_on_submit=True):
            # Input fields with emojis
            name = st.text_input("Full Name 📝", placeholder="Enter your full name")
            email = st.text_input("Email 📧", placeholder="Enter your email")
            password = st.text_input("Password 🔑", type="password", placeholder="Enter your password")
            
            submit_button = st.form_submit_button("Login / Sign Up 👏")
            
            if submit_button:
                if not name or not email or not password:
                    st.error("⚠️ Please fill in all fields.", icon="🚨")
                    return False
                try:
                    email_norm = email.strip().lower()
                    try:
                        user = auth.get_user_by_email(email_norm)
                        st.success(f"Welcome back, {user.display_name or name}! 🎉")
                        st.session_state.uid = user.uid
                        st.session_state.chat_history = load_user_chat_history(user.uid)
                    except auth.UserNotFoundError:
                        user = auth.create_user(email=email_norm, password=password, display_name=name)
                        st.success(f"Account created! Welcome, {name}! 🌟")
                        st.session_state.uid = user.uid
                        st.session_state.chat_history = []
                    st.session_state["user_uid"] = user.uid
                    st.session_state["user_name"] = name
                    st.session_state["authenticated"] = True
                    st.rerun()
                except Exception as e:
                    st.error(f"⚠️ Authentication failed: {e}")
                    return False

        # Removed unnecessary footer
        st.markdown('</div>', unsafe_allow_html=True)

    login_screen()
    st.stop()

# ----------------- Main chat handler (auto pipeline selection) -----------------

st.title(f"What's your agenda today? {st.session_state.get('user_name', 'User')} 👋")

# ----------------------
# 1️⃣ Load embedding model
# ----------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Reference queries that always need deep reasoning
deep_reasoning_refs = [
    "Explain how something works",
    "Compare advantages and disadvantages",
    "Predict the outcome based on data",
    "Evaluate the process step by step",
    "Explain the impact or effect of X",
    "Provide a detailed reasoning or analysis"
]
deep_ref_embeddings = embed_model.encode(deep_reasoning_refs, convert_to_tensor=True)

# ----------------------
# 2️⃣ DeepThink heuristic
# ----------------------
def should_use_deepthink(query: str) -> bool:
    """Decides if a query needs deep reasoning."""
    q = query.strip().lower()

    # keyword heuristics
    reasoning_keywords = [
        "why", "how", "explain", "difference", "compare",
        "advantages", "disadvantages", "steps", "process",
        "predict", "evaluate", "simulate", "impact", "effect"
    ]
    factoid_keywords = [
        "what is", "who is", "when is", "define",
        "location", "fee", "contact", "hostel", "mess", "address"
    ]

    if any(k in q for k in reasoning_keywords):
        return True
    if any(k in q for k in factoid_keywords):
        return False

    # length heuristic
    if len(q.split()) > 15:
        return True

    # semantic similarity
    query_embedding = embed_model.encode(q, convert_to_tensor=True)
    score = util.cos_sim(query_embedding, deep_ref_embeddings).max().item()
    return score > 0.6
# ----------------------
# 3️⃣ Modular pipeline executor
# ----------------------
def execute_pipeline(query: str, context: str, language: str, deepthink: bool):
    """
    Executes the chat pipeline for a user query.
    Animates the assistant's response while keeping alignment correct.
    """
    mode_badge = "🧠 Deep Thinking" if deepthink else "⚡ Quick Answer"

    # 1️⃣ Append empty assistant message first to keep alignment
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": "",  # will be filled during animation
        "badge": mode_badge
    })
    msg_index = len(st.session_state.chat_history) - 1

    final_answer = ""
    rag_result = {}

    try:
        if deepthink:
            # DeepThink multi-step reasoning
            thinking_prompt = build_thinking_prompt(query, context)
            thinking_text = query_models_with_fallbacks([MODEL_CHEAP] + MODEL_FALLBACKS, thinking_prompt)

            # Animate reasoning
            animated_thinking = ""
            for ch in thinking_text:
                animated_thinking += ch
                st.session_state.chat_history[msg_index]["content"] = f"**Thinking:** {animated_thinking}|"
                st.experimental_rerun()
                time.sleep(0.01)

            # Modular RAG final answer
            rag_result = modular_rag_smart_answer(context, query, lang=language)
            final_answer = rag_result.get("final", rag_result.get("error", "❌ Something went wrong."))

        else:
            # Quick vanilla RAG answer
            final_answer = vanilla_rag_answer(context, query, lang=language)
            rag_result = {
                "thinking": "",
                "primary": final_answer,
                "critique": "",
                "final": final_answer,
            }

        # Animate final answer
        animated_final = ""
        for c in final_answer:
            animated_final += c
            st.session_state.chat_history[msg_index]["content"] = animated_final
            st.rerun()
            time.sleep(0.004)

        # Ensure final answer is fully written
        st.session_state.chat_history[msg_index]["content"] = final_answer

    except Exception as e:
        st.session_state.chat_history[msg_index]["content"] = f"❌ Error: {e}"
        final_answer = f"Error: {e}"
        rag_result = {"final": final_answer}

    return final_answer, rag_result, mode_badge

# ----------------------
# Clarification detector
# ----------------------
def is_vague_query(query: str) -> bool:
    """
    Detects whether the query is vague / clarification-based rather than a new topic.
    Returns True if it's likely a clarification query (e.g., 'explain again', 'I didn’t get it').
    """
    q = query.lower().strip()

    # Very short queries (1–5 words) are often vague
    if len(q.split()) <= 5:
        return True

    # Explicit clarification patterns
    clarification_patterns = [
        r"\b(explain|repeat|rephrase|simplify|clarify)\b",
        r"\bi (did not|didn't|dont|don’t) understand\b",
        r"\bi (did not|didn't) get it\b",
        r"\btell me (again|differently)\b",
        r"\bwhat do you mean\b",
        r"\bmake (it|this) (simple|clear)\b"
    ]
    if any(re.search(p, q) for p in clarification_patterns):
        return True

    # If query is a generic question word + nothing else → vague
    generic_queries = {"what", "why", "how", "again", "repeat"}
    if q in generic_queries:
        return True

    # Otherwise, assume it's a new topic
    return False

# ----------------------
# 4️⃣ Chat Input Handler
# ----------------------
if user_query := st.chat_input("💬 Ask me about BITS Pilani Admission"):
    query = user_query.strip()

    if not query:
        st.warning("⚠️ Please type a question before submitting.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": query})

        try:
            docs = retriever.get_relevant_documents(query)
            faiss_context = "\n".join([doc.page_content for doc in docs]) if docs else ""
        except Exception as e:
            faiss_context = ""
            st.warning(f"⚠️ Retriever failed: {e}")

        context = (
            st.session_state.bitsadmission_content + "\n\n"
            + faiss_context + "\n\n"
            + (st.session_state.get("uploaded_content", "") or "")
        )

        use_deepthink = should_use_deepthink(query)

        if is_vague_query(query) and len(st.session_state.chat_history) > 0:
            last_assistant_msg = next(
                (m["content"] for m in reversed(st.session_state.chat_history)
                 if m["role"] == "assistant"),
                ""
            )
            clarification_prompt = build_clarification_prompt(last_assistant_msg, query, language)
            final_answer = query_models_with_fallbacks([MODEL_MID] + MODEL_FALLBACKS, clarification_prompt)
            mode_badge = "♻️ Clarification Mode"
        else:
            final_answer, _, mode_badge = execute_pipeline(query, context, language, use_deepthink)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": final_answer, "badge": mode_badge}
        )

        if "uid" in st.session_state:
            save_user_chat_history(st.session_state.uid, st.session_state.chat_history)

# ----------------------
# Display chat history
# ----------------------
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        if chat["role"] == "assistant" and "badge" in chat:
            st.markdown(
                f"""
                <div style="padding:6px 12px; background-color:#f0f5ff;
                            border-left:4px solid #2f54eb; border-radius:4px;
                            display:inline-block; margin-bottom:6px;">
                    {chat['badge']}
                </div>
                """,
                unsafe_allow_html=True
            )
        st.markdown(chat["content"], unsafe_allow_html=True)

# ----------------- Footer -----------------
st.markdown(
    """
    <style>
        .footer {
            background: linear-gradient(to right, red 33.3%, lightblue 33.3% 66.6%, yellow 66.6%);
            padding: 20px 0;
            text-align: center;
            color: #222;
            font-size: 14px;
        }
        .footer a {
            color: inherit;
            text-decoration: none;
            font-weight: bold;
        }
    </style>

    <div class="footer">
        Built with ❤️ by <b>BITS Pilani</b> · Pilani Campus ·
        <br>📬 Email: <a href="mailto:f20240347@pilani.bits-pilani.ac.in">Contact us</a>
    </div>
    """,
    unsafe_allow_html=True,
)

