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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

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

EMBED_MODEL = os.getenv("EMBED_MODEL") or "all-mpnet-base-v2"
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

st.markdown("Have a question about BITS Pilani admissions? Ask anytime.")

with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🧭 Start New Chat"):
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
EMBED_MODEL = "all-mpnet-base-v2"
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
            def invoke(self, query):
                return []

        return EmptyRetriever()

# ---------------- LOAD RETRIEVER ----------------
retriever = load_vector_db_from_hf()
from bs4 import BeautifulSoup

# ---------------------- 0️⃣ Real-time BITSAdmission content ----------------------
BITSADMISSION_URLS = [
    "https://www.bitsadmission.com/index.html",
    "https://www.bitsadmission.com/FD/FD.html","https://timesofindia.indiatimes.com/education/news/bitsat-2026-registration-to-begin-on-this-day-bits-pilani-announces-fresh-exam-schedule-details-here/articleshow/125927906.cms",
    "https://admissions.bits-pilani.ac.in/FD/BITSAT_cutOffs.html?06012025","https://admissions.bits-pilani.ac.in/FD/BITSAT_cutOffs.html?FQwp43qOeKhayi8LEQVUtJn3QNZ0TciWLP4NKxNMfcgzQdzcqZCCLqDBZRDnjcsHWFGgSC&yr=2024-2025&eKhayi8LEQwp4NKxN+CfCh+3qOVUtJn3QNZ0TciWLP4",
    "https://admissions.bits-pilani.ac.in/FD/BITSAT_cutOffs.html?FQwp43qOeKhayi8LEQVUtJn3QNZ0TciWLP4NKxNMfcgzQdzcqZCCLqDBZRDnjcsHWFGgSC&yr=2023-2024&eKhayi8LEQwp4NKxN+CfCh+3qOVUtJn3QNZ0TciWLP4",
    "https://admissions.bits-pilani.ac.in/FD/BITSAT_cutOffs.html?FQwp43qOeKhayi8LEQVUtJn3QNZ0TciWLP4NKxNMfcgzQdzcqZCCLqDBZRDnjcsHWFGgSC&yr=2022-2023&eKhayi8LEQwp4NKxN+CfCh+3qOVUtJn3QNZ0TciWLP4","https://cdn3.digialm.com/EForms/configuredHtml/1823/94103/Index.html?_gl=1*1lsubut*_gcl_au*MTUwMDQ5NzU0Ni4xNzY1NjA4MjIy","https://admissions.bits-pilani.ac.in/Privacy.html?_gl=1*bcbvph*_gcl_au*MTUwMDQ5NzU0Ni4xNzY1NjA4MjIy*_ga*NzI3OTE4MjcyLjE3NjU5NDcyOTY.*_ga_DYQ0HEBE5Z*czE3NjU5NDcyOTUkbzEkZzEkdDE3NjU5NDcyOTUkajYwJGwwJGgyMDIyNDI4MjAx",
    "https://admissions.bits-pilani.ac.in/FD/FD_brochure.html?06012025&_gl=1*gfndeo*_gcl_au*MTUwMDQ5NzU0Ni4xNzY1NjA4MjIy*_ga*NzI3OTE4MjcyLjE3NjU5NDcyOTY.*_ga_DYQ0HEBE5Z*czE3NjU5NDcyOTUkbzEkZzEkdDE3NjU5NDc0NTgkajYwJGwwJGgyMDIyNDI4MjAx",
    "https://cdn3.digialm.com/EForms/configuredHtml/1823/96992/Index.html?_gl=1*gscehk*_gcl_au*MTUwMDQ5NzU0Ni4xNzY1NjA4MjIy",
    "https://admissions.bits-pilani.ac.in/ISA/ISA.html"
]
def extract_numeric_chunks(
    text: str,
    min_numbers: int = 2,
    window: int = 600
) -> List[str]:
    """
    Extracts text windows that contain multiple numeric values.
    Useful for fee tables / form-based pages where embeddings fail.
    """
    numeric_chunks = []
    lines = text.splitlines()

    buffer = []
    num_count = 0

    for line in lines:
        buffer.append(line)
        num_count += len(re.findall(r"\d+", line))

        if len("\n".join(buffer)) >= window:
            if num_count >= min_numbers:
                numeric_chunks.append("\n".join(buffer))
            buffer = []
            num_count = 0

    # final remainder
    if buffer and num_count >= min_numbers:
        numeric_chunks.append("\n".join(buffer))

    return numeric_chunks
def retrieve_live_context(query: str, full_text: str, top_k: int = 4) -> str:
    """
    Retrieves relevant chunks from live website content.
    ENHANCED: also injects numeric-heavy chunks for factoid queries,
    with prioritization for form-based (Digialm / configuredHtml) pages.
    """

    # ----------------------------
    # Existing semantic logic (UNCHANGED)
    # ----------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_text(full_text)

    if not chunks:
        return ""

    query_emb = embed_model.encode(query, convert_to_tensor=True)
    chunk_embs = embed_model.encode(chunks, convert_to_tensor=True)

    scores = util.cos_sim(query_emb, chunk_embs)[0]
    top_indices = scores.topk(top_k).indices.tolist()

    semantic_context = "\n".join(chunks[i] for i in top_indices)

    # ----------------------------
    # NEW: numeric fallback logic (ADDED, old behavior preserved)
    # ----------------------------
    FACTOID_KEYWORDS = [
        "fee", "fees", "application fee",
        "exam fee", "registration fee",
        "amount", "payment"
    ]

    numeric_context = ""

    if any(k in query.lower() for k in FACTOID_KEYWORDS):
        numeric_chunks = extract_numeric_chunks(full_text)

        # Detect form-based Digialm pages
        is_form_page = (
            "configuredhtml" in full_text.lower()
            or "digialm" in full_text.lower()
        )

        # Preserve old behavior, enhance when form page detected
        if is_form_page:
            numeric_context = "\n\n".join(numeric_chunks[:3])
        else:
            numeric_context = "\n\n".join(numeric_chunks[:2])

    # ----------------------------
    # Merge contexts (UNCHANGED)
    # ----------------------------
    return "\n\n".join(
        part for part in [numeric_context, semantic_context] if part
    )

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
                                       "Use emojis, be concise and helpful. Provide actionable steps if relevant.You know only what’s in the provided CONTEXT — nothing else exists for you")},
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
                                       f"revise your original answer.Never invent or use outside knowledge.You know only what’s in the provided CONTEXT — nothing else exists for you. Be clear and concise in {lang}.") },
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
        {"role": "system", "content": f"You are BitsBuddy, a helpful BITS assistant.You know only what’s in the provided CONTEXT — nothing else exists for you. Answer clearly in {lang}."},
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
                f"Re-explain it clearly, differently, and simply. "
                f"Do NOT introduce new context or use outside information. "
                f"Just restate or simplify your last response. Answer in {lang}."
            )
        },
        {
            "role": "user",
            "content": f"Previous Answer:\n{last_answer}\n\nUser Query:\n{user_query}"
        }
    ]
# ----------------- Session init (robust logout) -----------------

# Helper: keys we'll clear on logout (covers both 'uid' styles used in your code)
_LOGOUT_KEYS = [
    "authenticated", "user_uid", "uid", "user_name", "chat_history",
    "just_streamed", "logout_requested", "do_logout", "logout_confirmed"
]

# If a logout was confirmed on a previous run, perform the actual cleanup first
if st.session_state.get("do_logout", False):
    # Optional: delete from Firebase permanently (uncomment if you want)
    # try:
    #     uid_to_delete = st.session_state.get("uid") or st.session_state.get("user_uid")
    #     if uid_to_delete:
    #         db.reference(f"user_chats/{uid_to_delete}").delete()
    # except Exception as e:
    #     st.warning(f"Failed to delete remote history: {e}")

    for _k in _LOGOUT_KEYS:
        if _k in st.session_state:
            st.session_state.pop(_k, None)
    # Immediately rerun so login screen shows
    st.rerun()

# Normal authenticated flow
if st.session_state.get("authenticated", False):

    # Put logout controls into sidebar
    with st.sidebar:
        st.markdown("---")
        st.header("🔒 Account")

        # Step 1: initial Logout button
        if st.button("🚪 Logout"):
            st.session_state["logout_requested"] = True
            st.rerun()  # show confirmation buttons immediately

        # Step 2: show explicit Confirm / Cancel buttons when requested
        if st.session_state.get("logout_requested", False):
            st.warning("You’re about to log out — nothing will be lost. Continue?") 
            col_yes, col_no = st.columns([1, 1])
            with col_yes:
                if st.button("Yes, sign me out"):
                    # mark for logout and rerun so cleanup happens at top of file
                    st.session_state["do_logout"] = True
                    # remove the request flag to avoid loops
                    st.session_state.pop("logout_requested", None)
                    st.rerun()

            with col_no:
                if st.button("No — Keep me signed in"):
                    # cancel the logout flow and rerun to clear confirmation UI
                    st.session_state.pop("logout_requested", None)
                    st.rerun()

    # Continue loading chat history / session initialization
    if "chat_history" not in st.session_state:
        uid = st.session_state.get("user_uid") or st.session_state.get("uid")
        st.session_state.chat_history = load_user_chat_history(uid) if uid else []
    if "just_streamed" not in st.session_state:
        st.session_state.just_streamed = False

else:
    # ----------------- Login Screen -----------------
    def login_screen():
        st.title("🔐 BITS Buddy Login")
        st.markdown("To continue, please log in or sign up — it’s quick and secure.")
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
                # keep both keys to stay compatible with other code paths
                st.session_state["user_uid"] = st.session_state.get("uid")
                st.session_state["user_name"] = name
                st.session_state["authenticated"] = True
                st.rerun()
            except Exception as e:
                st.error(f"Authentication failed: {e}")
                return False

    login_screen()
    st.stop()

# ----------------- Main chat handler (auto pipeline selection) -----------------

st.title(f"Which part of BITS admissions shall we tackle first? {st.session_state.get('user_name', 'User')} 👋")

# ----------------------
# 1️⃣ Load embedding model
# ----------------------
embed_model = SentenceTransformer("all-mpnet-base-v2")

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
def is_query_relevant_to_context(query: str, retriever, threshold_base: float = 0.45) -> bool:
    """
    🚨 Ultra-strict semantic relevance filter for domain control.
    Ensures the query is genuinely related to BITS Pilani Admissions.
    Combines semantic similarity, dispersion check, and adaptive thresholding.
    """

    try:
        # Fetch top results from FAISS / retriever
        docs = retriever.invoke(query)
        if not docs:
            return False

        # Encode query and top document embeddings
        query_emb = embed_model.encode(query, convert_to_tensor=True)
        doc_texts = [d.page_content for d in docs[:5]]
        doc_embs = embed_model.encode(doc_texts, convert_to_tensor=True)
        sim_scores = util.cos_sim(query_emb, doc_embs).cpu().numpy()[0]

        max_score = float(max(sim_scores))
        mean_score = float(sum(sim_scores) / len(sim_scores))
        score_std = float(
            (sum((x - mean_score) ** 2 for x in sim_scores) / len(sim_scores)) ** 0.5
        )

        # 🔧 Dynamic threshold tuning
        q_len = len(query.split())
        dyn_threshold = threshold_base + (0.08 if q_len < 5 else 0.0) - (0.03 if q_len > 15 else 0.0)
        dyn_threshold = max(0.4, min(0.55, dyn_threshold))  # tightened range

        # 🧠 Strict decision logic:
        # - Must have strong match (max_score)
        # - Multiple docs agree (mean_score)
        # - Consistency across docs (low std)
        if (
            max_score >= dyn_threshold
            and mean_score >= dyn_threshold * 0.9
            and score_std < 0.12
        ):
            return True
        else:
            return False

    except Exception as e:
        # 🚫 Fail-closed: block query if anything goes wrong
        st.warning(f"⚠️ Relevance check error: {e}")
        return False

# ----------------------
# 3️⃣ Modular pipeline executor
# ----------------------
def execute_pipeline(query: str, context: str, language: str, deepthink: bool):
    mode_badge = "🧠 Deep Thinking" if deepthink else "⚡ Quick Answer"
    placeholder = st.empty()
    final_answer = ""
    rag_result = {}

    try:
        placeholder.markdown(f"{mode_badge} — preparing response...")

        if deepthink:
            thinking_prompt = build_thinking_prompt(query, context)
            thinking_text = query_models_with_fallbacks([MODEL_CHEAP] + MODEL_FALLBACKS, thinking_prompt)

            # Animate reasoning output
            animated = ""
            for ch in thinking_text:
                animated += ch
                placeholder.markdown(f"{mode_badge}\n\n**Thinking:** {animated}|")
                time.sleep(0.01)
            placeholder.markdown(f"{mode_badge}\n\n**Thinking:** {animated}")

            # Modular RAG for final deep answer
            time.sleep(0.25)
            placeholder.markdown(f"{mode_badge}\n\n🔁 Reasoning...\n\n• ✏️ Drafting initial answer...")
            rag_result = modular_rag_smart_answer(context, query, lang=language)
            final_answer = rag_result.get("final", rag_result.get("error", "❌ Something went wrong."))

        else:
            # Vanilla RAG
            final_answer = vanilla_rag_answer(context, query, lang=language)
            rag_result = {
                "thinking": "",
                "primary": final_answer,
                "critique": "",
                "final": final_answer,
            }

        # Animate final answer
        animated = "|"
        for c in final_answer:
            animated += c
            placeholder.markdown(f"{mode_badge}\n\n{animated}|")
            time.sleep(0.004)
        placeholder.markdown(f"{mode_badge}\n\n{animated}")

    except Exception as e:
        placeholder.markdown(f"❌ Error: {e}")
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
# =========================================
# 🧠 Streamlit Chat Section
# =========================================

# Display chat input box
if user_query := st.chat_input("💬 Curious about BITS Pilani admissions? Ask me anything!"):
    query = user_query.strip()

    if not query:
        st.warning("⚠️ Oops! Looks like you forgot to type your question. What would you like to ask?")
    else:
        # 🗓 Save user query
        st.session_state.chat_history.append({
            "role": "user",
            "content": query,
            "ts": time.time()
        })

        # 🧱 Retrieve FAISS context safely
        try:
            docs = retriever.invoke(query)
            faiss_context = "\n".join([doc.page_content for doc in docs]) if docs else ""
        except Exception as e:
            faiss_context = ""
            st.warning(f"⚠️ Retriever failed: {e}")
        live_context=retrieve_live_context(query,st.session_state.bitsadmission_content)

        context = (
            live_context + "\n\n"
            + faiss_context + "\n\n"
            + (st.session_state.get("uploaded_content", "") or "")
        )

        # 🧭 Decide pipeline (normal vs deepthink)
        use_deepthink = should_use_deepthink(query)

        # =========================================
        # 🔒 Ultra-Strict Domain Relevance Filter
        # =========================================
        try:
            is_relevant = is_query_relevant_to_context(query, retriever, threshold_base=0.45)

            # 🔐 Keyword fallback (second layer of defense)
            admission_keywords = [
                "bitsat", "bits pilani", "bits", "admission", "cutoff",
                "iteration", "merit", "eligibility", "fees", "counselling",
                "scholarship", "branch", "placement", "campus", "application"
            ]
            keyword_match = any(k in query.lower() for k in admission_keywords)

            # 🚫 Block if neither semantic nor keyword match
            if not is_relevant and not keyword_match:
                final_answer = (
                    "⚠️ Hmm… I specialize in BITS Pilani admissions. It looks like your question is outside my data. 😊 "
                    "Could you try asking something about admissions?"
                )
                mode_badge = "🚫 Out-of-Domain Filter"

            else:
                # ✅ Only proceed for valid admission-related queries
                if is_vague_query(query) and len(st.session_state.chat_history) > 0:
                    last_assistant_msg = next(
                        (m["content"] for m in reversed(st.session_state.chat_history)
                         if m["role"] == "assistant"),
                        ""
                    )
                    clarification_prompt = build_clarification_prompt(
                        last_assistant_msg, query, language
                    )
                    final_answer = query_models_with_fallbacks(
                        [MODEL_MID] + MODEL_FALLBACKS, clarification_prompt
                    )
                    mode_badge = "♻️ Clarification Mode"
                else:
                    final_answer, _, mode_badge = execute_pipeline(
                        query, context, language, use_deepthink
                    )

        except Exception as e:
            # 🧤 Fail-safe — block everything if filter fails
            st.warning(f"⚠️ Domain relevance check failed: {e}")
            final_answer = (
                "⚠️ Hmm… I specialize in BITS Pilani admissions. It looks like your question is outside my data. 😊 "
                "Could you try asking something about admissions?"
            )
            mode_badge = "🚫 Safety Filter (Fail-Safe)"

        # =========================================
        # 💬 Save and Display Assistant Reply
        # =========================================
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": final_answer,
            "badge": mode_badge,
            "ts": time.time()
        })

        # 🔐 Optional Firebase Sync
        if "uid" in st.session_state:
            save_user_chat_history(st.session_state.uid, st.session_state.chat_history)

# ----------------------
# Display chat history in chronological order
# Use .get("ts", 0) to avoid KeyError
# ----------------------
for chat in sorted(st.session_state.chat_history, key=lambda m: m.get("ts", 0)):
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
# ----------------- Disclaimer -----------------
st.markdown(
    """
    <div style="background-color:#fff3cd; border-left:6px solid #ffecb5;
                padding:10px; border-radius:6px; margin-top:25px; font-size:15px;">
        ⚠️ <b>Important:</b> BITS Buddy is designed to assist, not replace official sources.
        For final and critical admission decisions, please confirm details on the
        <a href="https://www.bitsadmission.com" target="_blank">official BITS Admission website</a>.
    </div>
    """,
    unsafe_allow_html=True,
)

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
