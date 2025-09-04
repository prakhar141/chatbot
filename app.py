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

# LangChain/FAISS imports
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

import firebase_admin
from firebase_admin import credentials, auth, db

# ========== CONFIG (tweak these models per your OpenRouter access) ==========
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or "YOUR_API_KEY"
MODEL_CHEAP = os.getenv("MODEL_CHEAP") or "deepseek/deepseek-chat-v3-0324:free"
MODEL_MID = os.getenv("MODEL_MID") or "openai/gpt-oss-20b:free"
MODEL_HIGH = os.getenv("MODEL_HIGH") or "deepseek/deepseek-r1-0528:free"
MODEL_FALLBACKS = [MODEL_MID, MODEL_CHEAP]

EMBED_MODEL = os.getenv("EMBED_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
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
                                       "Use emojis, be concise and helpful. Provide actionable steps if relevant.Answer questions which are relevanto bits only.otherwise politely tell ur capabilities")},
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
                                       f"revise your original answer. Be clear and concise in {lang}.") },
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
        {"role": "system", "content": f"You are BitsBuddy, a helpful BITS assistant.Never guess or make up facts. Answer ONLY if the question is directly related to BITS Pilani,otherwise tell ur capabilities politely. Answer clearly in {lang}."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
    ]
    try:
        return query_models_with_fallbacks([MODEL_MID] + MODEL_FALLBACKS, prompt)
    except Exception as e:
        return f"⚠️ Error generating answer: {e}"

# ----------------- Session init -----------------
if "authenticated" in st.session_state and st.session_state["authenticated"]:
    if "chat_history" not in st.session_state:
        uid = st.session_state.get("user_uid")
        st.session_state.chat_history = load_user_chat_history(uid) if uid else []
    if "just_streamed" not in st.session_state:
        st.session_state.just_streamed = False
else:
    # show login screen if not authenticated (define login_screen elsewhere or reuse your function)
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

# ----------------- Main chat handler (auto pipeline selection) -----------------
import time
import streamlit as st
from sentence_transformers import SentenceTransformer, util

st.title(f"Welcome {st.session_state.get('user_name', 'User')} 👋")

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
# 4️⃣ Chat input handler
# ----------------------
if user_query := st.chat_input("Ask me about BITS Pilani anything"):
    query = user_query.strip()
    if not query:
        st.warning("Please type a question.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Retrieve context
        try:
            docs = retriever.get_relevant_documents(query)
            context = "\n".join([doc.page_content for doc in docs]) if docs else (
                st.session_state.get("uploaded_content", "") or ""
            )
        except Exception as e:
            context = st.session_state.get("uploaded_content", "") or ""
            st.warning(f"Retriever failed: {e}")

        # Decide pipeline automatically using semantic DeepThink
        use_deepthink = should_use_deepthink(query)

        # Execute selected pipeline
        final_answer, rag_result, mode_badge = execute_pipeline(
            query, context, language, use_deepthink
        )

        st.session_state.chat_history.append({"role": "assistant", "content": final_answer})
        st.session_state.just_streamed = True

        # Save chat to Firebase if logged in
        if "uid" in st.session_state:
            save_user_chat_history(st.session_state.uid, st.session_state.chat_history)
# ----------------- Display chat history (non-streamed older messages) -----------------
if st.session_state.just_streamed and len(st.session_state.chat_history) > 0:
    history_to_show = st.session_state.chat_history[:-1]
else:
    history_to_show = st.session_state.chat_history

for chat in (history_to_show):
    with st.chat_message("user" if chat.get("role") == "user" else "assistant"):
        st.markdown(chat.get("content", ""))

if st.session_state.just_streamed:
    st.session_state.just_streamed = False

# ----------------- Sidebar history preview -----------------

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

