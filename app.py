import os
import time
import json
import math
import sqlite3
import hashlib
import threading
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import requests
from PIL import Image
import fitz  # PyMuPDF

# Optional — used for simple keyword fallback search
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# LangChain community vectorstore and embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ========== CONFIG (env / defaults) ==========
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_API_KEY")
MODEL_CHEAP = os.getenv("MODEL_CHEAP", "deepseek/deepseek-chat-v3-0324:free")
MODEL_MID = os.getenv("MODEL_MID", "openai/gpt-oss-20b:free")
MODEL_HIGH = os.getenv("MODEL_HIGH", "deepseek/deepseek-r1-0528:free")
MODEL_FALLBACKS = [MODEL_MID, MODEL_CHEAP]
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
K_DEFAULT = int(os.getenv("K_VAL", "4"))
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "./bits_buddy_cache.db")
ENABLE_PERSISTENT_CACHE = True
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")

# ========== STREAMLIT UI SETUP ==========
st.set_page_config(page_title="BITS Buddy", layout="wide")
st.title("🎓 BITS Buddy")
st.markdown("Ask me anything about BITS Pilani")

with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🔁 Start New Chat"):
        st.session_state.clear()
        st.experimental_rerun()

    uploaded_file = st.file_uploader("📄 Upload PDF or image", type=["pdf"])
    language = st.selectbox("🌐 Response Language", ["English", "Hindi", "Telugu", "Tamil", "Marathi", "Bengali"], index=0)
    st.checkbox("Enable Persistent SQLite Cache", value=ENABLE_PERSISTENT_CACHE, key="enable_sqlite")
    st.checkbox("Enable Clarifying Follow-ups", value=True, key="enable_clarify")
    st.markdown("---")
    st.write("Model tiering (env to change):")
    st.write(f"cheap: {MODEL_CHEAP}\nmid: {MODEL_MID}\nhigh: {MODEL_HIGH}")

# ========== SQLITE CACHE HELPERS ==========
_sql_conn: Optional[sqlite3.Connection] = None

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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            doc_id TEXT PRIMARY KEY,
            embedding BLOB,
            text_snippet TEXT,
            ts REAL
        )
    """)
    conn.commit()
    return conn

if ENABLE_PERSISTENT_CACHE:
    try:
        _sql_conn = init_sqlite(SQLITE_DB_PATH)
    except Exception as e:
        st.warning(f"Could not init sqlite: {e}")
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

# ========== SIMPLE IN-MEMORY LRU CACHE ==========
if 'lru_cache' not in st.session_state:
    st.session_state.lru_cache = {}

CACHE_MAX = 3000

def _lru_set(key: str, value: Any):
    if len(st.session_state.lru_cache) >= CACHE_MAX:
        # evict oldest
        oldest = min(st.session_state.lru_cache.items(), key=lambda kv: kv[1]['ts'])[0]
        st.session_state.lru_cache.pop(oldest, None)
    st.session_state.lru_cache[key] = {'value': value, 'ts': time.time()}


def _lru_get(key: str) -> Optional[Any]:
    v = st.session_state.lru_cache.get(key)
    return v['value'] if v else None

# ========== UTIL FUNCTIONS ==========

def make_cache_key(*args) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(args, sort_keys=True, ensure_ascii=False).encode('utf-8'))
    return digest.hexdigest()


def approx_complexity_score(question: str) -> float:
    """Estimate complexity by number of words and presence of open-ended tokens."""
    words = question.split()
    score = len(words)
    open_tokens = ['why', 'how', 'explain', 'discuss', 'compare', 'benefit', 'advantages']
    if any(t in question.lower() for t in open_tokens):
        score *= 1.7
    return float(score)

# ========== DOCUMENT INGESTION & SEMANTIC CHUNKING ==========

@st.cache_resource
def load_documents_from_folder(folder='.'):
    docs = []
    for file in os.listdir(folder):
        if file.lower().endswith('.pdf'):
            try:
                with fitz.open(os.path.join(folder, file)) as doc:
                    pages = [page.get_text() for page in doc]
                    text = "\n\n".join(pages)
                    # semantic chunking: split by double newlines (paragraphs), then merge to target size
                    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                    chunks = []
                    buf = ''
                    for p in paragraphs:
                        if len(buf) + len(p) < 1200:
                            buf = (buf + '\n\n' + p).strip()
                        else:
                            chunks.append(buf.strip())
                            buf = p
                    if buf:
                        chunks.append(buf.strip())

                    # add overlap
                    for c in chunks:
                        docs.append(Document(page_content=c, metadata={'source': file}))
            except Exception as e:
                st.warning(f"Could not read {file}: {e}")
    return docs

# Build and cache vector DB retriever
@st.cache_resource
def build_vector_retriever(docs: List[Document], embed_model=EMBED_MODEL, k=K_DEFAULT):
    if not docs:
        class EmptyRetriever:
            def get_relevant_documents(self, q): return []
        return EmptyRetriever()
    embedder = HuggingFaceEmbeddings(model_name=embed_model)
    vectordb = FAISS.from_documents(docs, embedder)
    return vectordb.as_retriever(search_type='similarity', k=k), embedder

# ========== MMR DIVERSITY SELECTION ==========

def mmr_select(query_embedding, candidate_embeddings, candidate_docs, k=4, lambda_mult=0.6):
    """Simple MMR implementation.
    query_embedding: 1D list
    candidate_embeddings: list of 1D lists
    candidate_docs: list of Document
    returns selected docs (up to k)
    """
    import numpy as np
    if len(candidate_docs) <= k:
        return candidate_docs
    query_vec = np.array(query_embedding)
    cand = np.array(candidate_embeddings)
    # cosine similarities
    def cos(a, b):
        na = a / (np.linalg.norm(a) + 1e-12)
        nb = b / (np.linalg.norm(b) + 1e-12)
        return float(np.dot(na, nb))
    similarities = [cos(query_vec, c) for c in cand]
    selected = []
    selected_idx = []
    # first select most similar
    idx = int(np.argmax(similarities))
    selected_idx.append(idx)
    selected.append(candidate_docs[idx])
    # iteratively select
    for _ in range(min(k-1, len(candidate_docs)-1)):
        scores = []
        for j in range(len(candidate_docs)):
            if j in selected_idx:
                scores.append(-1e9)
                continue
            sim_q = similarities[j]
            sim_to_selected = max(cos(cand[j], cand[s]) for s in selected_idx)
            mmr_score = lambda_mult * sim_q - (1 - lambda_mult) * sim_to_selected
            scores.append(mmr_score)
        next_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        selected_idx.append(next_idx)
        selected.append(candidate_docs[next_idx])
    return selected

# ========== HYBRID RETRIEVAL: EMBEDDING + KEYWORD FALLBACK ==========

def hybrid_retrieve(retriever, embedder, docs: List[Document], query: str, k_base: int = K_DEFAULT) -> List[Document]:
    """Perform embedding retrieval and a lightweight keyword-based fallback. Then apply MMR."""
    # Adaptive k based on complexity
    score = approx_complexity_score(query)
    k = max(2, min(12, int(k_base * (1 + score/10))))

    # 1) semantic retrieval via vector retriever
    candidate_docs = retriever.get_relevant_documents(query)

    # 2) keyword fallback using TF-IDF if available
    if SKLEARN_AVAILABLE and len(docs) > 0:
        texts = [d.page_content for d in docs]
        tfidf = TfidfVectorizer(stop_words='english', max_features=2000)
        X = tfidf.fit_transform(texts)
        qv = tfidf.transform([query])
        sims = cosine_similarity(qv, X)[0]
        # pick top 6 keyword matches
        top_idxs = sims.argsort()[-6:][::-1]
        for i in top_idxs:
            candidate_docs.append(docs[i])

    # deduplicate preserving order
    seen = set()
    unique_candidates = []
    for d in candidate_docs:
        h = hashlib.sha1(d.page_content.encode('utf-8')).hexdigest()
        if h not in seen:
            seen.add(h)
            unique_candidates.append(d)

    # compute embeddings for candidates and apply MMR
    # use embedder to get embeddings for candidates
    texts = [d.page_content for d in unique_candidates]
    try:
        cand_embeddings = embedder.embed_documents(texts)
        q_emb = embedder.embed_query(query)
        selected = mmr_select(q_emb, cand_embeddings, unique_candidates, k=k, lambda_mult=0.7)
    except Exception:
        # fallback: just take top-k
        selected = unique_candidates[:k]
    return selected

# ========== OPENROUTER CALL WITH BACKOFF & CACHING ==========
HEADERS_BASE = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}


def query_openrouter_with_backoff(model: str, messages: List[Dict[str, str]], max_retries: int = 4, timeout: int = 30) -> str:
    key = make_cache_key(model, messages)
    # 1) In-memory
    cached = _lru_get(key)
    if cached:
        return cached
    # 2) sqlite
    if st.session_state.get('enable_sqlite', ENABLE_PERSISTENT_CACHE) and _sql_conn:
        s = sql_get(key)
        if s:
            _lru_set(key, s)
            return s

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
            if isinstance(data.get('choices'), list) and data['choices']:
                c = data['choices'][0]
                msg = c.get('message') or c.get('delta') or c
                content = msg.get('content') if isinstance(msg, dict) else str(msg)
            elif data.get('text'):
                content = data.get('text')
            else:
                content = json.dumps(data)

            _lru_set(key, content)
            if st.session_state.get('enable_sqlite', ENABLE_PERSISTENT_CACHE) and _sql_conn:
                try:
                    sql_set(key, model, messages, content)
                except Exception:
                    pass
            return content
        except requests.HTTPError as e:
            if str(e).find('429') >= 0:
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
    raise RuntimeError('Failed to get response from OpenRouter')


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

# ========== PROMPT BUILDERS ==========

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
        {"role": "system", "content": f"You are BitsBuddy, a BITSian senior. Answer in {lang}. Use emojis, be concise and helpful."},
        {"role": "user", "content": scratchpad_reasoning(context, question)}
    ]


def build_critic_prompt(context: str, question: str, answer: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are an honest critic checking the assistant’s answer for factual errors, incompleteness, or hallucinations. Keep critiques short."},
        {"role": "user", "content": f"Context:\n{(context[:1500] + '...') if len(context) > 1500 else context}\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\nCritique and list corrections:"}
    ]


def build_final_prompt(context: str, question: str, answer: str, critique: str, lang: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": f"You are BitsBuddy+ with self-evaluation enabled. Based on critique, revise your original answer. Be clear and concise in {lang}."},
        {"role": "user", "content": f"Original Answer:\n{answer}\n\nCritique:\n{critique}\n\nNow improve the answer accordingly."}
    ]

# ========== CONTEXT COMPRESSION (simple LLM-based summarization) ==========

def compress_context(embedder, long_context: str, budget_chars: int = 2000) -> str:
    """If context is very long, call a cheap model to summarize into budget_chars.
    We cache summarizations per context hash.
    """
    key = make_cache_key('compress', long_context[:5000])
    cached = _lru_get(key)
    if cached:
        return cached
    # cheap summarization prompt
    prompt = [
        {"role": "system", "content": "You are a concise summarizer. Produce an accurate short summary."},
        {"role": "user", "content": f"Summarize (max {budget_chars} chars):\n\n{long_context}"}
    ]
    try:
        out = query_models_with_fallbacks([MODEL_CHEAP] + MODEL_FALLBACKS, prompt)
    except Exception:
        out = long_context[:budget_chars]
    _lru_set(key, out)
    return out

# ========== MODULAR RAG PIPELINE ==========

def modular_rag_smart_answer(retriever, embedder, docs: List[Document], context_src: str, question: str, lang: str = 'English') -> Dict[str, Any]:
    """Stages: clarify (optional), thinking (cheap), primary (mid), critique (cheap), final (high).
    Retrieval is hybrid and uses MMR. Context compression is applied if context too long.
    """
    out = {}
    try:
        # 0) optionally clarify short ambiguous queries
        clarify_enabled = st.session_state.get('enable_clarify', True)
        if clarify_enabled and len(question.split()) < 5 and ('?' in question or any(w in question.lower() for w in ['which', 'who', 'when', 'where'])):
            # Ask user for one-line clarification — in a real flow we'd do it as a stream; here we return special key
            return {'clarify': True, 'clarify_question': 'Could you give a bit more detail? For example, what section or year do you mean?'}

        # 1) hybrid retrieve
        candidates = hybrid_retrieve(retriever, embedder, docs, question)
        context = '\n\n---\n\n'.join([f"Source: {d.metadata.get('source','unknown')}\n{d.page_content[:2000]}" for d in candidates])

        # 1b) compress context if too long
        if len(context) > 5000:
            compressed = compress_context(embedder, context, budget_chars=3000)
            context_for_llm = compressed
            out['context_compressed'] = True
        else:
            context_for_llm = context
            out['context_compressed'] = False

        # Thinking (cheap)
        thinking_msgs = build_thinking_prompt(question, context_for_llm)
        thinking = query_models_with_fallbacks([MODEL_CHEAP] + MODEL_FALLBACKS, thinking_msgs)
        out['thinking'] = thinking

        # Primary (mid)
        primary_msgs = build_primary_prompt(context_for_llm, question, lang)
        primary = query_models_with_fallbacks([MODEL_MID] + MODEL_FALLBACKS, primary_msgs)
        out['primary'] = primary

        # Critique (cheap)
        critique_msgs = build_critic_prompt(context_for_llm, question, primary)
        critique = query_models_with_fallbacks([MODEL_CHEAP] + MODEL_FALLBACKS, critique_msgs)
        out['critique'] = critique

        # If critique asks for more sources or we detect 'missing' in critique, attempt mid-retrieval
        if any(tok in critique.lower() for tok in ['missing', 'not found', 'insufficient', 'need more']):
            # do an extended retrieval with higher k
            extra = hybrid_retrieve(retriever, embedder, docs, question, k_base=12)
            extra_context = '\n\n---\n\n'.join([f"Source: {d.metadata.get('source','unknown')}\n{d.page_content[:2000]}" for d in extra])
            # append extra to primary context and re-run a lightweight primary revision (cheap)
            primary_msgs2 = build_primary_prompt(context_for_llm + '\n\n' + extra_context, question, lang)
            primary = query_models_with_fallbacks([MODEL_MID] + MODEL_FALLBACKS, primary_msgs2)
            out['primary_revised_with_extra'] = primary

        # Final (high quality)
        final_msgs = build_final_prompt(context_for_llm, question, primary, critique, lang)
        final = query_models_with_fallbacks([MODEL_HIGH] + MODEL_FALLBACKS, final_msgs)
        out['final'] = final
        out['sources'] = [d.metadata.get('source','unknown') for d in candidates]
        return out
    except Exception as e:
        return {'error': str(e)}

# ========== SESSION STATE INIT ==========
if 'chat' not in st.session_state:
    st.session_state.chat = []

# ========== LOAD DOCS & RETRIEVER ==========
all_docs = load_documents_from_folder('.')
retriever_resource = None
embedder_resource = None
if all_docs:
    retriever_resource, embedder_resource = build_vector_retriever(all_docs, EMBED_MODEL, k=K_DEFAULT)
else:
    class EmptyRetriever:
        def get_relevant_documents(self, q): return []
    retriever_resource = EmptyRetriever()

# ========== FILE UPLOAD HANDLING ==========
uploaded_content = ''
if uploaded_file:
    file_type = uploaded_file.type
    if file_type == 'application/pdf':
        try:
            with fitz.open(stream=uploaded_file.read(), filetype='pdf') as doc:
                uploaded_content = '\n'.join(page.get_text() for page in doc)
        except Exception as e:
            st.warning(f"PDF read error: {e}")
    elif 'image' in file_type:
        try:
            img = Image.open(uploaded_file)
            uploaded_content = '[Image content; enable OCR to extract text]'
        except Exception as e:
            st.warning(f"Image read error: {e}")

    if uploaded_content.strip():
        st.success('✅ Extracted content from file.')
        st.text_area('📄 Preview (first 2000 chars)', uploaded_content[:2000], height=250)
    else:
        st.warning("⚠️ Couldn't extract readable text from the file.")

# ========== CHAT UI & HANDLER ==========
query = st.chat_input("💬 Ask anything about BITS Pilani...")
if query:
    with st.chat_message('assistant'):
        placeholder = st.empty()
        try:
            # if we have uploaded content prefer that as primary source
            context_source = uploaded_content if uploaded_content else ''
            if not context_source:
                # run hybrid retrieval to build context
                candidates = hybrid_retrieve(retriever_resource, embedder_resource, all_docs, query)
                context_source = '\n\n---\n\n'.join([f"Source: {d.metadata.get('source','unknown')}\n{d.page_content}" for d in candidates])
            else:
                candidates = []

            # show quick "thinking" animation using cheap model
            thinking_prompt = build_thinking_prompt(query, context_source[:1500])
            try:
                thinking_text = query_models_with_fallbacks([MODEL_CHEAP] + MODEL_FALLBACKS, thinking_prompt)
            except Exception as e:
                thinking_text = 'Hmm... thinking'

            animated = ''
            for ch in thinking_text:
                animated += ch
                placeholder.markdown(f"**Thinking:** {animated}|")
                time.sleep(0.01)
            placeholder.markdown(f"**Thinking:** {animated}")

            # run the modular RAG pipeline
            rag_result = modular_rag_smart_answer(retriever_resource, embedder_resource, all_docs, context_source, query, lang=language)

            # handle clarify
            if rag_result.get('clarify'):
                placeholder.markdown(f"⚠️ Clarifying question: {rag_result['clarify_question']}")
                # store a partial chat record
                st.session_state.chat.append({'question': query, 'final': rag_result['clarify_question'], 'thinking': thinking_text})
            elif 'error' in rag_result:
                placeholder.markdown(f"❌ Error generating answer: {rag_result['error']}")
                st.session_state.chat.append({'question': query, 'final': f"Error: {rag_result['error']}", 'thinking': thinking_text})
            else:
                final_answer = rag_result.get('final', 'Sorry — something went wrong.')
                # stream final
                animated = ''
                for c in final_answer:
                    animated += c
                    placeholder.markdown(animated + '|')
                    time.sleep(0.003)
                placeholder.markdown(animated)

                record = {
                    'question': query,
                    'thinking': rag_result.get('thinking',''),
                    'primary': rag_result.get('primary',''),
                    'critique': rag_result.get('critique',''),
                    'final': final_answer,
                    'sources': rag_result.get('sources', [])
                }
                st.session_state.chat.append(record)

        except Exception as e:
            placeholder.markdown(f"❌ Unexpected error: {e}")
            st.session_state.chat.append({'question': query, 'final': f"Error: {e}", 'thinking': ''})

# ========== DISPLAY HISTORY ==========
for chat in reversed(st.session_state.chat):
    with st.chat_message('user'):
        st.markdown(chat.get('question', ''))
    with st.chat_message('assistant'):
        st.markdown(chat.get('final',''))
        if chat.get('sources'):
            st.markdown('\n**Sources:** ' + ', '.join(chat['sources']))

# ========== SIDEBAR HISTORY ==========
with st.sidebar:
    st.subheader('📂 Chat History')
    for i, chat in enumerate(reversed(st.session_state.chat)):
        st.markdown(f"**Q{i+1}:** {chat.get('question','')}")
        st.markdown(f"**A{i+1}:** {chat.get('final','')[:150]}...")
        st.markdown('---')

# ========== FOOTER ==========
st.markdown("""
<hr style="margin-top: 40px;">
<div style='text-align: center; color: #888; font-size: 14px;'>
    Built with ❤️ by <b>Prakhar Mathur</b> · BITS Pilani ·
    <br>📬 Email: <a href="mailto:f20240347@pilani.bits-pilani.ac.in">Contact Prakhar</a>
</div>
""", unsafe_allow_html=True)
