import os
import fitz  # PyMuPDF
import streamlit as st
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ========== Load Env Vars ==========
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or "YOUR_API_KEY"
MODEL_NAME = "deepseek/deepseek-chat:free"

# ========== UI Setup ==========
st.set_page_config(page_title="🎓 Quillify", layout="wide")
st.markdown("""
    <style>
        .big-title { font-size: 36px; font-weight: 800; margin-bottom: 10px; color: #3B82F6; }
        .subtitle { font-size: 16px; color: gray; margin-top: -10px; }
        .stTextInput > div > div > input { font-size: 18px; }
    </style>
""", unsafe_allow_html=True)
st.markdown("<div class='big-title'>🎓 Quillify</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Ask anything about BITS – syllabus, events, academics, policies, and more</div>", unsafe_allow_html=True)

# ========== PDF Loading ==========
def load_all_pdfs():
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    for file in os.listdir():
        if file.endswith(".pdf"):
            with fitz.open(file) as doc:
                full_text = "\n".join([page.get_text() for page in doc])
            chunks = splitter.split_text(full_text)
            docs.extend([Document(page_content=chunk, metadata={"source": file}) for chunk in chunks])
    return docs

@st.cache_resource(show_spinner="📚 Reading...")
def setup_vector_db():
    documents = load_all_pdfs()
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
    vectordb = FAISS.from_documents(documents, embeddings)
    return vectordb.as_retriever(search_type="similarity", k=2)

retriever = setup_vector_db()

# ========== OpenRouter Answering ==========
def get_answer(query):
    context_docs = retriever.get_relevant_documents(query)
    context_text = "\n\n".join([doc.page_content for doc in context_docs])

    prompt = f"""You are a helpful assistant answering questions based on BITS Pilani documents.

Only use the provided context to answer. Do not hallucinate or make up data.

Context:
{context_text}

Question: {query}
"""

    with st.expander("🧠 Prompt sent"):
        st.code(prompt)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://chat.openai.com",  # optional; change for deployment
        "X-Title": "Quillify BITS Assistant"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a BITS Pilani assistant who answers accurately using only the given documents."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ========== User Tracking ==========
user_id = st.user.get("email", "anonymous_user")
if "user_log" not in st.session_state:
    st.session_state.user_log = set()
if user_id not in st.session_state.user_log:
    st.session_state.user_log.add(user_id)
    st.info(f"👋 New user session started: `{user_id}`")

# ========== Chat Interface ==========
if "chat" not in st.session_state:
    st.session_state.chat = []

query = st.chat_input("💬 Ask me anything about BITS...")

if query:
    with st.spinner("🤖 Thinking..."):
        answer = get_answer(query)
        st.session_state.chat.append({"question": query, "answer": answer})

# ========== Chat History ==========
for entry in reversed(st.session_state.chat):
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        st.markdown(entry["answer"])

# ========== Footer ==========
st.markdown("""
    <hr style="margin-top: 40px; margin-bottom: 10px;">
    <div style='text-align: center; color: #aaa; font-size: 14px;'>
         Built with ❤️ by <b>Prakhar Mathur</b> · BITS Pilani ·
    </div>
""", unsafe_allow_html=True)
