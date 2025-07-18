import os
import fitz  # PyMuPDF
import streamlit as st
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ========== OpenAI (ChatAnywhere) Setup ==========
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY") or "sk-REPLACE_WITH_YOUR_KEY",
    base_url="https://api.chatanywhere.tech/v1"
)

# ========== UI Setup ==========
st.set_page_config(page_title="🎓 Quillify", page_icon="🤖", layout="wide")
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

@st.cache_resource(show_spinner="📚 Indexing...")
def setup_vector_db():
    documents = load_all_pdfs()
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
    vectordb = FAISS.from_documents(documents, embeddings)
    return vectordb.as_retriever(search_type="similarity", k=2)

retriever = setup_vector_db()

# ========== OpenAI Answering ==========
def get_answer(query):
    context_docs = retriever.get_relevant_documents(query)
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    
    prompt = f"""You are a helpful assistant answering questions based on BITS Pilani PDFs.

Answer the following question using only the given context.

Context:
{context_text}

Question: {query}
"""

    # 🔍 Show the prompt
    with st.expander("🧠 Prompt sent"):
        st.code(prompt)

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant answering questions about BITS Pilani based on context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
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
        🤖 Built with ❤️ by <b>Prakhar Mathur</b> · BITS Pilani ·
    </div>
""", unsafe_allow_html=True)
