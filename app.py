import os
import fitz  # PyMuPDF
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import TooManyRequests

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
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    for file in os.listdir():
        if file.endswith(".pdf"):
            with fitz.open(file) as doc:
                full_text = "\n".join([page.get_text() for page in doc])
            chunks = splitter.split_text(full_text)
            docs.extend([Document(page_content=chunk, metadata={"source": file}) for chunk in chunks])
    return docs

@st.cache_resource(show_spinner="📚 Indexing PDFs...")
def setup_vector_db():
    documents = load_all_pdfs()
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
    vectordb = FAISS.from_documents(documents, embeddings)
    return vectordb.as_retriever(search_type="similarity", k=4)

# ========== Gemini LLM Loading ==========
@st.cache_resource(show_spinner="🤖 Warming up LLM...")
def load_llm():
    try:
        return ChatGoogleGenerativeAI(
            model="models/gemini-1.5-flash",
            google_api_key=st.secrets["GOOGLE_API_KEY"],
            temperature=0.3
        )
    except TooManyRequests:
        st.error("😵 Gemini API quota exceeded or rate limited. Please try later.")
        st.stop()

# ========== Retrieval Chain ==========
retriever = setup_vector_db()
llm = load_llm()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ========== User Tracking ==========
user_id = st.experimental_user.get("email", "anonymous_user")
if "user_log" not in st.session_state:
    st.session_state.user_log = set()
if user_id not in st.session_state.user_log:
    st.session_state.user_log.add(user_id)
    st.info(f"👋 New user session started: `{user_id}`")

# ========== Chat UI ==========
if "chat" not in st.session_state:
    st.session_state.chat = []

query = st.chat_input("💬 Ask me anything about BITS...")

if query:
    with st.spinner("🤖 Thinking..."):
        try:
            answer = qa_chain.run(query)
        except Exception as e:
            answer = f"❌ Failed to process your query due to: `{str(e)}`"
        st.session_state.chat.append({"question": query, "answer": answer})

# ========== Display Chat History ==========
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
