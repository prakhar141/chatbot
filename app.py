import os
import fitz  # PyMuPDF
import streamlit as st
import google.generativeai as genai
from typing import Optional, List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM

# ======= Gemini Wrapper =======
class GeminiLLM(LLM):
    model: str = "gemini-1.5-flash"
    api_key: str = ""

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)
        response = model.generate_content(prompt)
        return response.text

    @property
    def _llm_type(self) -> str:
        return "custom-gemini"

# ======= Load all PDFs from current directory =======
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

# ======= UI Config =======
st.set_page_config(page_title="🎓 Quillify", page_icon="🤖", layout="wide")
st.markdown(
    """
    <style>
        .big-title { font-size: 36px; font-weight: 800; margin-bottom: 10px; color: #3B82F6; }
        .subtitle { font-size: 16px; color: gray; margin-top: -10px; }
        .stTextInput > div > div > input {
            font-size: 18px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("<div class='big-title'>🎓 Quillify</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Ask anything about BITS – syllabus, events, academics, policies, and more</div>", unsafe_allow_html=True)

# ======= Load Embeddings and Vector DB Once =======
@st.cache_resource(show_spinner="📚 Thinking...")
def setup_vector_db():
    documents = load_all_pdfs()
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
    vectordb = FAISS.from_documents(documents, embeddings)
    retriever = vectordb.as_retriever(search_type="similarity", k=4)
    return retriever

retriever = setup_vector_db()
llm = GeminiLLM(api_key=st.secrets["GEMINI_API_KEY"])
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ======= Chatbot Interaction =======
if "chat" not in st.session_state:
    st.session_state.chat = []

query = st.chat_input("💬 I know more about BITS than your CGPA does.")

if query:
    with st.spinner("🤖 Thinking..."):
        answer = qa_chain.run(query)
        st.session_state.chat.append({"question": query, "answer": answer})

# ======= Display Chat =======
for entry in reversed(st.session_state.chat):
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        st.markdown(entry["answer"])

# ======= Footer =======
st.markdown(
    """
    <hr style="margin-top: 40px; margin-bottom: 10px;">
    <div style='text-align: center; color: #aaa; font-size: 14px;'>
        🤖 Built with ❤️ by <b>Prakhar Mathur</b> · BITS Pilani ·
    </div>
    """,
    unsafe_allow_html=True
)
