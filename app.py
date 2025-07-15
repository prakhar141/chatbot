import os
import fitz  # PyMuPDF
import streamlit as st
from typing import Optional, List
import google.generativeai as genai

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM

# ======= Custom Gemini LLM Wrapper =======
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

# ======= Load PDFs from Folder and Chunk =======
def load_pdfs_from_directory(folder_path):
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            chunks = splitter.split_text(text)
            docs = [Document(page_content=chunk, metadata={"source": filename}) for chunk in chunks]
            all_docs.extend(docs)
    return all_docs

# ======= Streamlit UI =======
st.set_page_config(page_title="📘 College ChatBot", page_icon="🤖")
st.title("🎓 College ChatBot")
st.markdown("Ask me anything from the college documents and get answers powered by Gemini!")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("💬 Ask a question:")
mode = st.selectbox("🔍 Mode", ["QA", "Summarize", "Keywords", "Generate Q&A"])

# ======= Load & Embed once =======
with st.spinner("📚 Loading documents..."):
    docs = load_pdfs_from_directory("data/")  # <== Put all your PDFs inside `data/` folder
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
    vectordb = FAISS.from_documents(docs, embeddings)
    retriever = vectordb.as_retriever(search_type="similarity", k=3)

llm = GeminiLLM(api_key=st.secrets["GEMINI_API_KEY"])
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ======= Run Query =======
if query:
    with st.spinner("🤖 Thinking..."):
        if mode == "QA":
            answer = qa_chain.run(query)
        elif mode == "Summarize":
            docs = retriever.get_relevant_documents(query)
            context = "\n".join([doc.page_content for doc in docs])
            prompt = f"Summarize this content:\n\n{context}"
            answer = llm._call(prompt)
        elif mode == "Keywords":
            docs = retriever.get_relevant_documents(query)
            context = "\n".join([doc.page_content for doc in docs])
            prompt = f"Extract important keywords:\n\n{context}"
            answer = llm._call(prompt)
        elif mode == "Generate Q&A":
            context = "\n".join([doc.page_content for doc in docs[:5]])
            prompt = f"Generate 5 question-answer pairs from this content:\n\n{context}"
            answer = llm._call(prompt)

        # Store & Display
        st.session_state.history.append((query, answer))
        st.success("✅ Answer:")
        st.markdown(answer)

        with st.expander("📄 Sources"):
            for doc in retriever.get_relevant_documents(query):
                st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                st.code(doc.page_content[:300])

# ======= Chat History =======
with st.expander("📜 Chat History"):
    for q, a in reversed(st.session_state.history):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.markdown("---")

# ======= Footer =======
st.markdown("""
<hr style="margin-top: 30px; margin-bottom: 10px;">
<div style='text-align: center; color: gray; font-size: 14px;'>
Built with ❤️ by <b>Prakhar Mathur</b> · BITS Pilani
</div>
""", unsafe_allow_html=True)
