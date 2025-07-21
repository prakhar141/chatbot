import os
import fitz  # PyMuPDF
import streamlit as st
import requests
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ========== API Setup ==========
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or "YOUR_API_KEY"
MODEL_NAME = "deepseek/deepseek-chat:free"  # ✅ Free model

# ========== UI Setup ==========
st.set_page_config(page_title="📄 Quiliffy", layout="wide")
st.title("📘 Chat with your PDFs")
st.markdown("This app reads **all PDFs** from the `./pdfs/` folder. Ask anything!")

# ========== PDF Folder Setup ==========
PDF_FOLDER = "."

# ========== PDF Processing ==========
@st.cache_resource(show_spinner="📚 Reading & indexing all PDFs...")
def build_vector_db_from_all_pdfs(folder_path):
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            with fitz.open(file_path) as doc:
                text = "\n".join([page.get_text() for page in doc])
            chunks = splitter.split_text(text)
            docs = [Document(page_content=chunk, metadata={"source": filename}) for chunk in chunks]
            all_docs.extend(docs)

    embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
    vectordb = FAISS.from_documents(all_docs, embedder)
    return vectordb.as_retriever(search_type="similarity", k=4)

# ========== Chat Function ==========
def ask_deepseek(context, query):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://chat.openai.com",  # optional, for API usage tracking
        "X-Title": "PDF Chatbot"
    }
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    payload = {"model": MODEL_NAME, "messages": messages}
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

# ========== Load Vector DB ==========
pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
if not pdf_files:
    st.error("❌ No PDF files found in the current directory.")
    st.stop()

retriever = build_vector_db_from_all_pdfs(PDF_FOLDER)
st.success("✅ All PDFs indexed. You can now ask questions.")

# ========== Chat Interface ==========
if "chat" not in st.session_state:
    st.session_state.chat = []

query = st.chat_input("💬 Ask a question from the PDFs")

if query:
    with st.spinner("🤖 Thinking..."):
        try:
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            answer = ask_deepseek(context, query)
        except Exception as e:
            answer = f"❌ Error: {str(e)}"
        st.session_state.chat.append({"question": query, "answer": answer, "sources": docs})

# ========== Chat History ==========
for chat in reversed(st.session_state.chat):
    with st.chat_message("user"):
        st.markdown(chat["question"])
    with st.chat_message("assistant"):
        st.markdown(chat["answer"])
        for doc in chat["sources"]:
            st.caption(f"📄 Source: `{doc.metadata['source']}`")

# ========== Footer ==========
st.markdown("""
<hr style="margin-top: 40px;">
<div style='text-align: center; color: #888; font-size: 14px;'>
    Built with ❤️ by <b>Prakhar Mathur</b> · BITS Pilani · 
    <br>📬 Email: <a href="mailto:prakhar.mathur2020@gmail.com">prakhar.mathur2020@gmail.com</a>
</div>
""", unsafe_allow_html=True)
