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
MODEL_NAME = "deepseek/deepseek-chat-v3-0324:free"

# ========== UI Setup ==========
st.set_page_config(page_title="📄 Quiliffy", layout="wide")
st.title("🎓 Welcome to Quiliffy")
st.markdown("Ask anything like Bhawan Guide, Events, Clubs")

# ========== Reset Button ==========
if st.button("🔁 Reset Chat"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.experimental_rerun()

# ========== Load PDFs from Current Directory ==========
@st.cache_resource(show_spinner="📚 Preparing... Please wait.")
def build_vector_db_from_folder(folder_path="."):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)

            file_size_kb = round(os.path.getsize(file_path) / 1024, 2)
            with fitz.open(file_path) as doc:
                text = "\n".join([page.get_text() for page in doc])
                page_count = len(doc)
            #st.markdown(f"✅ Loaded **{filename}** — {file_size_kb} KB, {page_count} pages")

            chunks = splitter.split_text(text)
            file_docs = [Document(page_content=chunk, metadata={"source": filename}) for chunk in chunks]
            docs.extend(file_docs)

    embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
    vectordb = FAISS.from_documents(docs, embedder)
    return vectordb.as_retriever(search_type="similarity", k=4)

# ========== Ask Function ==========
def ask_deepseek(context, query):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://chat.openai.com",
        "X-Title": "PDF Chatbot"
    }
    messages = [
        {"role": "system", "content": "You're Quiliffy, a witty and knowledgeable assistant from BITS Pilani. Answer questions based on the context provided, but keep your tone light, humorous, and engaging — like a cool senior guiding a confused fresher. If the question is silly, respond playfully. Be helpful, but never boring.sprinkle in relevant emojis to make your replies fun and engaging."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    payload = {"model": MODEL_NAME, "messages": messages}
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ API Error: {e}"

# ========== Main ==========
pdf_files = [f for f in os.listdir(".") if f.endswith(".pdf")]
if not pdf_files:
    st.warning("⚠️ No PDF files found in current directory.")
    st.stop()

retriever = build_vector_db_from_folder()

# ========== Chat State ==========
if "chat" not in st.session_state:
    st.session_state.chat = []

query = st.chat_input("💬 Ask something about the BITS…")

if query:
    with st.spinner("🤖 Thinking..."):
        try:
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            answer = ask_deepseek(context, query)
        except Exception as e:
            answer = f"❌ Error: {e}"
        st.session_state.chat.append({
            "question": query,
            "answer": answer,
            "sources": list(set([doc.metadata['source'] for doc in docs]))
        })

# ========== Display Chat ==========
for chat in reversed(st.session_state.chat):
    with st.chat_message("user"):
        st.markdown(chat["question"])
    with st.chat_message("assistant"):
        st.markdown(chat["answer"])
        for src in chat["sources"]:
            st.caption(f"📄 Source: `{src}`")

# ========== Expandable Chat History ==========
with st.expander("📜 Full Chat History"):
    for i, chat in enumerate(st.session_state.chat):
        st.markdown(f"**Q{i+1}:** {chat['question']}")
        st.markdown(f"**A{i+1}:** {chat['answer']}")
        st.markdown("---")

# ========== Footer ==========
st.markdown("""
<hr style="margin-top: 40px;">
<div style='text-align: center; color: #888; font-size: 14px;'>
    Built with ❤️ by <b>Prakhar Mathur</b> · BITS Pilani · 
    <br>📬 <a href="mailto:prakhar.mathur2020@gmail.com">prakhar.mathur2020@gmail.com</a>
</div>
""", unsafe_allow_html=True)
