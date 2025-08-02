import os
import fitz  # PyMuPDF
import streamlit as st
import requests
from PIL import Image
import pytesseract
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import time

# ========== CONFIG ========== #
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or "YOUR_API_KEY"
MODEL_NAME = "deepseek/deepseek-chat-v3-0324:free"
EMBED_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
K_VAL = 4

st.set_page_config(page_title="📄 BITS Pilani", layout="wide")
st.title("🎓 BITS Buddy")
st.markdown("Ask me anything about Bhawans, Clubs, Events, Professors, or Campus Life!")

# ========== Sidebar Upload & Settings ========== #
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🔁 Start New Chat"):
        st.session_state.clear()
        st.rerun()

    st.subheader("📄 Upload PDF or Image")
    uploaded_file = st.file_uploader("Upload a PDF or image", type=["pdf", "png", "jpg", "jpeg"])

    st.subheader("🌐 Choose Response Language")
    language = st.selectbox("Select language for response", ["English", "Hindi", "Telugu", "Tamil", "Marathi", "Bengali"])

# ========== File Processing ========== #
uploaded_content = ""
uploaded_filename = ""

if uploaded_file:
    file_type = uploaded_file.type
    uploaded_filename = uploaded_file.name

    if file_type == "application/pdf":
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            uploaded_content = "\n".join(page.get_text() for page in doc)
    elif "image" in file_type:
        image = Image.open(uploaded_file)
        uploaded_content = pytesseract.image_to_string(image)

    if uploaded_content.strip():
        st.success("✅ Extracted content from uploaded file.")
        st.text_area("📄 File Content Preview", uploaded_content[:1000], height=200)

        confirm = st.button("🤔 Check if this file is about BITS Pilani")

        if confirm:
            result = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "HTTP-Referer": "https://chat.openai.com",
                    "X-Title": "Check BITS relevance"
                },
                json={
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": "You're a BITS Pilani expert. Just say Yes or No."},
                        {"role": "user", "content": f"Content:\n{uploaded_content}\n\nIs this about BITS Pilani?"}
                    ]
                }
            )
            verdict = result.json()["choices"][0]["message"]["content"].strip().lower()

            if "yes" in verdict:
                st.success("🎉 This file is about BITS Pilani!")
                if file_type == "application/pdf":
                    save_path = os.path.join(".", uploaded_filename)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.info("📁 File saved and will be used in future context retrieval.")
            else:
                st.warning("❌ This file doesn’t seem to be about BITS Pilani.")
    else:
        st.warning("⚠️ No readable text found in the file.")

# ========== Vector DB ========== #
@st.cache_resource(show_spinner="🔍 Indexing PDFs...")
def load_pdfs(folder="."):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)
            with fitz.open(path) as doc:
                text = "\n".join(page.get_text() for page in doc)
                chunks = splitter.split_text(text)
                docs.extend([Document(page_content=c, metadata={"source": file}) for c in chunks])
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = FAISS.from_documents(docs, embedder)
    return vectordb.as_retriever(search_type="similarity", k=K_VAL)

# ========== Ask Function ========== #
def ask_deepseek(context, query, lang="English", history=[]):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://chat.openai.com",
        "X-Title": "PDF Chatbot"
    }
    messages = [
        {"role": "system", "content": f"You're BitsBuddy, a witty and helpful BITSian senior. Answer in {lang}. Use only the given context. Be friendly, emoji-rich, and informal. Be strict but kind if rules are broken. Don't allow excuses unless the document permits."}
    ]
    for h in history[-3:]:
        messages.append({"role": "user", "content": h["question"]})
        messages.append({"role": "assistant", "content": h["answer"]})
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})
    payload = {"model": MODEL_NAME, "messages": messages}

    try:
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ API Error: {e}"

# ========== Chat Setup ========== #
pdf_files = [f for f in os.listdir(".") if f.endswith(".pdf")]
if not pdf_files:
    st.error("⚠️ No PDF files found. Please upload them to the current directory.")
    st.stop()

retriever = load_pdfs()

if "chat" not in st.session_state:
    st.session_state.chat = []

query = st.chat_input("💬 Ask something about BITS Pilani...")
if query:
    with st.spinner("🤖 Thinking..."):
        try:
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            history = st.session_state.chat
            answer = ask_deepseek(context, query, lang=language, history=history)
            sources = list(set(doc.metadata['source'] for doc in docs))
        except Exception as e:
            answer = f"❌ Error: {e}"
            sources = []

        st.session_state.chat.append({
            "question": query,
            "answer": answer,
            "sources": sources
        })

# ========== Animated Display + Regenerate ========== #
for idx, chat in enumerate(reversed(st.session_state.chat)):
    with st.chat_message("user"):
        st.markdown(chat["question"])
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        for chunk in chat["answer"]:
            full_response += chunk
            response_placeholder.markdown(full_response + "|")
            time.sleep(0.01)
        response_placeholder.markdown(full_response)

        with st.expander("📄 Sources"):
            for src in chat["sources"]:
                st.markdown(f"**`{src}`**")

        if st.button(f"🔄 Regenerate Answer {idx+1}"):
            docs = retriever.get_relevant_documents(chat["question"])
            context = "\n\n".join([doc.page_content for doc in docs])
            new_answer = ask_deepseek(context, chat["question"], lang=language)
            chat["answer"] = new_answer
            st.rerun()

# ========== Sidebar History ========== #
with st.sidebar:
    st.subheader("📚 Chat History")
    for i, chat in enumerate(reversed(st.session_state.chat)):
        st.markdown(f"**Q{i+1}:** {chat['question']}")
        st.markdown(f"**A{i+1}:** {chat['answer']}")
        st.markdown("---")

# ========== Footer ========== #
st.markdown("""
<hr style="margin-top: 40px;">
<div style='text-align: center; color: #888; font-size: 14px;'>
    Built with ❤️ by <b>Prakhar Mathur</b> · BITS Pilani · 
    <br>📬 Email: <a href="mailto:f20240347@pilani.bits-pilani.ac.in">Contact Prakhar</a>
</div>
""", unsafe_allow_html=True)
