import os, time, fitz, requests
from PIL import Image
import streamlit as st
import pytesseract
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ========== CONFIG ==========
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or "YOUR_API_KEY"
MODEL_NAME = "deepseek/deepseek-chat-v3-0324:free"
EMBED_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
K_VAL = 4

st.set_page_config(page_title="🤖 BITS ReAct Buddy", layout="wide")
st.title("🎓 BITS ReAct Buddy")
st.markdown("An agent that thinks like a senior — ask me anything about BITS Pilani!")

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🔁 Start New Chat"):
        st.session_state.clear()
        st.rerun()

    uploaded_file = st.file_uploader("📄 Upload PDF or image", type=["pdf", "png", "jpg", "jpeg"])
    language = st.selectbox("🌐 Response Language", ["English", "Hindi", "Telugu", "Tamil", "Marathi", "Bengali"])

# ========== FILE PROCESSING ==========
uploaded_content = ""
uploaded_filename = uploaded_file.name if uploaded_file else ""

if uploaded_file:
    file_type = uploaded_file.type
    if file_type == "application/pdf":
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            uploaded_content = "\n".join(page.get_text() for page in doc)
    elif "image" in file_type:
        uploaded_content = pytesseract.image_to_string(Image.open(uploaded_file))

    if uploaded_content.strip():
        st.success("✅ Extracted content from file.")
        st.text_area("📄 Preview", uploaded_content[:1000], height=200)
    else:
        st.warning("⚠️ Couldn't extract readable text.")

# ========== VECTOR DB ==========
@st.cache_resource(show_spinner="🔍 Indexing documents...")
def load_vector_db(folder="."):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            with fitz.open(os.path.join(folder, file)) as doc:
                text = "\n".join(page.get_text() for page in doc)
                chunks = splitter.split_text(text)
                docs.extend([Document(page_content=c, metadata={"source": file}) for c in chunks])
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = FAISS.from_documents(docs, embedder)
    return vectordb.as_retriever(search_type="similarity", k=K_VAL)

retriever = load_vector_db()

# ========== REACT + RAG ==========
def react_rag_agent(question, lang="English", history=[]):
    system_prompt = (
        f"You're BitsBuddy, a smart and funny BITSian senior who uses ReAct (Reasoning + Action). "
        f"Answer in {lang}. Be informal, emoji-loving, and context-aware. "
        f"You can THINK, then SEARCH (retrieve), then OBSERVE, and finally ANSWER."
    )

    messages = [{"role": "system", "content": system_prompt}]
    for h in history[-3:]:
        messages.append({"role": "user", "content": h["question"]})
        messages.append({"role": "assistant", "content": h["answer"]})
    messages.append({
        "role": "user",
        "content": (
            f"Question: {question}\n"
            "Use the format:\n"
            "Thought: ...\n"
            "Action: Search[query]\n"
            "Observation: ...\n"
            "Thought: ...\n"
            "Final Answer: ..."
        )
    })

    try:
        # Initial ReAct step
        res = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "https://chat.openai.com",
                "X-Title": "ReAct RAG Agent"
            },
            json={"model": MODEL_NAME, "messages": messages}
        )
        res.raise_for_status()
        model_response = res.json()["choices"][0]["message"]["content"]

        # Extract search queries and rerun with document context
        if "Search[" in model_response:
            search_queries = []
            lines = model_response.splitlines()
            for line in lines:
                if line.strip().startswith("Action: Search["):
                    q = line.strip().split("Search[")[1].rstrip("]").strip("]")
                    search_queries.append(q)

            context = ""
            for sq in search_queries:
                docs = retriever.get_relevant_documents(sq)
                context += "\n\n".join([doc.page_content for doc in docs])

            # Final answer based on reasoning + retrieved context
            messages.append({
                "role": "assistant",
                "content": model_response
            })
            messages.append({
                "role": "user",
                "content": f"Use this retrieved context to write the final answer:\n{context}"
            })

            final = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "HTTP-Referer": "https://chat.openai.com",
                    "X-Title": "Final Answer Generator"
                },
                json={"model": MODEL_NAME, "messages": messages}
            )
            final.raise_for_status()
            return final.json()["choices"][0]["message"]["content"]
        else:
            return model_response

    except Exception as e:
        return f"❌ Error: {e}"

# ========== CHAT SESSION ==========
if "chat" not in st.session_state:
    st.session_state.chat = []

query = st.chat_input("💬 Ask anything about BITS Pilani...")
if query:
    with st.spinner("🧠 Thinking like a senior..."):
        answer = react_rag_agent(query, lang=language, history=st.session_state.chat)
        st.session_state.chat.append({
            "question": query,
            "answer": answer,
            "language": language
        })

# ========== CHAT DISPLAY ==========
for idx, chat in enumerate(reversed(st.session_state.chat)):
    with st.chat_message("user"):
        st.markdown(chat["question"])

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        animated = ""
        for c in chat["answer"]:
            animated += c
            response_placeholder.markdown(animated + "|")
            time.sleep(0.005)
        response_placeholder.markdown(animated)

# ========== CHAT HISTORY ==========
with st.sidebar:
    st.subheader("🗂️ Chat History")
    for i, chat in enumerate(reversed(st.session_state.chat)):
        st.markdown(f"**Q{i+1}:** {chat['question']}")
        st.markdown(f"**A{i+1}:** {chat['answer'][:150]}...")
        st.markdown("---")

# ========== FOOTER ==========
st.markdown("""
<hr style="margin-top: 40px;">
<div style='text-align: center; color: #888; font-size: 14px;'>
    Built with 🧠 by <b>Prakhar Mathur</b> · BITS Pilani · 
    <br>📬 Email: <a href="mailto:f20240347@pilani.bits-pilani.ac.in">Contact Prakhar</a>
</div>
""", unsafe_allow_html=True)
