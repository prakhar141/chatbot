# Rewriting your Streamlit app with Firebase Google Authentication integrated

import os
import time
import fitz
import requests
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ========== CONFIG ==========
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or "YOUR_API_KEY"
MODEL_NAME = "deepseek/deepseek-chat-v3-0324:free"
EMBED_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
K_VAL = 4

st.set_page_config(page_title="BITS Buddy", layout="wide")
st.title("🎓 BITS Buddy")
st.markdown("Ask me anything about BITS Pilani")

# ========== FIREBASE AUTH COMPONENT ==========
if "user_email" not in st.session_state:

    firebase_auth_html = """
    <!DOCTYPE html>
    <html>
      <head>
        <script src="https://www.gstatic.com/firebasejs/9.6.1/firebase-app.js"></script>
        <script src="https://www.gstatic.com/firebasejs/9.6.1/firebase-auth.js"></script>
        <script>
          const firebaseConfig = {
            apiKey: "YOUR_API_KEY",
            authDomain: "your-project.firebaseapp.com",
            projectId: "your-project",
          };
          firebase.initializeApp(firebaseConfig);

          function signInWithGoogle() {
            const provider = new firebase.auth.GoogleAuthProvider();
            firebase.auth().signInWithPopup(provider)
              .then((result) => {
                const email = result.user.email;
                const idToken = result.user.getIdToken().then((token) => {
                  const message = { email: email, token: token };
                  window.parent.postMessage(message, "*");
                });
              }).catch((error) => {
                console.log("Login Error", error);
              });
          }

          window.addEventListener("load", () => {
            const btn = document.createElement("button");
            btn.innerText = "Sign in with Google";
            btn.onclick = signInWithGoogle;
            document.body.appendChild(btn);
          });

          window.addEventListener("message", (event) => {
            if (event.data.type === "streamlit:setComponentValue") {
              console.log("Component set value:", event.data.value);
            }
          });
        </script>
      </head>
      <body style="text-align: center;">
      </body>
    </html>
    """
    components.html(firebase_auth_html, height=500)
    st.stop()

# Show logged-in user email
st.success(f"✅ Logged in as {st.session_state.get('user_email', 'User')}")

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🔁 Start New Chat"):
        st.session_state.clear()
        st.rerun()

    uploaded_file = st.file_uploader("📄 Upload PDF or image", type=["pdf"])
    language = st.selectbox("🌐 Response Language", ["English", "Hindi", "Telugu", "Tamil", "Marathi", "Bengali"])

# ========== FILE PROCESSING ==========
uploaded_content = ""
if uploaded_file:
    file_type = uploaded_file.type
    if file_type == "application/pdf":
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            uploaded_content = "\n".join(page.get_text() for page in doc)

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

# ========== LLM Query Function ==========
def query_llm(messages):
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://chat.openai.com",
            "X-Title": "Modular RAG Buddy"
        },
        json={"model": MODEL_NAME, "messages": messages}
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# ========== Prompt Generators ==========
def scratchpad_reasoning(context, question):
    return f"Let's think step-by-step.\n\nContext:\n{context}\n\nQuestion:\n{question}"

def build_primary_prompt(context, question, lang):
    return [
        {"role": "system", "content": f"You are BitsBuddy, a BITSian senior. Answer in {lang}. Use emojis, step-by-step reasoning, and be helpful."},
        {"role": "user", "content": scratchpad_reasoning(context, question)}
    ]

def build_critic_prompt(context, question, answer):
    return [
        {"role": "system", "content": "You are an honest critic checking the assistant’s answer for factual errors, incompleteness, or hallucinations."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\nCritique the above and suggest corrections if needed."}
    ]

def build_final_prompt(context, question, answer, critique, lang):
    return [
        {"role": "system", "content": f"You are BitsBuddy+ with self-evaluation enabled. Based on critique, revise your original answer. Respond in {lang} with clarity."},
        {"role": "user", "content": f"Original Answer:\n{answer}\n\nCritique:\n{critique}\n\nNow improve the answer accordingly."}
    ]

# ========== SMART MODULAR RAG ANSWER ==========
def modular_rag_smart_answer(question, lang="English"):
    try:
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Step 1: Primary Answer
        primary = query_llm(build_primary_prompt(context, question, lang))

        # Step 2: Critique (fact-check / flaw detection)
        critique = query_llm(build_critic_prompt(context, question, primary))

        # Step 3: Final Revised Answer
        improved = query_llm(build_final_prompt(context, question, primary, critique, lang))

        return improved
    except Exception as e:
        return f"❌ Error: {e}"

# ========== CHAT SESSION HANDLER ==========
if "chat" not in st.session_state:
    st.session_state.chat = []

query = st.chat_input("💬 Ask anything about BITS Pilani...")
if query:
    with st.spinner("🧠 Thinking, checking, and reflecting..."):
        answer = modular_rag_smart_answer(query, lang=language)
        st.session_state.chat.append({
            "question": query,
            "answer": answer,
            "language": language
        })

# ========== DISPLAY CHAT ==========
for chat in reversed(st.session_state.chat):
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

# ========== SIDEBAR HISTORY ==========
with st.sidebar:
    st.subheader("📂 Chat History")
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

