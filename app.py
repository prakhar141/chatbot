import os
import time
import fitz
import requests
from PIL import Image
import streamlit as st
# import pytesseract
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
    # elif "image" in file_type:
    #     uploaded_content = pytesseract.image_to_string(Image.open(uploaded_file))

    if uploaded_content.strip():
        st.success("✅ Extracted content from file.")
        st.text_area("📄 Preview (first 1000 chars)", uploaded_content[:1000], height=200)
    else:
        st.warning("⚠️ Couldn't extract readable text from the file.")

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
    if not docs:
        # return an empty retriever-like object that returns no docs
        class EmptyRetriever:
            def get_relevant_documents(self, q): return []
        return EmptyRetriever()
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = FAISS.from_documents(docs, embedder)
    return vectordb.as_retriever(search_type="similarity", k=K_VAL)

retriever = load_vector_db()

# ========== LLM QUERY FUNCTION ==========
def query_llm(messages):
    """
    messages: list of {role, content}
    returns single string content
    """
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

# ========== PROMPT BUILDERS ==========
def scratchpad_reasoning(context, question):
    return f"Let's think step-by-step.\n\nContext:\n{context}\n\nQuestion:\n{question}"

def build_thinking_prompt(question, context):
    return [
        {"role": "system", "content": (
            "You are an assistant that narrates a concise, casual internal monologue "
            "before answering. Keep it 3-5 short sentences, conversational, use 'Hmm...', 'Oh I see...', 'Wait...', "
            "and DO NOT give the final answer — only describe what you are thinking and what you plan to do next."
        )},
        {"role": "user", "content": f"Question: {question}\n\nRelevant context:\n{context}"}
    ]

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

# ========== SMART MODULAR RAG ANSWER (returns stages) ==========
def modular_rag_smart_answer(context, question, lang="English"):
    """
    Returns dict with keys: primary, critique, final
    """
    try:
        # Step 1: Primary Answer (draft)
        primary = query_llm(build_primary_prompt(context, question, lang))

        # Step 2: Critique (fact-check / flaw detection)
        critique = query_llm(build_critic_prompt(context, question, primary))

        # Step 3: Final Revised Answer
        improved = query_llm(build_final_prompt(context, question, primary, critique, lang))

        return {"primary": primary, "critique": critique, "final": improved}
    except Exception as e:
        return {"error": str(e)}

# ========== CHAT SESSION HANDLER ==========
if "chat" not in st.session_state:
    st.session_state.chat = []

query = st.chat_input("💬 Ask anything about BITS Pilani...")
if query:
    # Create the assistant chat bubble and stream "thinking" + final answer inside it
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        try:
            # Retrieve context once and reuse
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in docs]) if docs else ""

            # 0) Generate AI "thinking out loud" monologue
            thinking_prompt = build_thinking_prompt(query, context)
            thinking_text = query_llm(thinking_prompt)

            # Stream the thinking monologue (GPT-like)
            animated = ""
            for c in thinking_text:
                animated += c
                thinking_placeholder.markdown(f"**Thinking:** {animated}|")
                time.sleep(0.01)  # adjust speed as you like
            thinking_placeholder.markdown(f"**Thinking:** {animated}")  # finalize thinking

            # short dramatic pause
            time.sleep(0.35)

            # 1-3) Run modular RAG pipeline (primary, critique, final)
            # Show small stage updates while the calls are happening (so user sees progress)
            thinking_placeholder.markdown("🔁 Reasoning...\n\n• ✏️ Drafting initial answer...")
            rag_result = modular_rag_smart_answer(context, query, lang=language)

            if "error" in rag_result:
                thinking_placeholder.markdown(f"❌ Error while generating answer: {rag_result['error']}")
                st.session_state.chat.append({
                    "question": query,
                    "thinking": thinking_text,
                    "primary": rag_result.get("primary", ""),
                    "critique": rag_result.get("critique", ""),
                    "final": rag_result.get("final", rag_result.get("error", "")),
                    "language": language
                })
            else:
                # Stream the final polished answer by replacing the thinking monologue
                final_answer = rag_result["final"]
                animated = ""
                for c in final_answer:
                    animated += c
                    # show a | cursor while streaming
                    thinking_placeholder.markdown(animated + "|")
                    time.sleep(0.005)
                thinking_placeholder.markdown(animated)

                # Save full stages to chat history
                st.session_state.chat.append({
                    "question": query,
                    "final": rag_result["final"],
                    "language": language
                })

        except Exception as e:
            thinking_placeholder.markdown(f"❌ Error: {e}")
            st.session_state.chat.append({
                "question": query,
                "final": f"Error: {e}",
                "language": language
            })

# ========== DISPLAY CHAT (history) ==========
for chat in reversed(st.session_state.chat):
    with st.chat_message("user"):
        st.markdown(chat["question"])

    with st.chat_message("assistant"):
        # show the final (polished) answer
        # present a compact "thinking" expander with the AI monologue + stages if user wants to inspect
        st.markdown(chat["final"])

        with st.expander("🧾 Show model reasoning (thinking)", expanded=False):
            st.markdown("**Thinking:**")
            st.markdown(chat.get("thinking", ""))
            st.markdown("---")
           # st.markdown("**Draft (primary):**")
            #st.markdown(chat.get("primary", ""))
            #st.markdown("---")
            #st.markdown("**Critique:**")
            #st.markdown(chat.get("critique", ""))
            #st.markdown("---")
            st.markdown("**Final:**")
            st.markdown(chat.get("final", ""))

# ========== SIDEBAR HISTORY ==========
with st.sidebar:
    st.subheader("📂 Chat History")
    for i, chat in enumerate(reversed(st.session_state.chat)):
        st.markdown(f"**Q{i+1}:** {chat['question']}")
        st.markdown(f"**A{i+1}:** {chat['final'][:150]}...")
        st.markdown("---")

# ========== FOOTER ==========
st.markdown("""
<hr style="margin-top: 40px;">
<div style='text-align: center; color: #888; font-size: 14px;'>
    Built with ❤️ by <b>Prakhar Mathur</b> · BITS Pilani · 
    <br>📬 Email: <a href="mailto:f20240347@pilani.bits-pilani.ac.in">Contact Prakhar</a>
</div>
""", unsafe_allow_html=True)
