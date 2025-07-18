import os
import fitz  # PyMuPDF
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

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
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)  # 🔹 smaller chunks to avoid overflow
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
    return vectordb.as_retriever(search_type="similarity", k=2)  # 🔹 fewer docs to stay under token limit

# ========== Load LaMini-Flan Model ==========
@st.cache_resource(show_spinner="🤖 Loading LaMini-Flan LLM...")
def load_lamini_pipeline():
    tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-783M")
    model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-Flan-T5-783M")
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300)

# ========== Retrieval + LLM Chain ==========
retriever = setup_vector_db()
lamini_pipe = load_lamini_pipeline()

def get_answer(query):
    context_docs = retriever.get_relevant_documents(query)
    context_text = "\n\n".join([doc.page_content for doc in context_docs])

    # 🔹 Limit context tokens to fit within 512-token model limit
    tokenizer = lamini_pipe.tokenizer
    context_tokens = tokenizer.encode(context_text, truncation=True, max_length=400)
    context_text = tokenizer.decode(context_tokens, skip_special_tokens=True)

    prompt = f"Answer the following question based on the context:\n\nContext:\n{context_text}\n\nQuestion: {query}"
    
    # 🔹 Optional: Show prompt for debugging
    with st.expander("🧠 Prompt sent to LaMini-Flan"):
        st.code(prompt)
    
    result = lamini_pipe(prompt)
    return result[0]["generated_text"]

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
        try:
            answer = get_answer(query)
        except Exception as e:
            answer = f"❌ Error: {str(e)}"
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
