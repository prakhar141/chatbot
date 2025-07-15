import os
import fitz  # PyMuPDF
import streamlit as st
from typing import Optional, List

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

# ========== Load all PDFs ==========
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

# ========== Streamlit Config ==========
st.set_page_config(page_title="🎓 Quillify+", page_icon="🦉", layout="wide")
st.markdown("""
    <style>
        .big-title { font-size: 38px; font-weight: bold; color: #0077b6; }
        .subtitle { font-size: 16px; color: gray; margin-top: -10px; }
    </style>
""", unsafe_allow_html=True)
st.markdown("<div class='big-title'>🦉 Quillify+</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Ask anything about BITS – notes, rules, events, or 'em proxy hacks 😜</div>", unsafe_allow_html=True)

# ========== Load & Embed Docs Once ==========
@st.cache_resource(show_spinner="📚 Thinking about you...")
def setup_vector_db():
    documents = load_all_pdfs()
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
    vectordb = FAISS.from_documents(documents, embeddings)
    retriever = vectordb.as_retriever(search_type="mmr", k=4)  # ✅ MMR enabled
    return retriever

retriever = setup_vector_db()
llm = Ollama(model="llama3", temperature=0.3)

# ========== Custom Prompt ==========
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You're a helpful and humorous college assistant at BITS Pilani.
Use only the context provided. No extra info.

Context:
{context}

Question: {question}

Helpful, short and slightly witty answer:
"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": custom_prompt}
)

# ========== Chat Memory ==========
if "chat" not in st.session_state:
    st.session_state.chat = []

query = st.chat_input("💬 Ask me anything... even why the mess food tastes like cardboard.")

if query:
    with st.spinner("🤖 Thinking..."):
        answer = qa_chain.run(query)
        st.session_state.chat.append({"question": query, "answer": answer})

# ========== Show Chat ==========
for entry in reversed(st.session_state.chat):
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        st.markdown(entry["answer"])

# ========== Follow-Up Suggestions ==========
if query:
    st.markdown("#### 🤔 You might also ask:")
    st.markdown("- What's the attendance policy?")
    st.markdown("- How do I apply for makeup tests?")
    st.markdown("- What clubs are active this semester?")
    st.markdown("- How to complain about WiFi in the hostel?")

# ========== Footer ==========
st.markdown("""
    <hr style="margin-top: 30px;">
    <div style='text-align: center; font-size: 14px; color: gray;'>
        🦉 Built with ❤️ by <b>Prakhar Mathur</b> · BITS Pilani
    </div>
""", unsafe_allow_html=True)
