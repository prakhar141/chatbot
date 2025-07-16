import os
import fitz  # PyMuPDF
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.prompts.prompt import PromptTemplate  # ✅ Inject smarter prompts

# ========== PDF Loader ==========
def load_all_pdfs():
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    for file in os.listdir():
        if file.endswith(".pdf"):
            with fitz.open(file) as doc:
                text = "\n".join([page.get_text() for page in doc])
            chunks = splitter.split_text(text)
            docs.extend([Document(page_content=chunk, metadata={"source": file}) for chunk in chunks])
    return docs

# ========== UI ==========
st.set_page_config(page_title="🎓 Quillify", page_icon="🤖", layout="wide")
st.markdown("""
    <style>
        .big-title { font-size: 36px; font-weight: 800; margin-bottom: 10px; color: #3B82F6; }
        .subtitle { font-size: 16px; color: gray; margin-top: -10px; }
    </style>
""", unsafe_allow_html=True)
st.markdown("<div class='big-title'>🎓 Quillify</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Ask anything about BITS – syllabus, events, academics, policies, and more</div>", unsafe_allow_html=True)

# ========== Vector DB ==========
@st.cache_resource(show_spinner="📚 Embedding PDFs...")
def setup_vector_db():
    documents = load_all_pdfs()
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
    vectordb = FAISS.from_documents(documents, embeddings)
    return vectordb.as_retriever(search_type="similarity", k=4)

retriever = setup_vector_db()

# ========== Load FLAN Model ==========
@st.cache_resource(show_spinner="🔗 Loading FLAN-T5...")
def load_llm():
    model_id = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    return HuggingFacePipeline(pipeline=pipe)

llm = load_llm()

# ========== Smarter Prompt Template ==========
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant at BITS Pilani. Use the provided context to answer the question in a clear and concise manner. If the answer is not in the context, say "Sorry, I couldn't find the answer in the material."

Context:
{context}

Question:
{question}

Answer:
"""
)

# ========== Retrieval + QA ==========
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template}
)

# ========== Chat Memory ==========
if "chat" not in st.session_state:
    st.session_state.chat = []

query = st.chat_input("💬 Ask me about BITS...")

if query:
    with st.spinner("🤖 Thinking..."):
        answer = qa_chain.run(query)
        st.session_state.chat.append({"question": query, "answer": answer})

# ========== Display Chat ==========
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
