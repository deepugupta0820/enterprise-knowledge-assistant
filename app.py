import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from config import *


# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="Enterprise Knowledge Assistant", layout="wide")

# Ensure required local ingestion directory exists
os.makedirs(DATA_DIR, exist_ok=True)


# CORE RAG ENGINE INTERFACES
@st.cache_resource
def load_embedding_model():
    """Initializes and caches the semantic text embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def initialize_vector_db():
    """Processes documents inside the data folder and saves embeddings locally."""
    if not os.listdir(DATA_DIR):
        return None

    # 1. Document Loading
    loader = PyPDFDirectoryLoader(DATA_DIR)
    documents = loader.load()

    # 2. Chunking Strategy (Targeting sentence structures with semantic overlap)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(documents)

    embeddings = load_embedding_model()
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(DB_FAISS_PATH)
    return vector_db

def load_vector_db():
    """Loads the pre-built FAISS vector storage from the local disk."""
    embeddings = load_embedding_model()
    if os.path.exists(DB_FAISS_PATH):
        return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

def format_docs(docs):
    """Combines documents and formats clear page/file metadata strings."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", 0) + 1  # Normalize index to human page count
        formatted.append(f"Content: {doc.page_content}\nSource: {os.path.basename(source)} (Page {page})\n---")
    return "\n\n".join(formatted)


# STREAMLIT USER INTERACTION INTERFACE
st.title("🏢 Enterprise Knowledge Assistant")
st.write("Ask questions and get answers grounded strictly in internal corporate documentation.")

# Load API key from .env
groq_api_key = os.getenv("GROQ_API_KEY")

# Sidebar Controls
with st.sidebar:
    st.header("📄 Document Ingestion")
    uploaded_files = st.file_uploader("Upload internal PDF files", type=["pdf"], accept_multiple_files=True)

    if st.button("Process & Index Knowledge Base", use_container_width=True):
        if uploaded_files:
            for uploaded_file in uploaded_files:
                with open(os.path.join(DATA_DIR, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
            with st.spinner("Analyzing document semantic structure..."):
                initialize_vector_db()
            st.success("Indexing complete! Knowledge base updated.")
        else:
            st.warning("Please upload valid PDF documentation first.")

# Check for initialized operational database
db = load_vector_db()

if db is None:
    st.info("👋 Welcome! Please upload and index documents using the sidebar panel to begin searching.")
else:
    user_query = st.text_input("Enter your policy, technical, or process question here:")

    if user_query:
        if not groq_api_key:
            st.error("⚠️ Please enter your Groq API key in the sidebar to continue.")
            st.stop()

        try:
            llm = ChatGroq(
                api_key=groq_api_key,
                model=GROQ_MODEL,
                temperature=0.0,
            )
            retriever = db.as_retriever(search_kwargs={"k": TOP_K})

            # Anti-Hallucination Prompt
            rag_prompt = ChatPromptTemplate.from_template("""
You are an expert Enterprise Knowledge Assistant. Answer the question below strictly using the provided context.
If the context does not contain the answer, state explicitly: "I am sorry, but that information is unavailable within current system documentation."
Do not make up or extrapolate facts under any circumstances. Always include inline mentions of your sources.

Context Base:
{context}

User Question: {question}

Helpful, Source-Cited Answer:
""")

            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | rag_prompt
                | llm
                | StrOutputParser()
            )

            with st.spinner("Synthesizing context sources..."):
                response = rag_chain.invoke(user_query)
                relevant_chunks = retriever.invoke(user_query)

            st.subheader("💡 Knowledge Assistant Response")
            st.write(response)

            # Context Transparency UI Dropdown Component
            with st.expander("View Supporting Source Document References"):
                for chunk in relevant_chunks:
                    src = os.path.basename(chunk.metadata.get("source", "Doc"))
                    pg = chunk.metadata.get("page", 0) + 1
                    st.caption(f"**Document Reference:** {src} | **Location:** Page {pg}")
                    st.write(chunk.page_content)
                    st.markdown("---")

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")