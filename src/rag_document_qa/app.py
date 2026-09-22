import streamlit as st
import os
import time
import tempfile

# Groq LLM
from langchain_groq import ChatGroq

# FREE local embeddings (no API key needed)
from langchain_huggingface import HuggingFaceEmbeddings

# Text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Document chain creation
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Prompt templates
from langchain_core.prompts import ChatPromptTemplate

# Retrieval chain
from langchain_classic.chains import create_retrieval_chain

# Vector stores
from langchain_community.vectorstores import FAISS

# PDF loader
from langchain_community.document_loaders import PyPDFLoader

# Environment variables
from dotenv import load_dotenv
load_dotenv()

# Set Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = groq_api_key

# Initialize Groq LLM
llm = ChatGroq(groq_api_key=groq_api_key, model="openai/gpt-oss-120b")

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)


# Cache the embedding model so it loads only once (downloads ~80MB first time)
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_vector_embedding(uploaded_files):
    """Build the FAISS vector store from uploaded PDFs (runs automatically)."""
    st.session_state.embeddings = get_embeddings()
    all_docs = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        all_docs.extend(docs)

    if not all_docs:
        st.warning("No documents found in uploaded PDFs.")
        st.session_state.vectors = None
        return

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    final_documents = text_splitter.split_documents(all_docs)

    if not final_documents:
        st.warning("No text chunks created from documents.")
        st.session_state.vectors = None
        return

    # Build FAISS index
    st.session_state.vectors = FAISS.from_documents(
        final_documents, st.session_state.embeddings
    )


# ---------------- Streamlit UI ----------------
st.title("📚 RAG Documents Q & A with Groq + HuggingFace Embeddings")

# Sidebar for file upload and options
with st.sidebar:
    st.header("📂 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files", type="pdf", accept_multiple_files=True
    )

    # Checkbox: restrict answers to uploaded files
    use_only_files = st.checkbox(
        "Answer only from uploaded files", value=True
    )

    # Automatically build (or rebuild) the vector DB whenever uploaded files change
    if uploaded_files:
        file_signature = tuple((f.name, f.size) for f in uploaded_files)
        if st.session_state.get("file_signature") != file_signature:
            with st.spinner("🔄 Building vector database from uploaded PDFs..."):
                create_vector_embedding(uploaded_files)
                st.session_state.file_signature = file_signature
                st.success("✅ Vector Database is ready!")
    else:
        # Clear stale state if files are removed
        st.session_state.vectors = None
        st.session_state.file_signature = None


# ---------------- Chat area ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("context"):
            with st.expander("📄 Document similarity search"):
                for doc in message["context"]:
                    st.write(doc.page_content)
                    st.write("-----------------------------")


# Chat input (clears automatically after submit)
user_prompts = st.chat_input("Ask a question about your PDFs...")

if user_prompts:
    # Show the user's message immediately
    st.session_state.messages.append({"role": "user", "content": user_prompts})
    with st.chat_message("user"):
        st.markdown(user_prompts)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("🔎 Fetching data..."):
            start = time.process_time()
            vectors_ready = st.session_state.get("vectors") is not None

            if use_only_files:
                # ---- Strict RAG mode ----
                if not vectors_ready:
                    answer = (
                        "⚠️ Please upload at least one PDF first. "
                        "The 'Answer only from uploaded files' option is enabled, "
                        "so I can only respond based on your documents."
                    )
                    context_docs = []
                else:
                    document_chain = create_stuff_documents_chain(llm, prompt)
                    retriever = st.session_state.vectors.as_retriever()
                    retrieval_chain = create_retrieval_chain(retriever, document_chain)
                    response = retrieval_chain.invoke({"input": user_prompts})
                    answer = response["answer"]
                    context_docs = response["context"]
            else:
                # ---- Normal LLM mode (use docs if available) ----
                if vectors_ready:
                    document_chain = create_stuff_documents_chain(llm, prompt)
                    retriever = st.session_state.vectors.as_retriever()
                    retrieval_chain = create_retrieval_chain(retriever, document_chain)
                    response = retrieval_chain.invoke({"input": user_prompts})
                    answer = response["answer"]
                    context_docs = response["context"]
                else:
                    response = llm.invoke(user_prompts)
                    answer = response.content
                    context_docs = []

            elapsed = time.process_time() - start

        st.markdown(answer)
        st.caption(f"⏱ Response time: {elapsed:.2f} seconds")

        if context_docs:
            with st.expander("📄 Document similarity search"):
                for doc in context_docs:
                    st.write(doc.page_content)
                    st.write("-----------------------------")

    # Save assistant message to history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "context": context_docs,
        }
    )