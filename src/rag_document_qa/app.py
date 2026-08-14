import streamlit as st
import os
import time
import tempfile
# Groq LLM
from langchain_groq import ChatGroq

# OpenAI embeddings
from langchain_openai import OpenAIEmbeddings

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

# Set API key
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = groq_api_key

# Initialize Groq LLM
llm = ChatGroq(groq_api_key=groq_api_key, model="qwen/qwen3.6-27b")

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

def create_vector_embedding(uploaded_files):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        all_docs = []

        # Save each uploaded PDF to a temporary file and load
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            all_docs.extend(docs)

        if not all_docs:
            st.warning("No documents found in uploaded PDFs.")
            return

        # Split into chunks
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(all_docs)

        if not st.session_state.final_documents:
            st.warning("No text chunks created from documents.")
            return

        # Build FAISS index
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings
        )

# Streamlit UI
st.title("📚 RAG Documents Q & A with Groq + OpenAI Embeddings")

# Sidebar for file upload and vector building
with st.sidebar:
    st.header("📂 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files", type="pdf", accept_multiple_files=True
    )

    if st.button("Build Vector Database") and uploaded_files:
        create_vector_embedding(uploaded_files)
        st.success("✅ Vector Database is ready!")

# Main area for chat
st.subheader("💬 Ask Questions About Your PDFs")
user_prompts = st.text_input("Enter your query")

if user_prompts:
    with st.spinner("🔎 Fetching data from documents..."):
        start = time.process_time()

        if "vectors" in st.session_state and uploaded_files:
            # Use retrieval augmented generation
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            response = retrieval_chain.invoke({"input": user_prompts})
            answer = response["answer"]
            context_docs = response["context"]
        else:
            # Fallback: general LLM response without docs
            response = llm.invoke(user_prompts)
            answer = response.content
            context_docs = []

        st.write(f"⏱ Response time: {time.process_time() - start:.2f} seconds")

    st.write("### Answer")
    st.write(answer)

    if context_docs:
        with st.expander("Document similarity search"):
            for i, doc in enumerate(context_docs):
                st.write(doc.page_content)
                #st.write("-----------------------------")
