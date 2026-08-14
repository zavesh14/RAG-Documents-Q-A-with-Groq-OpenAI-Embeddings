📚 RAG Documents Q&A with Groq + OpenAI Embeddings
An interactive Retrieval-Augmented Generation (RAG) application built with Streamlit, LangChain, Groq LLM, and OpenAI Embeddings.
This project allows users to upload multiple PDF documents, build a vector database using FAISS, and ask natural language questions that the AI answers based on the document content.
If no documents are uploaded, the app gracefully falls back to general AI responses.

🚀 Features
Multiple PDF Uploads — Upload and process several PDFs at once.

Automatic Chunking — Splits documents into manageable text chunks using LangChain’s RecursiveCharacterTextSplitter.

Vector Database Creation — Embeds document chunks using OpenAI embeddings and stores them in a FAISS index.

Contextual Q&A — Uses Groq’s qwen/qwen3.6-27b model to answer questions based on document context.

Fallback Mode — If no documents are uploaded, the AI provides general answers.

Spinner Feedback — Displays a loading spinner while fetching answers from the LLM.

Streamlit Sidebar Layout — File upload and database creation tools are neatly placed in the sidebar.

🧠 Tech Stack
Component	Description
Streamlit	Interactive web interface
LangChain	Framework for document loading, splitting, and retrieval
Groq LLM	High-performance language model for reasoning and answering
OpenAI Embeddings	Converts text into vector representations
FAISS	Efficient vector similarity search
dotenv	Loads environment variables securely
