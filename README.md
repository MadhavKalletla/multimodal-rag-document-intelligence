Multi-Modal RAG Document Intelligence System

A Retrieval-Augmented Generation (RAG) system that can analyze PDFs, scanned images (via OCR), and text documents to answer user questions with grounded, context-aware responses.
Supports OCR → chunking → embeddings → vector search → LLM answer generation with a clean Streamlit UI.

🚀 Features

Multi-modal document ingestion (Images + PDFs)

OCR processing for scanned documents

Text chunking with overlap

Embedding generation using Sentence-Transformers / LLM embeddings

Vector indexing with FAISS or Chroma

Semantic search for top-K relevant chunks

LLM-based answer generation using retrieved context

Streamlit UI for interactive querying

🧠 Architecture (Short Overview)
Document → OCR/Text Extraction → Chunking → Embeddings → FAISS Index → Semantic Retrieval → LLM Answer
🛠 Tech Stack

Python, Streamlit

LangChain, Sentence-Transformers

FAISS / Chroma

Tesseract OCR

LLM APIs (OpenAI / others)
▶️ Running the Project
pip install -r requirements.txt
streamlit run app.py

📌 Future Improvements

Better PDF parsing

Multi-language OCR

Improved monitoring for retrieval quality
