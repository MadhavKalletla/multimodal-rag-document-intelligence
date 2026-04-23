Multimodal RAG Document Intelligence System
A Retrieval-Augmented Generation system that answers user questions grounded in uploaded documents — supporting PDFs, scanned images, and raw text through a clean Streamlit interface.
What it does
Upload any document — digital PDF, scanned image, or plain text — ask a question, and get a context-aware answer generated directly from your document's content. No hallucination, no generic responses — every answer is backed by retrieved chunks from your actual files.
Architecture
Document → OCR / Text Extraction → Chunking (with overlap)
→ Sentence-Transformer Embeddings → FAISS Index
→ Semantic Top-K Retrieval → LLM Answer Generation → Streamlit UI
Tech Stack

Python, Streamlit — pipeline logic and interactive UI
LangChain — orchestration and chain management
Sentence-Transformers — embedding generation
FAISS — vector indexing and semantic search
Tesseract OCR — text extraction from scanned and handwritten sources
LLM APIs — answer generation from retrieved context

Getting Started
bashpip install -r requirements.txt
streamlit run app.py
Key Features

Multi-modal ingestion: PDFs, scanned images, and plain text
OCR preprocessing for handwritten and scanned documents
Overlapping text chunking for better context preservation
FAISS vector search for fast, relevant chunk retrieval
LLM answer generation grounded strictly in retrieved context
