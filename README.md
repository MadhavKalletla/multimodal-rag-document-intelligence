🧠 Multimodal RAG Document Intelligence System
A Retrieval-Augmented Generation (RAG) system that answers questions grounded in your own documents — PDFs, scanned images, and raw text — through a clean Streamlit interface.

💡 What it does
Upload any document, ask a question, get a precise answer pulled directly from your file's content. No hallucinations. No generic responses. Every answer is backed by semantically retrieved chunks from your actual documents.

🏗️ Architecture
Document → OCR / Text Extraction → Chunking (with overlap)
→ Sentence-Transformer Embeddings → FAISS Index
→ Semantic Top-K Retrieval → LLM Answer Generation → Streamlit UI

🛠️ Tech Stack
LayerToolsUIStreamlitOrchestrationLangChainEmbeddingsSentence-TransformersVector StoreFAISSOCRTesseractLLMOpenAI API / others

🚀 Getting Started
bashpip install -r requirements.txt
streamlit run app.py

✨ Key Features

📄 Multi-modal ingestion — PDFs, scanned images, plain text
🔍 OCR preprocessing for handwritten and scanned documents
✂️ Overlapping chunking for better context preservation
⚡ FAISS semantic search for fast, relevant retrieval
🤖 LLM answer generation grounded strictly in retrieved context
