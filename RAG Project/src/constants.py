# src/constants.py
"""
Central constants shared across the entire RAG pipeline.
Import from here — never hard-code collection names elsewhere.
"""

import os

# ── ChromaDB ──────────────────────────────────────────────────────────────────
COLLECTION_NAME = "rag_docs_v2"

# ── Paths ─────────────────────────────────────────────────────────────────────
_SRC_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_SRC_DIR, ".."))

VECTOR_STORE_DIR = os.path.join(_PROJECT_ROOT, "vector_store")
BM25_INDEX_PATH  = os.path.join(VECTOR_STORE_DIR, "bm25_index.pkl")
DATA_DIR         = os.path.join(_PROJECT_ROOT, "data")

# ── Embedder ──────────────────────────────────────────────────────────────────
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBED_DIM        = 1024
EMBED_BATCH_SIZE = 32

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_MAX_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64

# ── Retrieval ─────────────────────────────────────────────────────────────────
DEFAULT_TOP_K  = 5
RRF_K_CONSTANT = 60          # standard RRF constant

# ── LLM ───────────────────────────────────────────────────────────────────────
ANTHROPIC_MODEL   = "claude-haiku-4-5-20251001"
GROQ_MODEL_PRIMARY   = "llama-3.1-8b-instant"
GROQ_MODEL_FALLBACK  = "llama3-8b-8192"
LLM_MAX_TOKENS    = 512
LLM_TEMPERATURE   = 0.0
