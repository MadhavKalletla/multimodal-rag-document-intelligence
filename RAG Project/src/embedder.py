# src/embedder.py
"""
Production-ready embedder for your OCR → RAG pipeline.

Replaces the old DummyEmbedder with a real SentenceTransformer embedder.
Automatically caches model so it's loaded only once.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"   # Small, fast, high-quality model

_model = None

def get_model():
    """
    Loads SentenceTransformer model only one time.
    Reuses for all embeddings = MUCH faster.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts):
    """
    texts: List[str]
    returns: numpy array shape (len(texts), embedding_dim)
    """
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.astype("float32")
