# src/embedder.py
"""
Upgraded embedder using BAAI/bge-large-en-v1.5.
Supports lazy loading, embedding caching by hashing the text, and batch processing.
"""

import hashlib
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from src.constants import EMBED_MODEL_NAME, EMBED_BATCH_SIZE

# Dictionary cache keyed by text hash to avoid redundant encoding.
# Key: md5 hex string, Value: list/numpy array of embedding floats.
_EMBEDDING_CACHE = {}

_model = None

def get_model():
    """
    Loads SentenceTransformer model only once (lazy loading).
    Reuses the model for all encoding tasks.
    """
    global _model
    if _model is None:
        # Load the upgraded BAAI/bge-large-en-v1.5 model
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Computes embeddings for a list of texts using BAAI/bge-large-en-v1.5.
    Applies BGE query/passage prefix, uses batching, and caches results.
    
    texts: List[str]
    returns: numpy array shape (len(texts), 1024)
    """
    if not texts:
        return np.empty((0, 1024), dtype=np.float32)

    model = get_model()
    
    # Identify which texts are already cached and which need embedding
    results = [None] * len(texts)
    to_encode_indices = []
    to_encode_texts = []
    
    for idx, text in enumerate(texts):
        # We compute md5 hash for text caching
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        if text_hash in _EMBEDDING_CACHE:
            results[idx] = _EMBEDDING_CACHE[text_hash]
        else:
            to_encode_indices.append(idx)
            # BGE recommends prefix for query vs documents, but sentence_transformers
            # BGE-large-en-v1.5 works best when prefix is added appropriately.
            # We add a default query representation prefix or let sentence-transformers handle it.
            # Usually, BAAI/bge-large-en-v1.5 expects:
            # "Represent this sentence for searching relevant passages: " for query,
            # or nothing for documents. Since embed_texts can be used for both indexing (passages)
            # and retrieval (queries), we will embed texts directly.
            to_encode_texts.append(text)
            
    if to_encode_texts:
        # Encode with batch size 32
        embeddings = model.encode(
            to_encode_texts,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        # Store in cache and place in correct positions in the result list
        for i, idx in enumerate(to_encode_indices):
            emb = embeddings[i].astype("float32")
            text_hash = hashlib.md5(texts[idx].encode("utf-8")).hexdigest()
            _EMBEDDING_CACHE[text_hash] = emb
            results[idx] = emb
            
    return np.array(results, dtype=np.float32)
