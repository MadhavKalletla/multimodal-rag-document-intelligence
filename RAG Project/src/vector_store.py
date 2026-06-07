# src/vector_store.py
"""
ChromaDB persistent vector store combined with BM25 keyword search.
Implements Reciprocal Rank Fusion (RRF) to blend semantic and keyword results.
"""

import os
import pickle
import re
from typing import List, Dict, Any, Tuple
import chromadb
from rank_bm25 import BM25Okapi

from src.constants import (
    COLLECTION_NAME,
    VECTOR_STORE_DIR,
    BM25_INDEX_PATH,
    RRF_K_CONSTANT
)
from src.embedder import embed_texts

# Ensure persist directory exists
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Create persistent ChromaDB client
client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)

# Initialize collection using central COLLECTION_NAME constant
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

def get_chroma_collection():
    """Return the Chroma collection object."""
    return collection

def tokenize_for_bm25(text: str) -> List[str]:
    """Helper to split text into lowercase alphanumeric tokens for BM25 indexing."""
    return re.findall(r'\b\w+\b', text.lower())

def load_bm25_data() -> Dict[str, Any]:
    """Loads raw BM25 corpus and metadata from pickle file if it exists."""
    if os.path.exists(BM25_INDEX_PATH):
        try:
            with open(BM25_INDEX_PATH, "rb") as f:
                data = pickle.load(f)
                # Should contain: "documents": List[str], "metadatas": List[dict], "ids": List[str], "corpus_tokens": List[List[str]]
                return data
        except Exception as e:
            print(f"Warning: Failed to load BM25 index from {BM25_INDEX_PATH}: {e}")
    return {"documents": [], "metadatas": [], "ids": [], "corpus_tokens": []}

def save_bm25_data(data: Dict[str, Any]) -> None:
    """Saves raw BM25 corpus and metadata to pickle file."""
    try:
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Error saving BM25 index: {e}")

def update_bm25_index(new_docs: List[str], new_metadatas: List[Dict[str, Any]], new_ids: List[str]) -> None:
    """
    Appends new documents, metadatas, and ids to the BM25 index pickle.
    If document IDs already exist, they will be updated (overwritten).
    """
    data = load_bm25_data()
    
    # Create maps for updates
    doc_map = {id_: (doc, meta) for id_, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])}
    
    # Update with new items
    for id_, doc, meta in zip(new_ids, new_docs, new_metadatas):
        doc_map[id_] = (doc, meta)
        
    # Rebuild data structures
    updated_ids = list(doc_map.keys())
    updated_docs = [doc_map[id_][0] for id_ in updated_ids]
    updated_metadatas = [doc_map[id_][1] for id_ in updated_ids]
    updated_corpus_tokens = [tokenize_for_bm25(doc) for doc in updated_docs]
    
    save_bm25_data({
        "documents": updated_docs,
        "metadatas": updated_metadatas,
        "ids": updated_ids,
        "corpus_tokens": updated_corpus_tokens
    })

def clear_bm25_index() -> None:
    """Deletes the pickle file containing the BM25 index."""
    if os.path.exists(BM25_INDEX_PATH):
        try:
            os.remove(BM25_INDEX_PATH)
        except Exception as e:
            print(f"Error removing BM25 index file: {e}")

def query(text: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Standard ChromaDB vector-only query.
    """
    if not text.strip():
        return {"documents": [], "metadatas": [], "distances": []}

    q_emb = embed_texts([text])[0].tolist()

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k
    )

    return {
        "documents": res.get("documents", [[]])[0],
        "metadatas": res.get("metadatas", [[]])[0],
        "distances": res.get("distances", [[]])[0]
    }

def hybrid_query(text: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Performs hybrid retrieval using Reciprocal Rank Fusion (RRF)
    combining ChromaDB cosine similarity search and rank_bm25 keyword search.
    
    RRF Score = 1 / (60 + Rank_semantic) + 1 / (60 + Rank_keyword)
    """
    if not text.strip():
        return {"documents": [], "metadatas": [], "scores": []}

    # 1. Fetch top results from ChromaDB (Semantic Vector Search)
    # Get up to 20 results to have a decent candidate pool for fusion
    candidate_limit = max(20, top_k * 2)
    
    semantic_res = query(text, top_k=candidate_limit)
    semantic_docs = semantic_res.get("documents", [])
    semantic_metadatas = semantic_res.get("metadatas", [])
    
    # Store candidates keyed by document/chunk content or unique identifier.
    # To be safe and deterministic, let's map by content or by metadata's unique attributes.
    # Let's map by content since we want to align texts.
    semantic_ranks = {doc: rank for rank, doc in enumerate(semantic_docs)}
    
    # Metadata map to look up metadata of docs
    doc_metadata_map = {}
    for doc, meta in zip(semantic_docs, semantic_metadatas):
        doc_metadata_map[doc] = meta

    # 2. Fetch top results from BM25 (Keyword Search)
    bm25_data = load_bm25_data()
    bm25_docs = bm25_data.get("documents", [])
    bm25_metadatas = bm25_data.get("metadatas", [])
    bm25_corpus_tokens = bm25_data.get("corpus_tokens", [])
    
    keyword_ranks = {}
    if bm25_docs and bm25_corpus_tokens:
        try:
            bm25 = BM25Okapi(bm25_corpus_tokens)
            query_tokens = tokenize_for_bm25(text)
            # Get BM25 scores
            scores = bm25.get_scores(query_tokens)
            # Rank indices descending
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:candidate_limit]
            
            for rank, idx in enumerate(top_indices):
                doc_text = bm25_docs[idx]
                keyword_ranks[doc_text] = rank
                # Populate metadata if not already present from semantic search
                if doc_text not in doc_metadata_map:
                    doc_metadata_map[doc_text] = bm25_metadatas[idx]
        except Exception as e:
            print(f"Error computing BM25 query: {e}")

    # 3. Reciprocal Rank Fusion
    # Gather union of all document candidates
    all_candidate_docs = set(semantic_ranks.keys()).union(set(keyword_ranks.keys()))
    
    rrf_scores = {}
    for doc in all_candidate_docs:
        # Default ranks if not present in the top retrieve list is infinity (or just ignored in sum)
        sem_rank = semantic_ranks.get(doc, None)
        key_rank = keyword_ranks.get(doc, None)
        
        score = 0.0
        if sem_rank is not None:
            score += 1.0 / (RRF_K_CONSTANT + sem_rank)
        if key_rank is not None:
            score += 1.0 / (RRF_K_CONSTANT + key_rank)
            
        rrf_scores[doc] = score

    # Sort documents by final fused RRF score descending
    sorted_docs = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)[:top_k]
    
    final_docs = []
    final_metadatas = []
    final_scores = []
    
    for doc in sorted_docs:
        final_docs.append(doc)
        final_metadatas.append(doc_metadata_map.get(doc, {}))
        final_scores.append(rrf_scores[doc])
        
    return {
        "documents": final_docs,
        "metadatas": final_metadatas,
        "scores": final_scores
    }
