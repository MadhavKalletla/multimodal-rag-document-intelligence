# src/retriever.py
"""
Retriever helper function routing queries to hybrid search.
"""

from typing import Dict, Any, List
from src.constants import COLLECTION_NAME
from src.vector_store import hybrid_query

def retrieve_top_k(query_text: str, k: int = 4) -> Dict[str, Any]:
    """
    Query the database using hybrid search (BM25 + vector RRF) and
    reformat the result list for compatibility.
    """
    res = hybrid_query(query_text, top_k=k)
    
    docs = res.get("documents", [])
    metas = res.get("metadatas", [])
    scores = res.get("scores", [])
    
    matches = []
    for i in range(len(docs)):
        # Generate an ID if not stored inside metadata
        doc_id = metas[i].get("source", "unknown") + f"_chunk_{metas[i].get('chunk_index', i)}"
        matches.append({
            "id": doc_id,
            "doc": docs[i],
            "meta": metas[i],
            "score": scores[i] if i < len(scores) else None
        })
        
    return {"matches": matches}
