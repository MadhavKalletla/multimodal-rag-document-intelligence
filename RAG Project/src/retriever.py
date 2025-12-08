# src/retriever.py
from typing import List, Dict, Any
import os
import chromadb
from chromadb.config import Settings

def get_collection(persist_dir: str = "vector_store",
                   name: str = "docs_day4"):
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_collection(name=name)

def retrieve_top_k(query: str, k: int = 4,
                   persist_dir: str = "vector_store",
                   name: str = "docs_day4") -> Dict[str, Any]:
    col = get_collection(persist_dir, name)
    # Chroma will embed query internally if the collection was created with embeddings.
    res = col.query(query_texts=[query], n_results=k)
    # Normalize into a simple list of (doc, meta, id)
    out = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    for i in range(len(docs)):
        out.append({"id": ids[i], "doc": docs[i], "meta": metas[i]})
    return {"matches": out}
