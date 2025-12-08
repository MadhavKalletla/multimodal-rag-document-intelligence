# src/vector_store.py
"""
ChromaDB wrapper (new API) for persistent vector storage.
"""

import os
import chromadb

# Path where Chroma will persist DB files
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "vector_store")
os.makedirs(PERSIST_DIR, exist_ok=True)

# Create persistent client (NEW API)
client = chromadb.PersistentClient(path=PERSIST_DIR)

# Create or load collection
collection = client.get_or_create_collection(
    name="rag_docs",
    metadata={"hnsw:space": "cosine"}   # optional but recommended
)

def get_chroma_collection():
    """
    Return the Chroma collection object.
    """
    return collection


def query(text, top_k=3):
    """
    Query the Chroma collection for top_k similar documents.
    """
    if not text.strip():
        return {"documents": [], "metadatas": [], "distances": []}

    from src.embedder import embed_texts
    q_emb = embed_texts([text])[0].tolist()

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k
    )

    # Make sure missing keys won’t crash
    return {
        "documents": res.get("documents", [[]])[0],
        "metadatas": res.get("metadatas", [[]])[0],
        "distances": res.get("distances", [[]])[0]
    }
