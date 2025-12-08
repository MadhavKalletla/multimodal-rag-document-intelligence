# src/indexing.py
import os
from typing import List
import numpy as np

from src.vector_store import get_chroma_collection
from src.preprocess import clean_and_chunk
from src.embedder import embed_texts   # expects a function embed_texts(List[str]) -> np.ndarray

def index_text(text: str, source_name: str = "uploaded_image") -> int:
    """
    Index raw text into the persistent Chroma collection.
    Returns number of chunks added.
    """
    if not text or not text.strip():
        return 0

    chunks = clean_and_chunk(text)
    if not chunks:
        return 0

    # embeddings should be numpy array shape (n_chunks, dim)
    embs = embed_texts(chunks)
    if hasattr(embs, "tolist"):
        emb_list = embs.tolist()
    else:
        emb_list = [list(map(float, e)) for e in embs]

    collection = get_chroma_collection()

    # prepare ids and metadata, include a short snippet for previews
    ids = []
    metadatas = []
    docs = []
    for i, c in enumerate(chunks):
        chunk_id = f"{source_name}_chunk_{i}"
        ids.append(chunk_id)
        preview = c[:200].replace("\n", " ")
        metadatas.append({"source": source_name, "chunk_index": i, "preview": preview})
        docs.append(c)

    collection.upsert(
        ids=ids,
        documents=docs,
        embeddings=emb_list,
        metadatas=metadatas
    )

    return len(chunks)

def index_image_file(image_path: str, source_name: str = "image") -> int:
    """
    OCR -> index pipeline for image file.
    """
    from src.ocr import extract_text_from_image
    text = extract_text_from_image(image_path)
    return index_text(text, source_name=source_name)
