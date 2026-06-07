# src/indexing.py
"""
Indexing pipeline to parse documents, chunk them, embed, and store in both ChromaDB and BM25.
"""

from datetime import datetime
import os
from typing import List, Dict, Any

from src.constants import COLLECTION_NAME
from src.vector_store import get_chroma_collection, update_bm25_index
from src.preprocess import chunk_with_metadata
from src.embedder import embed_texts
from src.ocr import extract_text

def index_text(text: str, source_name: str = "uploaded_document", page_num: int = 0) -> int:
    """
    Indexes raw text into the persistent Chroma collection and BM25 index.
    
    Returns the number of chunks added.
    """
    if not text or not text.strip():
        return 0

    # Get metadata-tagged chunks using token-aware chunking (max_tokens=512)
    chunks_meta = chunk_with_metadata(text, source=source_name, page_num=page_num)
    if not chunks_meta:
        return 0

    chunks = [c["text"] for c in chunks_meta]

    # Generate BGE embeddings
    embs = embed_texts(chunks)
    emb_list = embs.tolist()

    collection = get_chroma_collection()

    ids = []
    metadatas = []
    timestamp_str = datetime.utcnow().isoformat()

    for i, c_meta in enumerate(chunks_meta):
        # Generate unique ID for the chunk
        chunk_id = f"{source_name}_chunk_{page_num}_{i}"
        ids.append(chunk_id)
        
        # Build metadata dictionary to be stored in Chroma and BM25
        metadata = {
            "source": c_meta["source"],
            "page_num": c_meta["page_num"],
            "chunk_index": c_meta["chunk_index"],
            "char_count": c_meta["char_count"],
            "token_count": c_meta["token_count"],
            "timestamp": timestamp_str,
            "preview": c_meta["text"][:200].replace("\n", " ")
        }
        metadatas.append(metadata)

    # Upsert into ChromaDB
    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=emb_list,
        metadatas=metadatas
    )

    # Update BM25 Index
    update_bm25_index(chunks, metadatas, ids)

    return len(chunks)

def index_image_file(image_path: str, source_name: str = "image") -> int:
    """
    OCR -> index pipeline for image file.
    """
    from src.ocr import extract_text_from_image
    text = extract_text_from_image(image_path)
    return index_text(text, source_name=source_name)

def index_file(file_path: str, source_name: str = None) -> int:
    """
    Extracts text from any supported file (PDF, Docx, Text, Image, Audio placeholder)
    and indexes it.
    
    Returns the number of chunks added.
    """
    if not source_name:
        source_name = os.path.basename(file_path)

    text = extract_text(file_path)
    return index_text(text, source_name=source_name)
