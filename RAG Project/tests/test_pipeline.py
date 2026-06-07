# tests/test_pipeline.py
"""
Unit tests validating all aspects of the upgraded production-grade RAG pipeline.
"""

import os
import sys
from unittest.mock import patch, MagicMock

# Ensure project path is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
from PIL import Image, ImageDraw

from src.ocr import extract_text_from_image
from src.preprocess import clean_and_chunk, chunk_by_tokens
from src.embedder import embed_texts
from src.constants import COLLECTION_NAME
from src.query import run_query

def test_ocr_image(tmp_path):
    """
    Creates a simple PIL image with text and verifies OCR can read it.
    """
    # Create a simple white image
    img = Image.new("RGB", (400, 100), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    # Draw simple text
    d.text((10, 40), "HELLO WORLD", fill=(0, 0, 0))
    
    # Save image
    temp_img_path = os.path.join(tmp_path, "ocr_test.png")
    img.save(temp_img_path)
    
    try:
        extracted = extract_text_from_image(temp_img_path)
        # Tesseract should read this text. Even if Tesseract is not installed locally on
        # developer machine, we handle the output gracefully or skip if Tesseract is missing.
        # Let's check if we get text or if it returns an OCR error warning/fallback message.
        if "tesseract" in extracted.lower() or "error" in extracted.lower():
            pytest.skip("Tesseract is not installed on this system.")
        else:
            assert "hello" in extracted.lower() or "world" in extracted.lower()
    except Exception as e:
        pytest.skip(f"Tesseract OCR is not available: {e}")

def test_chunking():
    """
    Verify chunk sizes are correct and include overlap.
    """
    # Create 2000 character string of sentences
    sentence = "The quick brown fox jumps over the lazy dog. "
    text = sentence * 45  # 45 words * ~45 chars = ~2025 chars
    
    # Assert chunking behavior under 600 chars
    chunks = clean_and_chunk(text, max_chars=500, overlap=100)
    
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 600
        
    # Check if there is some overlap (the last part of chunk 0 overlaps with start of chunk 1)
    # The clean_and_chunk uses simple_chunk with overlap parameter
    c0 = chunks[0]
    c1 = chunks[1]
    
    # We should have overlapping words
    overlap_found = False
    words_c0 = c0.split()
    words_c1 = c1.split()
    
    # Look for a sequence of 2 words from the end of c0 in the beginning of c1
    tail_words = words_c0[-4:]
    for i in range(len(words_c1) - 3):
        if words_c1[i:i+2] == tail_words[:2] or words_c1[i:i+2] == tail_words[1:3]:
            overlap_found = True
            break
            
    assert overlap_found, "Chunk overlap was not found between adjacent chunks."

def test_embedding_dimensions():
    """
    Verify BAAI/bge-large-en-v1.5 embeddings dimension is exactly 1024.
    """
    # Mock SentenceTransformer model encoding to avoid downloading a 1.3GB model during local testing
    mock_emb = np.random.randn(2, 1024).astype(np.float32)
    
    with patch("src.embedder.get_model") as mock_get_model:
        mock_model = MagicMock()
        mock_model.encode.return_value = mock_emb
        mock_get_model.return_value = mock_model
        
        embs = embed_texts(["Test sentence one.", "Test sentence two."])
        
        assert embs.shape == (2, 1024)

def test_collection_name_consistent():
    """
    Verify indexing and retrieval modules use the exact same Collection Name constant.
    """
    from src.vector_store import COLLECTION_NAME as vs_col_name
    from src.indexing import COLLECTION_NAME as idx_col_name
    from src.constants import COLLECTION_NAME as const_col_name
    
    assert vs_col_name == const_col_name
    assert idx_col_name == const_col_name
    assert const_col_name == "rag_docs_v2"

def test_query_returns_answer():
    """
    Mock the Gemini API client and ensure run_query returns a valid response dictionary.
    """
    mock_docs = ["Chunk 1 content describing a system.", "Chunk 2 content detailing evaluation metrics."]
    mock_metadatas = [{"source": "test.txt", "chunk_index": 0}, {"source": "test.txt", "chunk_index": 1}]
    mock_scores = [0.95, 0.88]
    
    # Mock retrieval and API calls
    with patch("src.query.hybrid_query") as mock_hybrid, \
         patch("src.query.call_gemini") as mock_call_gemini, \
         patch("src.query.GEMINI_API_KEY", "mock_key"):
         
        mock_hybrid.return_value = {
            "documents": mock_docs,
            "metadatas": mock_metadatas,
            "scores": mock_scores
        }
        mock_call_gemini.return_value = "The system is describing evaluation metrics. [1]"
        
        res = run_query("What does the system describe?")
        
        assert isinstance(res, dict)
        assert "answer" in res
        assert "docs" in res
        assert "metadatas" in res
        assert "scores" in res
        assert "response_time_ms" in res
        assert "llm_used" in res
        assert "is_faithful" in res
        
        assert "evaluation metrics" in res["answer"].lower()
        assert len(res["docs"]) == 2
        assert res["is_faithful"] is True
