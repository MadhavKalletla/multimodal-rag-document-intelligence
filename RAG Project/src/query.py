# src/query.py
"""
Query processor and answer generation engine using Google Gemini API.
Loads environment variables, retrieves context from hybrid_query,
sends query to Google Gemini API, tracks execution time,
and performs a local faithfulness verification.
"""

import os
import time
import re
import traceback
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file (explicit path)
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# Initialize Gemini Client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

from src.vector_store import hybrid_query

def build_context_string(docs: List[str]) -> str:
    """Formats retrieved document chunks into a numbered context block."""
    context_parts = []
    for idx, doc in enumerate(docs):
        context_parts.append(f"--- CHUNK {idx} ---\n{doc.strip()}")
    return "\n\n".join(context_parts)

def clean_tokens(text: str) -> Set[str]:
    """Helper to tokenize text into lowercase alphanumeric words, filtering out basic stopwords."""
    words = re.findall(r'\b\w+\b', text.lower())
    stopwords = {
        "the", "a", "an", "and", "or", "but", "if", "then", "else", "is", "are", 
        "was", "were", "to", "of", "in", "on", "at", "for", "with", "about", 
        "against", "between", "into", "through", "during", "before", "after", 
        "above", "below", "from", "up", "down", "in", "out", "on", "off", 
        "over", "under", "again", "further", "then", "once", "here", "there", 
        "when", "where", "why", "how", "all", "any", "both", "each", "few", 
        "more", "most", "other", "some", "such", "no", "nor", "not", "only", 
        "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", 
        "just", "don", "should", "now"
    }
    return {w for w in words if w not in stopwords and len(w) > 1}

def check_faithfulness(answer: str, context: str) -> bool:
    """
    Performs a local faithfulness check.
    Verifies that all meaningful non-stopword tokens in the answer
    exist within the retrieved context.
    """
    ans_text = answer.lower()
    if "i don't know" in ans_text or "i do not know" in ans_text or "no documents indexed" in ans_text:
        # Saying "I don't know" when answer is not in context is considered faithful.
        return True
        
    answer_tokens = clean_tokens(answer)
    context_tokens = clean_tokens(context)
    
    if not answer_tokens:
        return True
        
    # Check what % of answer tokens exist in the context
    intersection = answer_tokens.intersection(context_tokens)
    if not intersection:
        return False
    ratio = len(intersection) / len(answer_tokens)
    
    # We require at least 85% token containment to pass as faithful
    return ratio >= 0.85

def call_gemini(prompt: str, temperature: float = 0.0, max_tokens: int = 350) -> str:
    """
    Calls Google Gemini API using the new google-genai SDK.
    """
    if not client:
        raise RuntimeError("GEMINI_API_KEY not set in .env")

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction="You are a precise factual assistant. Answer using ONLY the provided context. If the answer is not in the context say 'I don't know'. Always cite source chunk indices in square brackets like [0], [1]. Keep answers to 2-4 sentences.",
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    )
    return response.text.strip()

def extractive_answer_from_chunks(chunks: List[str], question: str, k: int = 1) -> Dict[str, Any]:
    """
    Rank chunks using local BGE semantic embeddings and return the best matching chunk.
    """
    if not chunks:
        return {"answer": "No data", "sources": []}
    try:
        from src.embedder import embed_texts
        import numpy as np
        chunk_embs = embed_texts(chunks)
        q_emb = embed_texts([question])[0]
        scores = np.dot(chunk_embs, q_emb) / (np.linalg.norm(chunk_embs, axis=1) * np.linalg.norm(q_emb) + 1e-9)
        best_idx = int(np.argmax(scores))
        return {
            "answer": chunks[best_idx],
            "sources": [{"chunk_index": best_idx, "text": chunks[best_idx], "score": float(scores[best_idx])}]
        }
    except Exception as e:
        print("Extractive calculation failed:", e)
        return {
            "answer": chunks[0],
            "sources": [{"chunk_index": 0, "text": chunks[0], "score": 0.0}]
        }

def run_query(query_text: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Core query interface:
    1. Starts execution timer.
    2. Performs hybrid (semantic + keyword) retrieval.
    3. Triggers Google Gemini API / Extractive fallback.
    4. Evaluates local faithfulness.
    5. Returns response metadata.
    """
    start_time = time.time()
    
    # 1. Retrieve Candidate Passages via Hybrid (Chroma + BM25) search
    try:
        retrieval_res = hybrid_query(query_text, top_k=top_k)
        top_docs = retrieval_res.get("documents", [])
        top_metadatas = retrieval_res.get("metadatas", [])
        top_scores = retrieval_res.get("scores", [])
    except Exception as e:
        print(f"Hybrid retrieval error: {e}")
        top_docs, top_metadatas, top_scores = [], [], []

    if not top_docs:
        elapsed = int((time.time() - start_time) * 1000)
        return {
            "answer": "No documents indexed. Please upload and index documents in the Knowledge Base first.",
            "docs": [],
            "metadatas": [],
            "scores": [],
            "response_time_ms": elapsed,
            "llm_used": "None (No docs)",
            "is_faithful": True
        }

    # 2. Build contextual prompt
    context_str = build_context_string(top_docs)
    prompt = f"Context:\n{context_str}\n\nQuestion: {query_text}\nAnswer:"

    # 3. Call LLM with Gemini and fallback
    if client:
        try:
            answer_text = call_gemini(prompt, temperature=0.0, max_tokens=350)
            elapsed_ms = int((time.time() - start_time) * 1000)
            is_faithful = check_faithfulness(answer_text, context_str)
            return {
                "answer": answer_text,
                "docs": top_docs,
                "metadatas": top_metadatas,
                "scores": top_scores,
                "used_llm": True,
                "response_time_ms": elapsed_ms,
                "llm_used": "Google Gemini (gemini-2.0-flash)",
                "is_faithful": is_faithful
            }
        except Exception as e:
            print("Gemini call failed:", e)
            import traceback; traceback.print_exc()

    # fallback to extractive if Gemini fails
    elapsed_ms = int((time.time() - start_time) * 1000)
    extract_res = extractive_answer_from_chunks(top_docs, query_text, k=1)
    is_faithful = check_faithfulness(extract_res["answer"], context_str)
    return {
        "answer": extract_res["answer"],
        "docs": top_docs,
        "metadatas": extract_res.get("sources", []),
        "scores": top_scores,
        "used_llm": False,
        "response_time_ms": elapsed_ms,
        "llm_used": "Extractive Fallback",
        "is_faithful": is_faithful
    }