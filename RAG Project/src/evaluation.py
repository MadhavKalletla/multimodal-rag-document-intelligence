# src/evaluation.py
"""
Lightweight RAG Evaluation Metrics without external RAGAS dependency.
Computes:
1. Faithfulness Score: % of answer tokens that appear in context.
2. Relevance Score: Cosine similarity between query and answer embeddings.
3. Context Utilization: How many retrieved chunks contributed to the answer.
"""

import re
from typing import List, Dict, Any
import numpy as np

from src.embedder import embed_texts

def clean_tokens(text: str) -> List[str]:
    """Tokenizes text into lowercase alphanumeric words, filtering out punctuation and empty strings."""
    return re.findall(r'\b\w+\b', text.lower())

def faithfulness_score(answer: str, context_chunks: List[str]) -> float:
    """
    Measures faithfulness: What percentage of non-stopword tokens in the answer
    appear within the retrieved context chunks?
    Returns a score between 0.0 and 1.0.
    """
    ans_text = answer.lower()
    # If the answer is a variation of "I don't know" or "No documents indexed", faithfulness is 100% (1.0)
    if "i don't know" in ans_text or "i do not know" in ans_text or "no documents indexed" in ans_text:
        return 1.0

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

    answer_words = clean_tokens(answer)
    answer_keywords = [w for w in answer_words if w not in stopwords]
    
    if not answer_keywords:
        return 1.0
        
    context_combined = " ".join(context_chunks).lower()
    context_words = set(clean_tokens(context_combined))
    
    match_count = sum(1 for w in answer_keywords if w in context_words)
    return float(match_count / len(answer_keywords))

def relevance_score(query: str, answer: str) -> float:
    """
    Measures semantic relevance between the query and the generated answer
    using cosine similarity of their BGE embeddings.
    Returns a score between -1.0 and 1.0 (typically 0.0 to 1.0).
    """
    if not query.strip() or not answer.strip():
        return 0.0
        
    try:
        # Embed query and answer
        embs = embed_texts([query, answer])
        query_emb = embs[0]
        answer_emb = embs[1]
        
        dot_product = np.dot(query_emb, answer_emb)
        norm_q = np.linalg.norm(query_emb)
        norm_a = np.linalg.norm(answer_emb)
        
        if norm_q == 0 or norm_a == 0:
            return 0.0
            
        sim = dot_product / (norm_q * norm_a)
        # Clip to [0.0, 1.0] for standard visual range
        return float(np.clip(sim, 0.0, 1.0))
    except Exception as e:
        print(f"Error computing relevance score: {e}")
        return 0.0

def context_utilization(answer: str, chunks: List[str]) -> float:
    """
    Measures context utilization: How many retrieved chunks contributed to the answer?
    A chunk is considered utilized if:
    1. Its index (e.g. `[0]`, `[1]`) is cited in the answer text.
    2. Or, it shares at least 3 distinct non-stopword tokens with the answer.
    
    Returns a ratio (number of utilized chunks / total chunks).
    """
    if not chunks:
        return 0.0
        
    # Check for citation markers like [0], [1], [2]
    citations = set(map(int, re.findall(r'\[(\d+)\]', answer)))
    
    stopwords = {"the", "a", "an", "and", "or", "to", "of", "in", "is", "are", "that", "it", "for", "on", "with", "as"}
    answer_words = set(clean_tokens(answer)) - stopwords
    
    utilized_count = 0
    for idx, chunk in enumerate(chunks):
        # 1. Direct citation match
        if idx in citations:
            utilized_count += 1
            continue
            
        # 2. Key word overlap match (minimum 3 words overlap)
        chunk_words = set(clean_tokens(chunk)) - stopwords
        overlap = answer_words.intersection(chunk_words)
        if len(overlap) >= 3:
            utilized_count += 1
            
    return float(utilized_count / len(chunks))

def evaluate_response(query: str, answer: str, chunks: List[str]) -> Dict[str, float]:
    """
    Evaluates a generated response against retrieved chunks.
    """
    return {
        "faithfulness": faithfulness_score(answer, chunks),
        "relevance": relevance_score(query, answer),
        "context_utilization": context_utilization(answer, chunks)
    }
