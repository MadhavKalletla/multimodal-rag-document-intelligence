# src/query.py
import os
import traceback
from typing import List, Dict, Any

# GROQ
try:
    from groq import Groq
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
    if GROQ_API_KEY:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    else:
        _groq_client = None
except Exception:
    _groq_client = None
    GROQ_API_KEY = None

# Vector DB wrapper - must provide query(text, top_k) -> dict
from src.vector_store import query as db_query

# SBERT retranking + extractive fallback
try:
    from sentence_transformers import SentenceTransformer, util
    SBERT = SentenceTransformer("all-MiniLM-L6-v2")
    HAVE_SBERT = True
except Exception:
    SBERT = None
    HAVE_SBERT = False

def build_context_from_docs(docs: List[str], max_chars: int = 1500) -> str:
    pieces = []
    total = 0
    for i, d in enumerate(docs):
        snippet = d.strip().replace("\n", " ")
        piece = f"[{i}] {snippet}"
        if total + len(piece) > max_chars:
            break
        pieces.append(piece)
        total += len(piece)
    return "\n\n".join(pieces)

# ---------- rerank helper ----------
def rerank_docs_by_query(query: str, docs: List[str], top_k: int = 4) -> List[Dict[str, Any]]:
    """
    Return reranked docs using SBERT similarity.
    Prevents out-of-range errors when fewer docs exist.
    """
    if not docs:
        return []

    # if fewer docs exist, reduce top_k
    top_k = min(top_k, len(docs))

    if HAVE_SBERT:
        q_emb = SBERT.encode([query], convert_to_tensor=True)[0]
        s_embs = SBERT.encode(docs, convert_to_tensor=True)
        sims = util.cos_sim(q_emb, s_embs)[0]

        # again safe-top_k
        top_k = min(top_k, sims.shape[0])
        top_idxs = sims.topk(top_k).indices.tolist()

        out = []
        for idx in top_idxs:
            out.append({
                "doc": docs[idx],
                "score": float(sims[idx].item()),
                "index": idx
            })
        return out
    
    # fallback: no SBERT
    return [{"doc": docs[i], "score": None, "index": i} for i in range(top_k)]

# ---------- Groq call wrapper ----------
def call_groq_chat(prompt: str, temperature: float = 0.0, max_tokens: int = 350, model_name: str = "llama-3.1-8b-instant") -> str:
    if _groq_client is None:
        raise RuntimeError("Groq client not initialized. Set GROQ_API_KEY and `pip install groq`.")
    messages = [
        {"role": "system", "content": "You are a concise factual assistant. Use ONLY the provided CONTEXT. If the answer is not in the context, say 'I don't know.' Always include source indices in square brackets for any factual claim (e.g. [0], [1]). Keep the answer short (1-3 sentences)."},
        {"role": "user", "content": prompt}
    ]
    resp = _groq_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    # extract text defensively
    try:
        choice = resp.choices[0]
        if hasattr(choice, "message"):
            m = choice.message
            if isinstance(m, dict):
                return m.get("content", "").strip()
            return getattr(m, "content", str(choice)).strip()
        return getattr(choice, "text", str(choice)).strip()
    except Exception:
        return str(resp)

# ---------- extractive answer ----------
def extractive_answer_from_chunks(chunks: List[str], question: str, k: int = 1) -> Dict[str, Any]:
    """
    returns {"answer": str, "sources": [{"chunk_index":i, "text":..., "score":...}, ...]}
    """
    if not chunks:
        return {"answer": "No data", "sources": []}
    if not HAVE_SBERT:
        return {"answer": chunks[0], "sources": [{"chunk_index": 0, "text": chunks[0], "score": None}]}

    sentences = []
    meta = []
    for ci, c in enumerate(chunks):
        for sent in [s.strip() for s in c.split(".") if s.strip()]:
            if len(sent) < 10:
                continue
            sentences.append(sent)
            meta.append((ci, sent))
    if not sentences:
        return {"answer": chunks[0], "sources": [{"chunk_index": 0, "text": chunks[0], "score": None}]}

    sent_embs = SBERT.encode(sentences, convert_to_tensor=True)
    q_emb = SBERT.encode([question], convert_to_tensor=True)[0]
    scores = util.cos_sim(q_emb, sent_embs)[0]
    topk = min(len(sentences), k)
    best_idxs = scores.topk(topk).indices.tolist()
    best_sentences = []
    sources = []
    for idx in best_idxs:
        best_sentences.append(sentences[idx])
        ci, sent = meta[idx]
        sources.append({"chunk_index": ci, "text": sent, "score": float(scores[idx].item())})
    answer = " ".join(best_sentences)
    return {"answer": answer, "sources": sources}

# ---------- convenience wrapper for UI ----------
def answer_from_context(chunks: List[str], question: str) -> str:
    try:
        res = extractive_answer_from_chunks(chunks, question, k=1)
        ans = res.get("answer", "")
        if not ans or not ans.strip():
            return "I don't know."
        return ans
    except Exception:
        return "I don't know."

# ---------- MAIN RAG function ----------
def run_query(query_text: str, top_k: int = 4) -> Dict[str, Any]:
    """
    Returns:
      {
        "answer": str,
        "docs": [...],           # list of retrieved doc texts (ordered by relevance)
        "metadatas": [...],      # corresponding metadatas
        "scores": [...],         # original distances if present
        "used_llm": bool
      }
    """
    # 1) retrieve from DB
    try:
        res = db_query(query_text, top_k=top_k)
        # adapt to common chroma response shapes
        docs = res.get("documents", [[]])[0] if isinstance(res.get("documents", None), list) else res.get("documents", [])
        metadatas = res.get("metadatas", [[]])[0] if isinstance(res.get("metadatas", None), list) else res.get("metadatas", [])
        scores = res.get("distances", [[]])[0] if isinstance(res.get("distances", None), list) else res.get("distances", [])
    except Exception as e:
        print("Retrieval error:", e)
        docs, metadatas, scores = [], [], []

    if not docs:
        return {"answer": "No documents indexed. Please add documents to the Knowledge Base.", "docs": [], "metadatas": [], "scores": [], "used_llm": False}

    # 2) rerank retrieved docs using SBERT (improves precision)
    ranked = rerank_docs_by_query(query_text, docs, top_k=top_k)
    top_docs = [r["doc"] for r in ranked]
    top_scores = [r["score"] for r in ranked]
    # keep corresponding metadata if possible (match by index)
    top_metadatas = []
    for r in ranked:
        idx = r.get("index", None)
        if isinstance(metadatas, list) and idx is not None and idx < len(metadatas):
            top_metadatas.append(metadatas[idx])
        else:
            top_metadatas.append({})

    # 3) build compact context
    context = build_context_from_docs(top_docs, max_chars=1500)

    # 4) call LLM if available (Groq)
    if _groq_client is not None:
        prompt = (
            "You are provided CONTEXT which contains numbered chunks. Use ONLY this context to answer. "
            "If the answer is not in the context, say 'I don't know.' Always include source indices in square brackets for factual statements. "
            "Context:\n\n"
            f"{context}\n\n"
            f"Question: {query_text}\n\n"
            "Answer (short, 1-3 sentences, include source indices):"
        )
        try:
            answer_text = call_groq_chat(prompt, temperature=0.0, max_tokens=300)
            return {"answer": answer_text, "docs": top_docs, "metadatas": top_metadatas, "scores": top_scores, "used_llm": True}
        except Exception as e:
            print("Groq call failed:", e)
            traceback.print_exc()

    # 5) fallback to extractive
    extract_res = extractive_answer_from_chunks(top_docs, query_text, k=1)
    return {"answer": extract_res["answer"], "docs": top_docs, "metadatas": extract_res.get("sources", []), "scores": top_scores, "used_llm": False}
