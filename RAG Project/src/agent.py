# src/agent.py
from typing import List, Tuple
import os
from .retriever import retrieve_top_k

SYSTEM_PROMPT = (
    "You are a precise assistant. Answer using ONLY the provided context. "
    "If the answer isn't in the context, say 'I don't know'. Keep it brief."
)

def _format_context(chunks: List[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        src = c.get("meta", {}).get("source", c.get("id", f"chunk_{i}"))
        blocks.append(f"[{i}] (source: {src})\n{c['doc']}")
    return "\n\n".join(blocks)

def _openai_answer(prompt: str) -> str:
    # Lightweight: uses Chat Completions style; replace model with what you use.
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role":"system", "content": SYSTEM_PROMPT},
                  {"role":"user", "content": prompt}],
        temperature=float(os.getenv("TEMP", "0.1"))
    )
    return resp.choices[0].message.content.strip()

def _hf_answer(prompt: str) -> str:
    # Zero-cost fallback (text2text). Good enough for wiring.
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    model_name = os.getenv("HF_MODEL", "google/flan-t5-small")
    tok = AutoTokenizer.from_pretrained(model_name)
    m = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
    out = m.generate(**inputs, max_new_tokens=256)
    return tok.decode(out[0], skip_special_tokens=True).strip()

def answer_question(question: str, k: int = 4) -> Tuple[str, List[dict]]:
    hits = retrieve_top_k(question, k=k)["matches"]
    context = _format_context(hits)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    use_openai = bool(os.getenv("OPENAI_API_KEY"))
    answer = _openai_answer(prompt) if use_openai else _hf_answer(prompt)
    return answer, hits
