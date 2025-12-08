# src/preprocess.py
import re
from typing import List

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

def normalize_text(text: str) -> str:
    """
    Basic normalization: unify newlines, remove weird whitespace, and trim.
    """
    if not text:
        return ""
    text = text.replace('\r', '\n')
    # Replace multiple types of whitespace with single space except keep paragraph breaks
    text = re.sub(r'\t|\x0b|\x0c', ' ', text)
    # collapse >2 newlines into exactly two (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # collapse repeated spaces
    text = re.sub(r' {2,}', ' ', text)
    # trim each line
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(lines).strip()
    return text

def split_into_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter: uses punctuation (.,!?). Keeps sentences reasonably long.
    """
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    # remove very short fragments
    parts = [p.strip() for p in parts if p and len(p.strip()) > 10]
    return parts

def simple_chunk(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    """
    Create chunks with sentence boundaries:
    - Normalize text
    - Split into sentences
    - Greedily pack sentences into chunks of <= max_chars (with overlap)
    """
    if not text:
        return []

    text = text.strip()
    # preserve paragraphs as separate blocks if present
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    sentences = []
    for p in paras:
        sents = split_into_sentences(p)
        if not sents:
            # fallback to paragraph if splitting failed
            if len(p) > 0:
                sentences.append(p)
        else:
            sentences.extend(sents)

    chunks = []
    current = ""
    for s in sentences:
        if len(current) + 1 + len(s) <= max_chars:
            current = (current + " " + s).strip() if current else s
        else:
            # push current chunk
            if current:
                chunks.append(current.strip())
            # if sentence itself longer than max_chars, we must slice it
            if len(s) > max_chars:
                start = 0
                while start < len(s):
                    part = s[start:start+max_chars]
                    chunks.append(part.strip())
                    start += max_chars - overlap
                current = ""
            else:
                current = s
    if current:
        chunks.append(current.strip())

    # ensure some overlap by duplicating last N chars into next chunk (simple)
    if overlap and overlap > 0 and len(chunks) > 1:
        out = []
        for i, c in enumerate(chunks):
            if i == 0:
                out.append(c)
            else:
                prev = out[-1]
                # get tail of prev, but do not exceed its length
                tail = prev[-overlap:] if len(prev) > overlap else prev
                # prefix current with tail if not already present
                merged = (tail + " " + c).strip()
                out.append(merged)
        chunks = out

    # final dedupe and trim
    chunks = [c.strip() for c in chunks if c and len(c) > 20]
    return chunks

def clean_and_chunk(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    """
    Normalize and chunk text for indexing / RAG.
    """
    text = normalize_text(text)
    return simple_chunk(text, max_chars=max_chars, overlap=overlap)
