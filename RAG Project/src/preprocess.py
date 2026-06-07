# src/preprocess.py
"""
Text preprocessing and chunking for the RAG pipeline.

Keeps original normalize_text / split_into_sentences / simple_chunk / clean_and_chunk
and adds:
  - tiktoken-based token counting (fallback to char/4 estimate if not installed)
  - chunk_by_tokens()  — token-aware chunking with 512-token default
  - chunk_with_metadata() — returns List[Dict] with text + metadata per chunk
"""

import re
from typing import List, Dict, Any

# ── tiktoken (optional — graceful fallback) ───────────────────────────────────
try:
    import tiktoken
    _TOKENIZER = tiktoken.get_encoding("cl100k_base")   # GPT-4 / BGE compatible
    _HAVE_TIKTOKEN = True
except Exception:
    _TOKENIZER = None
    _HAVE_TIKTOKEN = False

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


# ── Token utilities ───────────────────────────────────────────────────────────
def count_tokens(text: str) -> int:
    """Count tokens using tiktoken; fall back to char/4 estimate."""
    if _HAVE_TIKTOKEN and _TOKENIZER is not None:
        return len(_TOKENIZER.encode(text))
    return max(1, len(text) // 4)


# ── Original functions (preserved + improved) ─────────────────────────────────
def normalize_text(text: str) -> str:
    """
    Basic normalization: unify newlines, remove weird whitespace, trim.
    """
    if not text:
        return ""
    text = text.replace('\r', '\n')
    text = re.sub(r'\t|\x0b|\x0c', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(lines).strip()
    return text


def split_into_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter using punctuation (. ! ?).
    Keeps sentences reasonably long (> 10 chars).
    """
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
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
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    sentences = []
    for p in paras:
        sents = split_into_sentences(p)
        if not sents:
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
            if current:
                chunks.append(current.strip())
            if len(s) > max_chars:
                start = 0
                while start < len(s):
                    part = s[start:start + max_chars]
                    chunks.append(part.strip())
                    start += max_chars - overlap
                current = ""
            else:
                current = s
    if current:
        chunks.append(current.strip())

    # Overlap: prefix each chunk with tail of previous chunk
    if overlap and overlap > 0 and len(chunks) > 1:
        out = []
        for i, c in enumerate(chunks):
            if i == 0:
                out.append(c)
            else:
                prev = out[-1]
                tail = prev[-overlap:] if len(prev) > overlap else prev
                merged = (tail + " " + c).strip()
                out.append(merged)
        chunks = out

    chunks = [c.strip() for c in chunks if c and len(c) > 20]
    return chunks


def clean_and_chunk(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    """
    Normalize and chunk text for indexing / RAG.
    Returns list of plain text chunk strings.
    """
    text = normalize_text(text)
    return simple_chunk(text, max_chars=max_chars, overlap=overlap)


# ── Token-aware chunking ──────────────────────────────────────────────────────
def chunk_by_tokens(
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
) -> List[str]:
    """
    Chunk text respecting token boundaries (using tiktoken).
    Falls back to character-based estimation if tiktoken unavailable.

    Parameters
    ----------
    text         : normalized input text
    max_tokens   : maximum tokens per chunk (default 512)
    overlap_tokens: tokens to carry forward into next chunk for continuity
    """
    if not text or not text.strip():
        return []

    if _HAVE_TIKTOKEN and _TOKENIZER is not None:
        token_ids = _TOKENIZER.encode(text)
        chunks: List[str] = []
        start = 0
        while start < len(token_ids):
            end = min(start + max_tokens, len(token_ids))
            chunk_ids = token_ids[start:end]
            chunk_text = _TOKENIZER.decode(chunk_ids).strip()
            if chunk_text:
                chunks.append(chunk_text)
            start += max_tokens - overlap_tokens
        return [c for c in chunks if len(c) > 20]

    # Fallback: char-based (4 chars ≈ 1 token)
    max_chars   = max_tokens * 4
    overlap_chars = overlap_tokens * 4
    return simple_chunk(text, max_chars=max_chars, overlap=overlap_chars)


# ── Metadata-tagged chunking ──────────────────────────────────────────────────
def chunk_with_metadata(
    text: str,
    source: str = "unknown",
    page_num: int = 0,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
) -> List[Dict[str, Any]]:
    """
    Chunk text and attach rich metadata to each chunk.

    Returns
    -------
    List of dicts:
    {
        "text":        str,   # chunk content
        "source":      str,   # original filename
        "page_num":    int,   # page number (0 if unknown)
        "chunk_index": int,   # 0-based index within this document
        "char_count":  int,   # character count
        "token_count": int,   # estimated token count
    }
    """
    normalized = normalize_text(text)
    raw_chunks  = chunk_by_tokens(normalized, max_tokens=max_tokens, overlap_tokens=overlap_tokens)

    result: List[Dict[str, Any]] = []
    for i, chunk_text in enumerate(raw_chunks):
        result.append({
            "text":        chunk_text,
            "source":      source,
            "page_num":    page_num,
            "chunk_index": i,
            "char_count":  len(chunk_text),
            "token_count": count_tokens(chunk_text),
        })
    return result
