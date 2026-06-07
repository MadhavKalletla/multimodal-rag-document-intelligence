# src/ocr.py
"""
Production-grade multimodal document extractor.

Supports:
  - PDF  (text-based)  → PyMuPDF fast extraction
  - PDF  (scanned)     → pdf2image + Tesseract OCR fallback
  - Images (PNG/JPG/WEBP/BMP/TIFF) → Pillow preprocessing + Tesseract
  - Plain text (.txt)  → direct read
  - Word docs (.docx)  → python-docx
  - Audio              → placeholder (Whisper-ready)

Cross-platform: uses shutil.which("tesseract") — no Windows hardcoding.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import pytesseract

from .preprocess import normalize_text

logger = logging.getLogger(__name__)

# ── Cross-platform Tesseract detection ────────────────────────────────────────
def _configure_tesseract() -> None:
    """
    Auto-detect Tesseract binary path without hard-coding.
    Falls back to common Windows install location ONLY if shutil.which fails.
    """
    tess_path = shutil.which("tesseract")
    if tess_path:
        pytesseract.pytesseract.tesseract_cmd = tess_path
        logger.debug(f"Tesseract found via PATH: {tess_path}")
        return

    # Windows fallback (non-breaking — just a hint, not hardcoded requirement)
    windows_fallback = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.isfile(windows_fallback):
        pytesseract.pytesseract.tesseract_cmd = windows_fallback
        logger.debug(f"Tesseract found at Windows default: {windows_fallback}")
        return

    logger.warning(
        "Tesseract not found via shutil.which() or Windows default path. "
        "Install tesseract-ocr and ensure it's on PATH. OCR will fail."
    )


_configure_tesseract()

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import fitz  # PyMuPDF
    _HAVE_FITZ = True
except ImportError:
    _HAVE_FITZ = False
    logger.warning("PyMuPDF (fitz) not installed — PDF text extraction disabled.")

try:
    from pdf2image import convert_from_path
    _HAVE_PDF2IMAGE = True
except ImportError:
    convert_from_path = None  # type: ignore[assignment]
    _HAVE_PDF2IMAGE = False
    logger.warning("pdf2image not installed — scanned PDF OCR fallback disabled.")

try:
    import docx  # python-docx
    _HAVE_DOCX = True
except ImportError:
    _HAVE_DOCX = False
    logger.warning("python-docx not installed — .docx extraction disabled.")

# OCR confidence threshold
_OCR_CONFIDENCE_THRESHOLD = 60.0


# ── Image preprocessing ───────────────────────────────────────────────────────
def _preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    """
    Improve OCR accuracy by:
    1. Converting to grayscale
    2. Upscaling to 300 DPI equivalent (2x if small)
    3. Applying Otsu-style thresholding via Pillow

    Returns a binary PIL image ready for Tesseract.
    """
    # 1. Grayscale
    img = img.convert("L")

    # 2. Upscale small images (helps Tesseract accuracy dramatically)
    w, h = img.size
    if w < 1000 or h < 1000:
        scale = max(2, 1500 // min(w, h))
        img = img.resize((w * scale, h * scale), Image.LANCZOS)

    # 3. Sharpening before threshold
    img = img.filter(ImageFilter.SHARPEN)

    # 4. Otsu binarisation via numpy
    arr = np.array(img, dtype=np.uint8)
    # compute histogram-based threshold (simplified Otsu)
    threshold = int(arr.mean())
    arr = ((arr > threshold) * 255).astype(np.uint8)
    img = Image.fromarray(arr)

    return img


# ── OCR with confidence scoring ───────────────────────────────────────────────
def _ocr_pil(img: Image.Image, lang: str = "eng", source_hint: str = "") -> str:
    """
    Run Tesseract OCR on a PIL image with confidence scoring.
    Logs a warning if mean confidence < _OCR_CONFIDENCE_THRESHOLD.
    """
    preprocessed = _preprocess_image_for_ocr(img)

    # Use image_to_data for confidence info
    try:
        data = pytesseract.image_to_data(
            preprocessed,
            lang=lang,
            output_type=pytesseract.Output.DICT,
        )
        # Filter valid confidence values (Tesseract returns -1 for non-text blocks)
        confidences = [
            float(c) for c in data["conf"] if str(c).strip() not in ("-1", "")
        ]
        if confidences:
            mean_conf = sum(confidences) / len(confidences)
            if mean_conf < _OCR_CONFIDENCE_THRESHOLD:
                logger.warning(
                    f"Low OCR confidence ({mean_conf:.1f}%) "
                    f"for '{source_hint or 'image'}'. "
                    "Consider higher-resolution input."
                )

        # Reconstruct text from data (respects word boundaries better)
        words = [
            w for w, c in zip(data["text"], data["conf"])
            if str(c).strip() not in ("-1", "") and w.strip()
        ]
        return " ".join(words)

    except Exception as e:
        logger.error(f"OCR failed for '{source_hint}': {e}")
        # Fallback to simple image_to_string
        try:
            return pytesseract.image_to_string(preprocessed, lang=lang)
        except Exception as e2:
            return f"[OCR error: {e2}]"


# ── Extractors ────────────────────────────────────────────────────────────────
def extract_text_from_image(path: str, lang: str = "eng") -> str:
    """
    Extract text from PNG/JPG/WEBP/BMP/TIFF via Tesseract OCR.
    Applies preprocessing (grayscale, upscale, binarize) before OCR.
    """
    p = Path(path)
    try:
        img = Image.open(p).convert("RGB")
    except Exception as e:
        return f"[Image open error: {e}]"

    raw = _ocr_pil(img, lang=lang, source_hint=p.name)
    return normalize_text(raw)


def extract_text_from_pdf(
    path: str,
    ocr_fallback: bool = True,
    poppler_path: Optional[str] = None,
) -> str:
    """
    Extract text from a PDF file.
    Strategy:
      1. Try PyMuPDF fast text extraction (text-based PDFs)
      2. If extracted text < 50 chars, fall back to pdf2image + Tesseract OCR
    """
    if not _HAVE_FITZ:
        return "[PyMuPDF not installed — cannot extract PDF text]"

    p = Path(path)
    text_parts: List[str] = []

    try:
        with fitz.open(str(p)) as doc:
            for page_num, page in enumerate(doc):
                t = page.get_text("text")
                if t and t.strip():
                    text_parts.append(f"[Page {page_num + 1}]\n{t}")
    except Exception as e:
        text_parts.append(f"[PDF open error: {e}]")

    text = normalize_text("\n".join(text_parts))

    # If text is sparse → likely a scanned PDF → fall back to OCR
    if ocr_fallback and len(text.strip()) < 50:
        if not _HAVE_PDF2IMAGE:
            return text + "\n[Warning: pdf2image not available for OCR fallback]"
        try:
            images = convert_from_path(str(path), dpi=300, poppler_path=poppler_path)
            ocr_pages: List[str] = []
            for i, img in enumerate(images):
                page_text = _ocr_pil(img, source_hint=f"{p.name}:page{i+1}")
                if page_text.strip():
                    ocr_pages.append(f"[Page {i+1}]\n{page_text}")
            text = normalize_text("\n\n".join(ocr_pages))
        except Exception as e:
            text += f"\n[OCR fallback failed: {e}]"

    return text


def extract_text_from_docx(path: str) -> str:
    """
    Extract text from a .docx Word document via python-docx.
    Preserves paragraph breaks.
    """
    if not _HAVE_DOCX:
        return "[python-docx not installed — cannot extract .docx]"
    try:
        doc = docx.Document(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return normalize_text("\n\n".join(paragraphs))
    except Exception as e:
        return f"[DOCX extraction error: {e}]"


def extract_text_from_txt(path: str) -> str:
    """
    Read a plain text file, handling common encodings gracefully.
    """
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as fh:
                return normalize_text(fh.read())
        except UnicodeDecodeError:
            continue
        except Exception as e:
            return f"[Text file read error: {e}]"
    return "[Could not decode text file with utf-8/latin-1/cp1252]"


def transcribe_audio(path: str) -> str:
    """Placeholder — replace with openai-whisper or similar."""
    return f"[Audio transcript placeholder from {path}]"


# ── Unified entry point ───────────────────────────────────────────────────────
def extract_text(path: str, poppler_path: Optional[str] = None) -> str:
    """
    Route a file path to the correct extractor based on extension.
    Returns extracted + normalized text.
    """
    lp = str(path).lower()

    if lp.endswith(".pdf"):
        return extract_text_from_pdf(path, ocr_fallback=True, poppler_path=poppler_path)
    elif lp.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")):
        return extract_text_from_image(path)
    elif lp.endswith(".docx"):
        return extract_text_from_docx(path)
    elif lp.endswith(".txt"):
        return extract_text_from_txt(path)
    elif lp.endswith((".mp3", ".wav", ".m4a", ".flac")):
        return transcribe_audio(path)
    else:
        # Generic fallback — try reading as text
        return extract_text_from_txt(path)


def yield_docs(
    paths: Iterable[str],
    poppler_path: Optional[str] = None,
) -> Iterable[Tuple[str, str, Dict]]:
    """
    Iterate over file paths, yield (path, text, metadata) tuples.
    Metadata includes source filename and file type.
    """
    for p in paths:
        meta = {"source": str(p), "file_type": Path(p).suffix.lower()}
        txt = extract_text(p, poppler_path=poppler_path)
        yield str(p), txt, meta
