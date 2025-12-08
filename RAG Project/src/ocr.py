from __future__ import annotations
from typing import Iterable, Tuple, Dict, List, Optional
from pathlib import Path
import fitz
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None
import pytesseract
from .preprocess import normalize_text

def _ocr_pil(img: Image.Image, lang: str = 'eng') -> str:
    try:
        return pytesseract.image_to_string(img, lang=lang)
    except Exception as e:
        return f"[OCR error: {e}]"

def extract_text_from_pdf(path: str, ocr_fallback: bool = True, poppler_path: Optional[str] = None) -> str:
    p = Path(path)
    text_parts: List[str] = []
    try:
        with fitz.open(p) as doc:
            for page in doc:
                t = page.get_text('text')
                if t and t.strip():
                    text_parts.append(t)
    except Exception as e:
        text_parts.append(f"[PDF open error: {e}]")
    text = normalize_text('\n'.join(text_parts))
    if ocr_fallback and len(text.strip()) < 50:
        if convert_from_path is None:
            return text + '\n[Warning] pdf2image not available for OCR fallback.'
        try:
            images = convert_from_path(path, dpi=300, poppler_path=poppler_path)
            ocr_texts = [_ocr_pil(img) for img in images]
            text = normalize_text('\n\n'.join(ocr_texts))
        except Exception as e:
            text += f"\n[OCR fallback failed: {e}]"
    return text

def extract_text_from_image(path: str, lang: str = 'eng') -> str:
    p = Path(path)
    try:
        img = Image.open(p).convert('RGB')
    except Exception as e:
        return f"[Image open error: {e}]"
    txt = _ocr_pil(img, lang=lang)
    return normalize_text(txt)

def transcribe_audio(path: str) -> str:
    return f"[Audio transcript placeholder from {path}]"

def yield_docs(paths: Iterable[str], poppler_path: Optional[str] = None) -> Iterable[Tuple[str, str, Dict]]:
    for p in paths:
        lp = str(p).lower()
        meta = {'source': str(p)}
        if lp.endswith(('.pdf',)):
            txt = extract_text_from_pdf(p, ocr_fallback=True, poppler_path=poppler_path)
            yield str(p), txt, meta
        elif lp.endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')):
            txt = extract_text_from_image(p)
            yield str(p), txt, meta
        elif lp.endswith(('.mp3', '.wav', '.m4a', '.flac')):
            txt = transcribe_audio(p)
            yield str(p), txt, meta
        else:
            try:
                with open(p, 'r', encoding='utf-8', errors='ignore') as fh:
                    txt = fh.read()
            except Exception as e:
                txt = f"[Open error: {e}]"
            yield str(p), normalize_text(txt), meta
