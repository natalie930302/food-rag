"""掃描 / 圖文混排 PDF 走 OCR(Tesseract)。

需要系統先安裝:
  - tesseract-ocr
  - tesseract-ocr-chi-tra(繁體中文語言包)
  - poppler-utils(pdf2image 依賴)
"""
from pathlib import Path
import pytesseract
from pdf2image import convert_from_path

from config.settings import settings
from ingest.normalizer import normalize


class OcrError(Exception):
    pass


def parse_pdf_ocr(path: Path | str) -> dict:
    """把 PDF 每頁轉圖,跑 Tesseract OCR。"""
    path = Path(path)

    try:
        kwargs = {"dpi": settings.ocr_dpi}
        if settings.poppler_path:
            kwargs["poppler_path"] = settings.poppler_path
        images = convert_from_path(str(path), **kwargs)
    except Exception as e:
        raise OcrError(f"pdf2image 失敗 ({path.name}): {e}")

    pages_text: list[str] = []
    for img in images:
        try:
            txt = pytesseract.image_to_string(img, lang=settings.tesseract_lang)
        except pytesseract.TesseractNotFoundError:
            raise OcrError(
                "Tesseract 未安裝或不在 PATH。\n"
                "Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-chi-tra\n"
                "macOS:  brew install tesseract tesseract-lang"
            )
        pages_text.append(txt)

    full = "\n\n".join(pages_text)
    full = normalize(full)
    return {
        "text": full,
        "tables": [],
        "meta": {"n_pages": len(images), "is_ocr": True},
    }
