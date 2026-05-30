"""正常文字 PDF 解析(無表格、無需 OCR)。"""
from pathlib import Path
import pdfplumber

from ingest.normalizer import normalize


def parse_pdf_text(path: Path | str) -> dict:
    """逐頁抽文字後 normalize。"""
    path = Path(path)
    pages_text: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages_text.append(text)

    full = "\n\n".join(pages_text)
    full = normalize(full)
    return {
        "text": full,
        "tables": [],
        "meta": {"n_pages": len(pages_text)},
    }
