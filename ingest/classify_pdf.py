"""PDF 內容分類器。

判斷 PDF 屬於哪種狀況,決定後續走哪條 parser:
  - normal_text  : 正常文字 PDF       → pdf_text.py
  - has_tables   : 含表格               → pdf_table.py
  - needs_ocr    : 圖文混排或掃描       → pdf_ocr.py
  - empty        : 完全抽不到內容       → failed_files
  - encrypted    : 加密                 → failed_files
  - corrupted    : 損毀                 → failed_files
"""
from pathlib import Path
from typing import Literal, TypedDict
import pdfplumber

from config.settings import settings
from ingest.normalizer import cid_ratio

# 超過此比例的 CID 碼 → 字型無法解碼,改走 OCR
CID_RATIO_THRESHOLD = 0.15


PdfType = Literal["normal_text", "has_tables", "needs_ocr",
                  "empty", "encrypted", "corrupted"]


class PdfClassification(TypedDict):
    type: PdfType
    n_pages: int
    avg_chars_per_page: float
    has_images: bool
    has_tables: bool
    table_pages: list[int]
    error: str | None


def classify_pdf(pdf_path: Path | str) -> PdfClassification:
    """檢測 PDF,回傳分類與輔助資訊。"""
    pdf_path = Path(pdf_path)
    result: PdfClassification = {
        "type": "empty",
        "n_pages": 0,
        "avg_chars_per_page": 0.0,
        "has_images": False,
        "has_tables": False,
        "table_pages": [],
        "error": None,
    }

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            n_pages = len(pdf.pages)
            result["n_pages"] = n_pages
            if n_pages == 0:
                result["type"] = "empty"
                return result

            total_text_chars = 0
            all_text_sample = []
            for page in pdf.pages:
                # 文字
                text = page.extract_text() or ""
                total_text_chars += len(text.strip())
                if len(all_text_sample) < 5:
                    all_text_sample.append(text)

                # 圖片
                if page.images:
                    result["has_images"] = True

                # 表格
                tables = page.extract_tables() or []
                if tables and any(t for t in tables):
                    result["has_tables"] = True
                    result["table_pages"].append(page.page_number)

            avg = total_text_chars / max(n_pages, 1)
            result["avg_chars_per_page"] = avg

            # CID 比例過高 → 字型無法解碼,改走 OCR
            sample_text = "\n".join(all_text_sample)
            if cid_ratio(sample_text) >= CID_RATIO_THRESHOLD:
                result["type"] = "needs_ocr"
                return result

            # 分類規則
            # 文字稀少時一律走 OCR(pdfplumber 的 images 偵測對部分掃描 PDF 不可靠)
            if avg < settings.ocr_min_chars_per_page:
                result["type"] = "needs_ocr"
            elif result["has_tables"]:
                result["type"] = "has_tables"
            else:
                result["type"] = "normal_text"

    except Exception as e:
        msg = str(e).lower()
        if "password" in msg or "encrypt" in msg:
            result["type"] = "encrypted"
        else:
            result["type"] = "corrupted"
        result["error"] = str(e)

    return result
