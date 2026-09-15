"""含表格 PDF 解析。

把表格抽出轉 Markdown,同時保留原始文字脈絡。
核心策略:用 find_tables() 取得表格 bounding box,
只排除「資料表格」的 bbox,「版面表格」仍由 extract_text() 自然抽取。
"""
from pathlib import Path

import pdfplumber

from ingest.normalizer import normalize


def _is_layout_table(table: list[list]) -> bool:
    """偵測 PDF 排版用的版面表格（非資料表格）。

    版面表格特徵：恰好 2 欄，且其中一欄 60% 以上的列是空的。
    常見於 PDF 用雙欄表格排版 QA 問題文字（Q 文字分兩欄換行）。
    這種表格不應轉成 markdown，應讓 extract_text() 自然抽取。
    """
    if not table:
        return False
    n_cols = max((len(row) for row in table), default=0)
    if n_cols != 2:
        return False
    n = len(table)
    col0_empty = sum(1 for row in table if not str(row[0] or "").strip())
    col1_empty = sum(1 for row in table
                     if len(row) < 2 or not str(row[1] or "").strip())
    return col0_empty / n >= 0.6 or col1_empty / n >= 0.6


def is_valid_table(table: list[list]) -> bool:
    """表格品質檢查。版面表格一律視為無效。"""
    if not table or len(table) < 2:
        return False
    if any(not row or len(row) < 2 for row in table):
        return False
    if _is_layout_table(table):
        return False
    cells = [c for row in table for c in row]
    if not cells:
        return False
    non_empty = sum(1 for c in cells if c and str(c).strip())
    return non_empty / len(cells) >= 0.3


def table_to_markdown(table: list[list]) -> str:
    """二維 list → Markdown 表格字串。"""
    if not table:
        return ""
    cleaned = []
    for row in table:
        cleaned.append([
            (str(c) if c is not None else "")
                .replace("\n", " ")
                .replace("|", "\\|")
                .strip()
            for c in row
        ])
    n_cols = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (n_cols - len(r)) for r in cleaned]

    md = []
    md.append("| " + " | ".join(cleaned[0]) + " |")
    md.append("|" + "|".join(["---"] * n_cols) + "|")
    for r in cleaned[1:]:
        md.append("| " + " | ".join(r) + " |")
    return "\n".join(md)


def _extract_text_outside_tables(page, valid_bboxes: list[tuple]) -> str:
    """排除「資料表格」bbox 後抽純文字。版面表格不排除，讓 extract_text() 自然處理。"""
    if not valid_bboxes:
        return page.extract_text() or ""

    def not_in_any_table(obj):
        for x0, top, x1, bottom in valid_bboxes:
            if (obj.get("x0", 0) >= x0 and obj.get("x1", 0) <= x1
                    and obj.get("top", 0) >= top and obj.get("bottom", 0) <= bottom):
                return False
        return True

    return page.filter(not_in_any_table).extract_text() or ""


def parse_pdf_with_tables(path: Path | str) -> dict:
    """逐頁:先抽非資料表格文字,再附上乾淨的資料表格 markdown。"""
    path = Path(path)
    parts: list[str] = []
    tables_md: list[dict] = []

    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            found_tables = page.find_tables()

            # 只排除「真正的資料表格」bbox
            valid_table_data = []
            valid_bboxes = []
            for t_obj in found_tables:
                raw = t_obj.extract()
                if is_valid_table(raw):
                    valid_table_data.append((t_obj, raw))
                    valid_bboxes.append(t_obj.bbox)

            text = _extract_text_outside_tables(page, valid_bboxes)

            for t_obj, raw in valid_table_data:
                md = table_to_markdown(raw)
                if md:
                    text += "\n\n" + md
                    tables_md.append({
                        "page": page.page_number,
                        "markdown": md,
                        "n_rows": len(raw),
                        "n_cols": len(raw[0]) if raw else 0,
                    })

            parts.append(text)

    full = "\n\n".join(parts)
    full = normalize(full)
    return {
        "text": full,
        "tables": tables_md,
        "meta": {"n_pages": len(parts), "n_tables": len(tables_md)},
    }
