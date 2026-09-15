"""DOCX 解析器 — 同時抽段落與表格,表格轉 Markdown。"""
from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph


def table_to_markdown(table: Table) -> str:
    """把 docx Table 轉成 Markdown 表格字串。"""
    rows = []
    for row in table.rows:
        cells = [cell.text.strip().replace("\n", " ").replace("|", "\\|")
                 for cell in row.cells]
        rows.append(cells)
    if not rows:
        return ""

    n_cols = max(len(r) for r in rows)
    # 補齊欄位
    rows = [r + [""] * (n_cols - len(r)) for r in rows]

    md = []
    md.append("| " + " | ".join(rows[0]) + " |")
    md.append("|" + "|".join(["---"] * n_cols) + "|")
    for r in rows[1:]:
        md.append("| " + " | ".join(r) + " |")
    return "\n".join(md)


def parse_docx(path: Path | str) -> dict:
    """解析 .docx,按文件順序輸出段落與表格的 Markdown。"""
    path = Path(path)
    doc = Document(str(path))

    parts: list[str] = []
    tables_md: list[str] = []

    # 遍歷 body 內所有元素(保持原始順序)
    for child in doc.element.body.iterchildren():
        tag = child.tag.split("}", 1)[-1] if "}" in child.tag else child.tag

        if tag == "p":  # 段落
            para = Paragraph(child, doc)
            text = para.text.strip()
            if text:
                parts.append(text)
        elif tag == "tbl":  # 表格
            tbl = Table(child, doc)
            md = table_to_markdown(tbl)
            if md:
                parts.append(md)
                tables_md.append(md)

    return {
        "text": "\n\n".join(parts),
        "tables": tables_md,
        "meta": {"n_tables": len(tables_md)},
    }
