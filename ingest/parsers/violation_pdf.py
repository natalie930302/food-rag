"""北市違規廣告月報的特殊解析器。

格式固定:9 欄表格,每列一個違規案例。
欄位:項次, 裁處書日期, 產品名稱, 來源, 違規情節, 處分商號, 罰鍰金額, 罰則註記, 排名
"""
import re
from pathlib import Path
import pdfplumber


# 從檔名抽年月,例:114年1月份 / 公告115年2月份
FILENAME_DATE_RE = re.compile(r"(11[4-9])年(\d{1,2})月")
# 從法條敘述抽條號,例:食品安全衛生管理法...第28條第1項
ARTICLE_NO_RE = re.compile(r"第\s*(\d+)(?:之\d+)?\s*條")


def parse_violation_pdf(path: Path | str) -> list[dict]:
    """解析一份北市月報 PDF,回傳 list of 案例 dict。"""
    path = Path(path)

    m = FILENAME_DATE_RE.search(path.name)
    if not m:
        return []
    year, month = int(m.group(1)), int(m.group(2))

    cases: list[dict] = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                for table in (page.extract_tables() or []):
                    for row in table:
                        if not row or not row[0]:
                            continue
                        first = str(row[0]).strip()
                        if not first.isdigit():  # 跳過 header
                            continue
                        if len(row) < 8:
                            continue

                        # 罰鍰金額處理
                        penalty_raw = str(row[6] or "")
                        penalty_digits = re.sub(r"[^\d]", "", penalty_raw)
                        penalty = int(penalty_digits) if penalty_digits else 0

                        # 法條條號
                        law_cited = str(row[7] or "").replace("\n", "")
                        am = ARTICLE_NO_RE.search(law_cited)
                        article_no = int(am.group(1)) if am else None

                        cases.append({
                            "year": year,
                            "month": month,
                            "no": int(first),
                            "date": str(row[1] or "").strip(),
                            "product": str(row[2] or "").replace("\n", " ").strip(),
                            "channel": str(row[3] or "").strip(),
                            "violation": str(row[4] or "").replace("\n", "").strip(),
                            "company": str(row[5] or "").replace("\n", "").strip(),
                            "penalty_twd": penalty,
                            "law_cited": law_cited.strip(),
                            "article_no": article_no,
                            "source_file": path.name,
                        })
    except Exception as e:
        print(f"  警告:解析 {path.name} 失敗: {e}")

    return cases
