"""TXT 檔解析器。

最簡單的 case:直接讀檔。但要注意 BOM 與編碼。
"""
from pathlib import Path
from ingest.normalizer import normalize


def parse_txt(path: Path | str) -> dict:
    """讀取 .txt,回傳 {text, tables (空), meta}。"""
    path = Path(path)
    # 試多種編碼
    for enc in ("utf-8", "utf-8-sig", "big5", "cp950"):
        try:
            raw = path.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raw = path.read_text(encoding="utf-8", errors="ignore")

    return {
        "text": raw,  # 注意:不在這裡 normalize,讓 chunker 看到原始的 # 與 @ 標記
        "tables": [],
        "meta": {"raw_text_len": len(raw)},
    }
