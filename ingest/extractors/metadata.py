"""從檔案路徑推導 chunk 的 metadata。

路徑樣本:
  食藥署/食安法/食安法第28條/「準則」Q&A.pdf
    → category=食藥署, primary_law=食安法第28條, subtopic=(無)

  食藥署/食安法/食安法第8條/GHP/GHP 液蛋.../液蛋指引.txt
    → category=食藥署, primary_law=食安法第8條, subtopic=GHP/GHP 液蛋...

  食藥署/113年版食品標示法規手冊指引與問答集/113年-食品標示法規手冊.pdf
    → category=食藥署, primary_law=None, subtopic=113年版標示手冊
"""
import re
from pathlib import Path


def derive_metadata(path: Path | str, root: str = "data/raw") -> dict:
    """從相對路徑推 metadata。"""
    path = Path(path)
    # 取得相對於 root 的路徑
    try:
        rel = path.relative_to(root)
    except (ValueError, TypeError):
        rel = path

    parts = list(rel.parts)
    filename = parts[-1] if parts else ""
    parts_no_file = parts[:-1]

    meta = {
        "category": None,
        "primary_law": None,
        "subtopic": None,
        "document": Path(filename).stem,
        "source_path": str(rel),
    }

    if not parts_no_file:
        return meta

    # 第一層 = category(食藥署 / 台北市政府公告... / ...)
    meta["category"] = parts_no_file[0]

    # 找 primary_law:符合「食安法第X條」或「健康食品管理法」等的資料夾
    for p in parts_no_file:
        if re.match(r"食安法第\d+條(?:之[一二三])?", p):
            meta["primary_law"] = p
            break
        if "健康食品管理法" in p:
            meta["primary_law"] = "健康食品管理法"
            break

    # subtopic = primary_law 之後的子資料夾(若有)
    if meta["primary_law"]:
        try:
            idx = parts_no_file.index(meta["primary_law"])
            after = parts_no_file[idx + 1:]
            if after:
                meta["subtopic"] = "/".join(after)
        except ValueError:
            pass
    else:
        # 沒對到法條,把 category 之後的所有層級當 subtopic
        after = parts_no_file[1:]
        if after:
            meta["subtopic"] = "/".join(after)
        elif len(parts_no_file) >= 1 and parts_no_file[0] != "食藥署":
            # 北市違規月報這類
            meta["subtopic"] = parts_no_file[0]

    return meta


def infer_kind(filename: str, primary_law: str | None) -> str:
    """從檔名與法條推導 chunk 類型。"""
    fname_lower = filename.lower()
    if any(k in filename for k in ("問答", "Q&A", "QA", "Q＆A")):
        return "qa"
    if "標準" in filename and "Q" not in filename:
        return "standard"
    if "指引" in filename or "規範" in filename or "原則" in filename:
        return "guide"
    if primary_law and "管理法" in primary_law:
        return "law_text"
    return "guide"
