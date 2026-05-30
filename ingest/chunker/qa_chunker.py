"""Q&A 格式 chunker。

每個 Q+A 配對是一個語意完整單位,應該一起 chunk。
支援多種編號樣式:
  Q1: ... / A1: ...
  Q1. / A1.
  Q1.1 / A1.1
  Q1、/ A1、
  Q1 ... A1 ...(無標點)
"""
import re

from config.settings import settings

# 清除 AI資料庫格式殘留的 # 分隔符、@# 佔位符
_HASH_RESIDUE = re.compile(r"[\s#@]+$")


# Q 編號只在行首才算分割點,避免把表格內的 Q25 誤判為新題目
Q_PATTERN = re.compile(r"^(?P<full>Q\d+(?:\.\d+)*)\s*[:：.。、]?", re.MULTILINE | re.IGNORECASE)

# 句子結尾分割點(用於 QA 塊過長時再切)
SENT_END = re.compile(r"(?<=[。！？\n])")


def _is_inside_table(text: str, pos: int) -> bool:
    """判斷 pos 位置是否落在 markdown 表格行內 (行首是 |)。"""
    line_start = text.rfind("\n", 0, pos)
    line_start = 0 if line_start == -1 else line_start + 1
    return text[line_start:line_start + 1] == "|"


def _split_long_qa(block: str, max_len: int, min_len: int) -> list[str]:
    """QA 塊超過 max_len 時,保留第一個 QA 配對,剩餘按句分段。"""
    if len(block) <= max_len:
        return [block]

    # 找到 A 結尾的位置作為第一個分割點
    # 先嘗試找到 A 回答的句尾,限制在 max_len 以內
    sents = [s for s in SENT_END.split(block) if s]
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + len(s) <= max_len:
            cur += s
        else:
            if cur and len(cur) >= min_len:
                chunks.append(cur.strip())
            cur = s
    if cur and len(cur) >= min_len:
        chunks.append(cur.strip())
    return chunks if chunks else [block[:max_len].strip()]


def split_by_qa(text: str, min_len: int | None = None,
                max_len: int | None = None) -> list[str]:
    """用 Q 編號切 chunk,每個 chunk 包含一組 Q+A。超過 max_len 時再切句。"""
    if min_len is None:
        min_len = settings.chunk_min_len
    if max_len is None:
        max_len = settings.chunk_max_len

    # 找所有行首 Q 編號,跳過表格行內的偽 Q 編號
    matches = [m for m in Q_PATTERN.finditer(text) if not _is_inside_table(text, m.start())]
    if len(matches) < 2:
        return []

    chunks: list[str] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        # 清除 AI資料庫格式在末 Q chunk 末尾帶入的 # @# @ 殘留
        block = _HASH_RESIDUE.sub("", block).strip()
        if len(block) < min_len:
            continue
        if len(block) <= max_len:
            chunks.append(block)
        else:
            chunks.extend(_split_long_qa(block, max_len, min_len))
    return chunks


def is_qa_format(text: str, filename: str = "") -> bool:
    """判斷文本是否為 Q&A 格式。"""
    if any(kw in filename for kw in ("問答", "Q&A", "QA", "Q＆A")):
        return True
    # 行首 Q\d+ 出現次數 >= 5
    real_q = [m for m in Q_PATTERN.finditer(text) if not _is_inside_table(text, m.start())]
    return len(real_q) >= 5
