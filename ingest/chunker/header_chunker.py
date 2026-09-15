"""依照公文常見結構標題切分。

中文公文標題層級(由高到低):
  壹、貳、參、肆、伍...
  一、二、三、四...
  (一)、(二)、(三)
  1. 2. 3.
  (1) (2) (3)
"""
import re

from config.settings import settings

# 三層標題,從最高到最低
TOP_HEADERS = [
    re.compile(r"^\s*[壹貳參肆伍陸柒捌玖拾]+[、,.]", re.MULTILINE),
    re.compile(r"^\s*[一二三四五六七八九十]+[、,.](?![一二三四五六七八九十])", re.MULTILINE),
    re.compile(r"^\s*[(\(][一二三四五六七八九十]+[)\)]", re.MULTILINE),
]


def has_structural_headers(text: str) -> bool:
    """文本是否有可用的結構標題。"""
    for pat in TOP_HEADERS:
        if len(pat.findall(text)) >= 2:
            return True
    return False


def split_by_headers(text: str,
                      max_len: int | None = None,
                      min_len: int | None = None) -> list[str]:
    """用結構標題切,每段超過 max_len 時再用句號切細。"""
    if max_len is None:
        max_len = settings.chunk_max_len
    if min_len is None:
        min_len = settings.chunk_min_len

    # 找最常見的層級
    best_pat = None
    best_count = 0
    for pat in TOP_HEADERS:
        n = len(pat.findall(text))
        if n > best_count:
            best_count = n
            best_pat = pat
    if best_pat is None or best_count < 2:
        return []

    # 在每個 header 位置切
    matches = list(best_pat.finditer(text))
    chunks_raw: list[str] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunks_raw.append(text[start:end].strip())

    # 過長的 chunk 用句號再切；太短的往下一個合併而非直接丟棄
    chunks: list[str] = []
    pending = ""
    for c in chunks_raw:
        combined = (pending + "\n" + c).strip() if pending else c
        if len(combined) <= max_len:
            if len(combined) >= min_len:
                pending = ""
                chunks.append(combined)
            else:
                # 還不夠長,繼續往下合併
                pending = combined
        else:
            # combined 超過 max_len
            if pending and len(pending) >= min_len:
                chunks.append(pending)
            pending = ""
            if len(c) <= max_len:
                if len(c) >= min_len:
                    chunks.append(c)
                else:
                    pending = c
            else:
                chunks.extend(_split_long(c, max_len, min_len))
    if pending and len(pending) >= min_len:
        chunks.append(pending)
    return chunks


def _split_long(text: str, max_len: int, min_len: int) -> list[str]:
    """過長文本切割：依序嘗試句號 → 段落 → 換行 → 硬截斷。"""
    def _by_pattern(t, pattern):
        """用 pattern 分割後合併成 ≤ max_len 的塊。
        空白部分或切不出多個片段時回傳 []，避免誤判有進展。
        """
        parts = [p for p in re.split(pattern, t) if p.strip()]
        if len(parts) <= 1:  # 沒有實際分割點
            return []
        out, cur = [], ""
        for s in parts:
            if len(cur) + len(s) > max_len and cur:
                if len(cur) >= min_len:
                    out.append(cur.strip())
                cur = s
            else:
                cur += s
        if cur and len(cur) >= min_len:
            out.append(cur.strip())
        return out

    for pattern in [r"(?<=[。！？])", r"\n{2,}", r"\n"]:
        result = _by_pattern(text, pattern)
        if result:
            final = []
            for chunk in result:
                if len(chunk) > max_len and chunk != text:
                    # 還太長且確實有進展，繼續切
                    final.extend(_split_long(chunk, max_len, min_len))
                elif len(chunk) >= min_len:
                    final.append(chunk)
            if final:
                return final

    # 最後手段：硬截斷
    return [text[i:i + max_len].strip()
            for i in range(0, len(text), max_len)
            if len(text[i:i + max_len].strip()) >= min_len]
