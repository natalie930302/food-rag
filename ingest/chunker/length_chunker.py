"""長度切分(fallback)。

當文件完全沒結構時使用。固定長度 + overlap。
"""
import re

from config.settings import settings


SENT_END = re.compile(r"(?<=[。!?\n])")


def split_by_length(text: str,
                     chunk_size: int | None = None,
                     overlap: int | None = None,
                     min_len: int | None = None) -> list[str]:
    """用句號優先分句、再合併成接近 chunk_size 的塊。"""
    if chunk_size is None:
        chunk_size = settings.chunk_max_len
    if overlap is None:
        overlap = settings.chunk_overlap
    if min_len is None:
        min_len = settings.chunk_min_len

    if not text or len(text) < min_len:
        return []

    sents = [s for s in SENT_END.split(text) if s.strip()]
    chunks: list[str] = []
    buf = ""
    for s in sents:
        if len(buf) + len(s) <= chunk_size:
            buf += s
        else:
            if buf:
                if len(buf) >= min_len:
                    chunks.append(buf)
                # overlap:把 buf 末尾 overlap 字當下個 chunk 開頭
                if overlap > 0 and len(buf) > overlap:
                    buf = buf[-overlap:] + s
                else:
                    buf = s
                # overlap 接上長句子可能讓 buf 超過 chunk_size，硬切
                while len(buf) > chunk_size:
                    chunks.append(buf[:chunk_size])
                    buf = buf[chunk_size - overlap:] if overlap > 0 else buf[chunk_size:]
            else:
                # 單句太長,硬切
                while len(s) > chunk_size:
                    chunks.append(s[:chunk_size])
                    s = s[chunk_size - overlap:] if overlap > 0 else s[chunk_size:]
                buf = s
    if buf and len(buf) >= min_len:
        chunks.append(buf)
    return chunks
