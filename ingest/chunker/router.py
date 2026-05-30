"""切碎策略路由:按優先序選擇最合適的 chunker。

優先序:
  1. #AI資料庫 格式  →  hash_delimiter
  2. Q&A 格式       →  qa_chunker
  3. 有結構標題     →  header_chunker
  4. fallback       →  length_chunker
"""
import re

from ingest.chunker.hash_delimiter import split_by_hash, is_hash_format
from ingest.chunker.qa_chunker import split_by_qa, is_qa_format
from ingest.chunker.header_chunker import split_by_headers, has_structural_headers
from ingest.chunker.length_chunker import split_by_length

# hash 格式 fallback 前清除殘留的 @header 行與 # 分隔符
_HASH_HEADER = re.compile(r"@[^\n]*\n?")
_HASH_SEP    = re.compile(r"\s*#\s*")


def _clean_hash_markers(text: str) -> str:
    """移除 AI資料庫格式的 @ 標頭行與 # 分隔符，供 fallback 策略使用。"""
    text = _HASH_HEADER.sub("", text)
    text = _HASH_SEP.sub("\n", text)
    return text.strip()


def chunk_document(text: str, filename: str = "") -> tuple[list[str], str]:
    """選擇 chunker 並切碎,回傳 (chunks, strategy_used)。"""
    if not text or not text.strip():
        return [], "empty"

    # 嘗試 1:hash 格式
    if is_hash_format(filename):
        chunks = split_by_hash(text)
        if chunks:
            return chunks, "hash"
        # hash 格式但 split_by_hash 沒切出東西 → 清掉 @# 標記再往下
        text = _clean_hash_markers(text)

    # 嘗試 2:Q&A 格式
    if is_qa_format(text, filename):
        chunks = split_by_qa(text)
        if chunks:
            return chunks, "qa"

    # 嘗試 3:結構標題
    if has_structural_headers(text):
        chunks = split_by_headers(text)
        if chunks:
            return chunks, "header"

    # 嘗試 4:長度切
    chunks = split_by_length(text)
    return chunks, "length"
