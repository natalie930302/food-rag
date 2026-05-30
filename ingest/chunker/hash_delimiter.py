"""處理「#AI資料庫.txt」格式的 chunker。

格式約定:
  @文件名 章節標題
  內容...
  #
  @文件名 下一章節
  ...

圖片佔位符: @# 或只有 @ 的行 → 代表該位置有圖片但無法提取文字
"""
import re

from config.settings import settings

# 殘留的 @ 標頭行、@# 圖片佔位符、行首/尾孤立 @
_HEADER_LINE = re.compile(r"@[^\n]*\n?")
_PLACEHOLDER = re.compile(r"@#?")
# hash 格式文件中的 # 分隔符本身不是內容
_TRAILING_HASH = re.compile(r"\s*#\s*$")


def split_by_hash(text: str, min_len: int | None = None) -> list[str]:
    """用 # 切 chunk,清除 chunk 內殘留的 @ 標頭行與圖片佔位符。"""
    if min_len is None:
        min_len = settings.chunk_min_len

    blocks = [b.strip() for b in text.split("#") if b.strip()]
    out: list[str] = []
    for b in blocks:
        # 移除所有 @xxx\n 行（包括 @# 佔位符行）
        cleaned = _HEADER_LINE.sub("", b)
        # 移除剩餘的孤立 @ 或 @#
        cleaned = _PLACEHOLDER.sub("", cleaned)
        # 移除 QA 末尾可能夾帶的 # 分隔符殘留
        cleaned = _TRAILING_HASH.sub("", cleaned).strip()
        if len(cleaned) >= min_len:
            out.append(cleaned)
    return out


def is_hash_format(filename: str) -> bool:
    """判斷是否為 #AI資料庫 / AI整理 格式。"""
    return ("#AI資料庫" in filename) or ("AI整理" in filename) or ("AI資料庫" in filename)
