"""統一的文字清理工具。

不同來源(PDF/DOCX/TXT/OCR)抽出來的文字常見問題:
  - 多餘的空白與換行
  - 頁碼、頁首頁尾
  - 全形/半形不一致
  - 不可見字元
這個模組統一清理。
"""
import re
import unicodedata


# (cid:NNN) — PDF 字型無 ToUnicode mapping 時的殘留碼
CID_PATTERN = re.compile(r"\(cid:\d+\)")

# 常見頁碼模式(中文公文常用)
PAGE_NUMBER_PATTERNS = [
    re.compile(r"第\s*\d+\s*頁[,，]\s*共\s*\d+\s*頁"),
    re.compile(r"-\s*\d+\s*-"),
    re.compile(r"\b\d+\s*/\s*\d+\b"),
    re.compile(r"^\s*\d{1,3}\s*$", re.MULTILINE),          # 單獨一行的純數字
    re.compile(r"^\s*\d{1,3}-\d{1,3}\s*$", re.MULTILINE),  # 3-14 型章節頁碼
    re.compile(r"^\d{1,3}\)\s*$", re.MULTILINE),            # 17) 型頁尾殘留
    re.compile(r"~\s*\d+\s*~"),                             # ~ 25 ~ 波浪號頁碼
    re.compile(r"^\s*\|\s*$", re.MULTILINE),                # 頁碼清除後殘留的孤立 |
]

# AI資料庫格式的圖片佔位符
AI_PLACEHOLDER_PATTERN = re.compile(r"@#|(?<!\S)@(?!\S)")


def normalize_unicode(text: str) -> str:
    """Unicode 正規化:NFKC 會把全形數字、英文字母統一成半形。"""
    return unicodedata.normalize("NFKC", text)


def remove_cid_codes(text: str) -> str:
    """移除 PDF 字型無法解碼的 CID 殘留碼 (cid:NNN)。"""
    return CID_PATTERN.sub("", text)


def remove_ai_placeholders(text: str) -> str:
    """移除 AI資料庫格式的圖片佔位符 @# 與孤立的 @。"""
    return AI_PLACEHOLDER_PATTERN.sub("", text)


def cid_ratio(text: str) -> float:
    """計算 CID 碼佔原始文字的比例,用於判斷是否需要 OCR。"""
    if not text:
        return 0.0
    cid_chars = sum(len(m.group()) for m in CID_PATTERN.finditer(text))
    return cid_chars / len(text)


def remove_page_numbers(text: str) -> str:
    """移除常見頁碼樣式。"""
    for pat in PAGE_NUMBER_PATTERNS:
        text = pat.sub("", text)
    return text


def merge_broken_lines(text: str) -> str:
    """合併不該斷的換行。

    中文段落如果在中間斷行(非標點結尾),通常是 PDF 抽取造成的偽斷行。
    """
    # 兩個連續換行 → 段落分隔保留
    # 單一換行 + 下一行不是以中文標點/數字編號開頭 → 合併
    lines = text.split("\n")
    result = []
    buf = ""
    for line in lines:
        line = line.rstrip()
        if not line:
            if buf:
                result.append(buf)
                buf = ""
            result.append("")
            continue

        # 判斷新段落的開頭
        is_new_para = bool(re.match(
            r"^(?:[壹貳參肆伍陸柒捌玖拾零一二三四五六七八九十]+[、.,]|"
            r"[(\(][一二三四五六七八九十0-9]+[)\)]|"
            r"\d+[、.,]|"
            r"Q\d+[:.：]|A\d+[:.：]|"
            r"第\s*[一二三四五六七八九十0-9]+\s*[條章節項款]|"
            r"\|)",   # markdown 表格行不合併到前一行
            line,
        ))

        if buf and is_new_para:
            result.append(buf)
            buf = line
        elif buf:
            # 中文行末非標點,直接接;有標點,加空格
            if re.search(r"[。!?;:,、:]$", buf):
                result.append(buf)
                buf = line
            else:
                buf += line
        else:
            buf = line

    if buf:
        result.append(buf)
    return "\n".join(result)


def fix_spaced_cjk(text: str) -> str:
    """修正 PDF 逐欄抽取造成的中文字間多餘空格。

    「食 品 負 責 廠 商」→「食品負責廠商」
    條件：連續 3 個以上「CJK字 空格」模式才觸發，避免誤刪有意義的空格。
    """
    return re.sub(
        r"((?:[一-鿿㐀-䶿豈-﫿] ){2,}[一-鿿㐀-䶿豈-﫿])",
        lambda m: m.group().replace(" ", ""),
        text,
    )


def remove_excess_whitespace(text: str) -> str:
    """壓縮過多空白與空行。"""
    # 多個空格 → 單一空格
    text = re.sub(r"[ \t]+", " ", text)
    # 三個以上換行 → 兩個
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_invisible_chars(text: str) -> str:
    """移除零寬字元、BOM 等。"""
    return text.replace("\ufeff", "").replace("\u200b", "").replace("\u00a0", " ")


def normalize(text: str) -> str:
    """執行完整清理流程。"""
    if not text:
        return ""
    text = remove_invisible_chars(text)
    text = remove_cid_codes(text)
    text = remove_ai_placeholders(text)
    text = normalize_unicode(text)
    text = remove_page_numbers(text)
    text = merge_broken_lines(text)
    text = fix_spaced_cjk(text)
    text = remove_excess_whitespace(text)
    return text
