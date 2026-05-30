"""法條引用偵測器。

每個 chunk 切完後掃描內容,找出所有法條引用,
並判斷 role(primary / penalty / definition / reference)。

支援多對多關係:一個 chunk 可關聯多個法條(網狀關聯)。
"""
import re
from dataclasses import dataclass


@dataclass
class LawRef:
    law_name: str           # 食安法 / 健康食品管理法
    article_no: int         # 28
    article_full: str       # "28" 或 "15之一"
    paragraph: str | None   # "1" / "2" / None
    role: str               # primary / penalty / definition / reference


# 主要 pattern:抓「<法律名>...第X條第Y項」
# 為避免吞太多字,中間最多容許 50 字
PATTERN_FOOD_SAFETY = re.compile(
    r"(?:食(?:品)?安(?:全衛生管理)?法)"
    r"[\s\S]{0,50}?"
    r"第\s*(\d+(?:之[一二三四五]|之\d+)?)\s*條"
    r"(?:[\s\S]{0,30}?第\s*(\d+)\s*項)?"
)

PATTERN_HEALTH_FOOD = re.compile(
    r"健康食品管理法"
    r"[\s\S]{0,50}?"
    r"第\s*(\d+(?:之[一二三四五]|之\d+)?)\s*條"
    r"(?:[\s\S]{0,30}?第\s*(\d+)\s*項)?"
)

# 「本法、同法、本準則」:相對指稱,需從 primary_law 推
PATTERN_RELATIVE = re.compile(
    r"(?:本法|同法|本準則|本標準|本辦法)"
    r"[\s\S]{0,30}?"
    r"第\s*(\d+(?:之[一二三四五]|之\d+)?)\s*條"
    r"(?:[\s\S]{0,30}?第\s*(\d+)\s*項)?"
)


# Role 推導:在 match 前後一段範圍內找關鍵字
ROLE_HINTS = {
    "penalty": ["處新臺幣", "處新台幣", "罰鍰", "罰則", "處.*?元"],
    "definition": ["所稱", "用詞.*?定義", "係指", "用語意義"],
    "reference": ["依.*?規定", "參照", "準用", "另.*?規定"],
}


def _chinese_to_int(s: str) -> int:
    """簡單中文數字轉阿拉伯數字(只處理 1-10 + 之X)。"""
    m = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}
    return m.get(s, 0)


def _normalize_article(article_full: str) -> int:
    """從 article_full 取主編號(int)。"""
    base = article_full.split("之")[0]
    try:
        return int(base)
    except ValueError:
        return 0


def _infer_role(text: str, match_start: int, match_end: int) -> str:
    """判斷此 match 在上下文中的角色。"""
    context = text[max(0, match_start - 30): min(len(text), match_end + 30)]
    for role, kws in ROLE_HINTS.items():
        for kw in kws:
            if re.search(kw, context):
                return role
    return "primary"


def detect_laws(chunk_text: str, primary_law_hint: str | None = None) -> list[LawRef]:
    """掃描 chunk,回傳所有偵測到的法條引用。

    primary_law_hint: 從路徑推來的主法條(食安法第28條),
                      用來解讀「本法、同法」相對指稱。
    """
    refs: list[LawRef] = []
    seen: set[tuple[str, str, str | None]] = set()

    def _add(law_name: str, article_full: str, paragraph: str | None,
             match_start: int, match_end: int):
        key = (law_name, article_full, paragraph)
        if key in seen:
            return
        seen.add(key)
        role = _infer_role(chunk_text, match_start, match_end)
        refs.append(LawRef(
            law_name=law_name,
            article_no=_normalize_article(article_full),
            article_full=article_full,
            paragraph=paragraph,
            role=role,
        ))

    # 食安法
    for m in PATTERN_FOOD_SAFETY.finditer(chunk_text):
        _add("食安法", m.group(1), m.group(2), m.start(), m.end())

    # 健食法
    for m in PATTERN_HEALTH_FOOD.finditer(chunk_text):
        _add("健康食品管理法", m.group(1), m.group(2), m.start(), m.end())

    # 相對指稱
    if primary_law_hint:
        # 判斷主法條的法律名
        if "食安法" in primary_law_hint or "食品安全" in primary_law_hint:
            relative_law = "食安法"
        elif "健康食品" in primary_law_hint:
            relative_law = "健康食品管理法"
        else:
            relative_law = None

        if relative_law:
            for m in PATTERN_RELATIVE.finditer(chunk_text):
                _add(relative_law, m.group(1), m.group(2), m.start(), m.end())

    return refs


def primary_law_to_ref(primary_law: str) -> LawRef | None:
    """把 metadata 的 primary_law(食安法第28條)轉成 LawRef。"""
    if not primary_law:
        return None
    m = re.match(r"食安法第(\d+)(?:之([一二三四五]|\d+))?條", primary_law)
    if m:
        article = m.group(1)
        suffix = m.group(2)
        article_full = f"{article}之{suffix}" if suffix else article
        return LawRef(
            law_name="食安法",
            article_no=int(article),
            article_full=article_full,
            paragraph=None,
            role="primary",
        )
    if "健康食品管理法" in primary_law:
        return LawRef(
            law_name="健康食品管理法",
            article_no=0,
            article_full="0",
            paragraph=None,
            role="primary",
        )
    return None
