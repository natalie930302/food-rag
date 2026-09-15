"""
答案層的引用驗證(citation grounding check)。

信心閘門管的是「檢索到的內容可不可信」,但 LLM 拿到可信內容之後,還是可能在答案裡
引用一個檢索內容裡根本沒出現的法條(例如檢索到的是第 22 條的內容,答案卻寫「依食安法
第 28 條」)——這是 confidently wrong 的另一種形態,信心閘門完全管不到。prompts/ 裡雖然
寫了「不得捏造條號」,但 prompt 只是請求不是保證,這裡用程式碼再把關一次:

  1. 從答案抽出所有「第 N 條(之 X)」引用
  2. 每一條檢查是否出現在任何一個提供給 LLM 的 chunk 裡(chunk 內文、或該 chunk 的
     primary_law metadata),阿拉伯數字跟中文數字都算(「第28條」/「第二十八條」)
  3. 回報沒有依據的引用清單;呼叫端可以選擇帶著回饋重新生成一次,或直接標記

只做「引用的條號有沒有出現在 context 裡」這種可機械判定的檢查,不做「引用得對不對」
的語意判斷——後者需要另一個 LLM 當裁判,會引入新的不確定性,先把能百分之百確定的
部分做扎實。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# 「第28條」「第 28 條」「第28條之一」「第二十八條」
_CITATION_RE = re.compile(
    r"第\s*(?P<num>\d{1,3}|[一二三四五六七八九十]{1,4})\s*條"
    r"(?:之\s*(?P<suffix>[一二三四五六七八九十]|\d))?"
)
# 條號前面 12 個字內若出現法律名,一併記下(只用來顯示,不影響驗證);取最靠近條號的那個
_LAW_NAME_RE = re.compile(r"(食安法|食品安全衛生管理法|健康食品管理法|健食法|藥事法|本法|同法)")

_CN_DIGITS = "零一二三四五六七八九"


def _cn_to_int(s: str) -> int:
    """一~九十九 的中文數字轉整數。"""
    if s.isdigit():
        return int(s)
    if s == "十":
        return 10
    if "十" in s:
        tens, _, ones = s.partition("十")
        return (_CN_DIGITS.index(tens) if tens else 1) * 10 + (_CN_DIGITS.index(ones) if ones else 0)
    return _CN_DIGITS.index(s)


def _int_to_cn(n: int) -> str:
    if n < 10:
        return _CN_DIGITS[n]
    if n < 20:
        return "十" + (_CN_DIGITS[n % 10] if n % 10 else "")
    return _CN_DIGITS[n // 10] + "十" + (_CN_DIGITS[n % 10] if n % 10 else "")


@dataclass(frozen=True)
class Citation:
    article: int
    suffix: str | None = None   # 「之一」的「一」
    law: str | None = None      # 答案裡寫的法律名(可能沒寫)

    @property
    def label(self) -> str:
        s = f"第{self.article}條"
        if self.suffix:
            s += f"之{self.suffix}"
        return (self.law or "") + s

    def surface_forms(self) -> list[str]:
        """這條在 chunk 文字裡可能出現的寫法(不含法律名,避免太嚴)。"""
        suffix = f"之{self.suffix}" if self.suffix else ""
        forms = [f"第{self.article}條{suffix}", f"第 {self.article} 條{suffix}",
                 f"第{_int_to_cn(self.article)}條{suffix}"]
        if self.suffix:
            # 「第15之一條」這種寫法
            forms.append(f"第{self.article}{suffix}條")
        return forms


@dataclass
class CitationCheck:
    citations: list[Citation] = field(default_factory=list)
    unsupported: list[Citation] = field(default_factory=list)

    @property
    def supported(self) -> bool:
        return not self.unsupported


def extract_citations(answer: str) -> list[Citation]:
    seen: dict[tuple, Citation] = {}
    for m in _CITATION_RE.finditer(answer):
        num = m.group("num")
        try:
            article = _cn_to_int(num)
        except ValueError:
            continue
        suffix = m.group("suffix")
        if suffix and suffix.isdigit():
            suffix = _int_to_cn(int(suffix))
        laws = _LAW_NAME_RE.findall(answer[max(0, m.start() - 12):m.start()])
        c = Citation(article=article, suffix=suffix, law=laws[-1] if laws else None)
        key = (c.article, c.suffix)
        if key not in seen:
            seen[key] = c
    return list(seen.values())


def _chunk_supports(citation: Citation, text: str, primary_law: str | None) -> bool:
    haystack = (text or "") + "\n" + (primary_law or "")
    return any(form in haystack for form in citation.surface_forms())


def verify_citations(answer: str, chunks) -> CitationCheck:
    """chunks:任何有 .text 與 .primary_law 的物件(RetrievedChunk),或 dict。"""
    citations = extract_citations(answer)
    contexts = []
    for c in chunks:
        if isinstance(c, dict):
            contexts.append((c.get("text", ""), c.get("primary_law")))
        else:
            contexts.append((getattr(c, "text", ""), getattr(c, "primary_law", None)))
    unsupported = [
        cit for cit in citations
        if not any(_chunk_supports(cit, text, law) for text, law in contexts)
    ]
    return CitationCheck(citations=citations, unsupported=unsupported)


def feedback_for_regeneration(check: CitationCheck) -> str:
    """給 LLM 的回饋文字,要求它只根據提供的內容重寫。"""
    labels = "、".join(c.label for c in check.unsupported)
    return (
        f"你上一版回答引用了 {labels},但這些條號並沒有出現在提供的法規內容裡。"
        "請重新回答,只引用提供內容裡確實出現的條號;如果提供的內容沒有明確條號,"
        "就不要寫條號,直接說明依據來自哪份文件。"
    )
