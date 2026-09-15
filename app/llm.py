"""LLM 整合層:組裝 prompt + 呼叫 OpenAI。"""
import os
import re
from pathlib import Path

from openai import OpenAI

from app.retrieval import RetrievedCase, RetrievedChunk
from config.settings import settings

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    """讀 prompts/<name>.txt。"""
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _format_chunks(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(沒有檢索到相關法規)"
    parts = []
    for c in chunks:
        law = c.primary_law or "未分類"
        doc = c.document or ""
        ocr = "（OCR 來源，建議核對原始文件）" if c.is_ocr else ""
        label = f"【{law}｜{doc}】{ocr}"
        parts.append(f"{label}\n{c.text}")
    return "\n\n---\n\n".join(parts)


_VIOLATION_MARKERS = (
    "內容述及略以：「", "內容述及略以:「", "其內容宣稱：「", "內容宣稱：「",
    "內容述及略以：", "其內容宣稱：", "宣稱：「", "述及：",
)

def _extract_violation_content(text: str, max_chars: int = 500) -> str:
    """跳過套語（受處分人/網址/下載日期），直取實際違規宣稱內容。"""
    for marker in _VIOLATION_MARKERS:
        pos = text.find(marker)
        if pos != -1:
            content = text[pos + len(marker):]
            return content[:max_chars] + ("..." if len(content) > max_chars else "")
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def _format_cases(cases: list[RetrievedCase]) -> str:
    if not cases:
        return "(沒有相關違規案例)"
    parts = []
    for i, c in enumerate(cases, 1):
        penalty = f"NT${c.penalty_twd:,}" if c.penalty_twd is not None else "不明"
        parts.append(
            f"[案例 {i}] {c.year}年{c.month}月 | "
            f"產品:{c.product} | 廠商:{c.company} | "
            f"罰鍰:{penalty} | 法條:{c.law_cited}\n"
            f"違規情節:{_extract_violation_content(c.violation or '')}"
        )
    return "\n---\n".join(parts)


def build_general_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    cases: list[RetrievedCase],
) -> tuple[str, str]:
    """產生 (system, user) prompt 內容。"""
    system = _load_prompt("system") or "你是食品法規助理,只依據提供的 context 回答。"
    qa_tmpl = _load_prompt("general_qa") or """
【法規 context】
{law_context}

【相關違規案例】
{case_context}

【使用者問題】
{question}

回答要求:
1. 必須引用具體法條(例如「依《食安法》第28條第1項」)
2. 引用案例時標明年月與罰鍰
3. 若 context 不足,直接說「現有資料庫未涵蓋此問題」
4. 不可使用 context 以外的知識
"""
    user = qa_tmpl.format(
        law_context=_format_chunks(chunks),
        case_context=_format_cases(cases),
        question=question,
    )
    return system, user


def build_review_prompt(
    ad_text: str,
    chunks: list[RetrievedChunk],
    cases: list[RetrievedCase],
    matched_keywords: list[str],
) -> tuple[str, str]:
    system = _load_prompt("system_review") or _load_prompt("system") or "你是食品法規助理,只依據提供的 context 回答。"
    tmpl = _load_prompt("ad_review") or """
你是食品廣告審稿專家。請依以下 3 段格式回答:

【廣告文案】
{ad_text}

【偵測到的高風險詞句】
{keywords}

【相關法規 context】
{law_context}

【歷史違規案例】
{case_context}

請依以下三段格式輸出(每段都要有):

一、依據法規:
   說明此文案違反哪一條法規,引用具體條款(食安法第X條第Y項)。

二、判案案例:
   說明過去類似違規的處罰結果(年月、罰鍰金額)。

三、修改建議:
   提供具體的改寫方向,避免醫療效能、誇張或易生誤解。

若文案完全合規,在「一、依據法規」說明「未發現違反現行法規之內容」,
其餘兩段可省略或填「無須建議」。
"""
    user = tmpl.format(
        ad_text=ad_text,
        keywords=", ".join(matched_keywords) if matched_keywords else "(無)",
        law_context=_format_chunks(chunks),
        case_context=_format_cases(cases),
    )
    return system, user

def call_llm(client: OpenAI, system: str, user: str) -> str:
    """呼叫 OpenAI,回傳回答字串。"""
    if os.getenv("LLM_DEBUG", "").lower() in ("1", "true"):
        print("\n" + "="*60)
        print("[SYSTEM]\n" + system)
        print("-"*60)
        print("[USER]\n" + user)
        print("="*60 + "\n")
    resp = client.chat.completions.create(
        model=settings.openai_model,
        temperature=settings.openai_temperature,
        max_tokens=settings.openai_max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


# ============ 廣告審稿關鍵字偵測 ============

# 取自食安法第28條認定準則 Q&A + 健康食品管理法第14條
HIGH_RISK_KEYWORDS = {
    # 食安法第28條：涉及器官功能、外觀改變（只保留 LLM 容易漏判的明確詞）
    "function": [
        "補腦", "增強記憶力",
        "清除自由基", "排毒素", "排出毒素",
        "豐胸", "減肥", "塑身", "纖體", "瘦身",
        "燃燒脂肪", "燃燒體脂", "阻斷澱粉", "快速瘦", "輕鬆瘦",
        "美白",
        # 血脂調節（健食法）— 900731 判例補充
        "清血", "淨血", "宿便",
        # 免疫調節（健食法）— 900731 判例補充；與白名單「維持免疫系統正常運作」不同
        "提高免疫力", "增強抵抗力", "強化免疫力",
        # 外觀／體型
        "雕塑", "解酒",
    ],
    # 食安法第28條：疾病治療/預防/改善，最高優先（命中即 high）
    "medical": [
        "治療", "治癒", "根治", "恢復視力",
        "壯陽", "治失眠",
        "降血壓", "降血糖", "降膽固醇", "降血脂",
        "降低血糖", "降低膽固醇", "降低血脂", "降低血壓",
        "控制血糖", "改善血糖", "改善糖尿病",
        "臨床證實", "臨床實證", "醫學實證",
        "抗癌", "防癌", "抑制腫瘤", "縮小腫瘤", "抑制癌細胞",
        # 900731 判例補充：明確疾病名稱
        "高血脂", "血栓", "動脈硬化", "骨質疏鬆", "老人痴呆",
    ],
    # 健康食品管理法第14條
    "health_food": [
        "健康食品", "衛署健食字", "小綠人標章",
    ],
}

# 合規詞白名單：這些詞出現且無 medical/function keywords 命中時，程式側降為 low
_COMPLIANT_PHRASES = {
    "有助於維持正常視覺功能", "有助於維持免疫系統正常運作",
    "有助於維持正常認知功能", "有助於維持正常代謝功能",
    "有助於減少疲勞感", "維持正常精力", "維持正常精力與活力",
    "幫助維持消化功能順暢", "促進腸道蠕動", "有助於消化道保健",
    "幫助維持肌膚正常代謝", "補充膠原蛋白原料，維持肌膚彈性",
    "適合體重管理計畫期間補充", "低熱量配方，適合控制體重時食用",
    "補充有益心臟的Omega-3", "支持每日心血管保養", "維持心血管正常功能",
}


def detect_risk_keywords(text: str) -> list[str]:
    """掃描廣告文案,回傳命中的高風險詞句。"""
    found = []
    for category, kws in HIGH_RISK_KEYWORDS.items():
        for kw in kws:
            if kw in text:
                found.append(kw)
    return list(dict.fromkeys(found))  # 去重保序


_VERDICT_LINE = re.compile(r"VERDICT:\s*(高風險|有疑慮|合規)")
_LLM_TO_LEVEL = {"高風險": "high", "有疑慮": "medium", "合規": "low"}
_LEVEL_ORDER = {"low": 0, "medium": 1, "high": 2}


def infer_verdict(matched_keywords: list[str], answer: str, ad_text: str = "") -> str:
    """keyword 層與 LLM 層各自判定，取較高風險值。
    例外：無 keyword 命中且廣告文案全部屬於合規白名單用語時，強制降為 low。
    """
    if matched_keywords:
        kw_level = "high" if any(kw in matched_keywords for kw in HIGH_RISK_KEYWORDS["medical"]) else "medium"
    else:
        kw_level = "low"

    m = _VERDICT_LINE.search(answer)
    llm_level = _LLM_TO_LEVEL.get(m.group(1), "low") if m else "low"

    combined = max(kw_level, llm_level, key=lambda x: _LEVEL_ORDER[x])

    # 白名單強制降級：無 keyword + LLM 判 medium + 文案全是合規用語 → low
    if combined == "medium" and kw_level == "low" and ad_text:
        if any(phrase in ad_text for phrase in _COMPLIANT_PHRASES):
            return "low"

    return combined
