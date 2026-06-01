"""LLM 整合層:組裝 prompt + 呼叫 OpenAI。"""
import os
import re
from pathlib import Path

from openai import OpenAI

from config.settings import settings
from app.retrieval import RetrievedChunk, RetrievedCase


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
            f"違規情節:{c.violation[:120]}{'...' if c.violation and len(c.violation) > 120 else ''}"
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
    system = _load_prompt("system") or "你是食品法規助理,只依據提供的 context 回答。"
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
    import os
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

# 取自食安法第28條認定準則 Q&A + 藥事法 + 健康食品管理法
HIGH_RISK_KEYWORDS = {
    # 食安法第28條 Q1：涉及維持或改變人體器官、組織、生理或外觀
    "function": [
        "保護眼睛", "增加血管彈性", "增強抵抗力", "強化細胞功能",
        "增智", "補腦", "增強記憶力", "改善體質", "解酒",
        "清除自由基", "排毒素", "分解有害物質",
        "改善更年期障礙", "平胃氣", "防止口臭",
        "豐胸", "預防乳房下垂", "減肥", "塑身", "增高",
        "使頭髮烏黑", "延遲衰老", "防止老化", "改善皺紋",
        "美白", "纖體", "瘦身",
    ],
    # 食安法第28條 Q3：涉及預防、改善、減輕、診斷或治療疾病
    "medical": [
        "治療", "恢復視力", "防止便秘", "利尿", "改善過敏體質",
        "壯陽", "強精", "減輕過敏", "治失眠", "防止貧血",
        "降血壓", "改善血濁", "清血", "調整內分泌",
        "防止更年期", "消滯", "降肝火", "改善喉嚨發炎",
        "祛痰止喘", "消腫止痛", "消除心律不整", "解毒",
        "降血糖", "降膽固醇", "降血脂",
    ],
    # 藥事法第65-66條：宣稱藥品療效
    "drug": [
        "藥效", "藥用", "處方", "醫師推薦", "臨床證實", "醫學實證",
        "藥理作用", "抗癌", "防癌", "抑制腫瘤", "消炎止痛",
        "退燒", "抗菌", "殺菌", "抗病毒", "增強免疫力",
    ],
    # 健康食品管理法第14條：未經認證宣稱健康食品
    "health_food": [
        "健康食品", "衛署健食字", "小綠人標章",
        "經衛生福利部認證", "通過衛福部審核",
    ],
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


def infer_verdict(matched_keywords: list[str], answer: str) -> str:
    """keyword 層與 LLM 層各自判定，取較高風險值。"""
    if matched_keywords:
        high_cats = HIGH_RISK_KEYWORDS["medical"] + HIGH_RISK_KEYWORDS["drug"]
        kw_level = "high" if any(kw in matched_keywords for kw in high_cats) else "medium"
    else:
        kw_level = "low"

    m = _VERDICT_LINE.search(answer)
    llm_level = _LLM_TO_LEVEL.get(m.group(1), "low") if m else "low"

    return max(kw_level, llm_level, key=lambda x: _LEVEL_ORDER[x])
