"""
單一入口(/query)的意圖路由:先分流,再決定要用多重的機制。

為什麼不是「所有問題都丟給 agent」:eval/ 量到單跳問題上 tool-calling agent 跟固定管線
打平(31/32 vs 30/32)但延遲約 2 倍、多 1~2 次 LLM 呼叫;只有需要查案例 / 關聯法條的
多步問題,agent 才真的有事做(eval_multihop.py 平均 2.85 次工具呼叫)。所以路由的原則是
Adaptive-RAG(Jeong et al., 2024)那種「依問題複雜度決定用多重的機制」:

  regulation_qa  單跳法規問答           → /ask 的固定管線(確定性、便宜、同樣準)
  case_lookup    找裁罰案例             → /ask 固定管線 + 強制查案例
  ad_review      拿一段文案來審         → /review
  multi_hop      法規 + 案例 / 關聯法條  → /ask_agent(付得起延遲跟成本的地方才用 agent)

兩層:先用關鍵字規則(零成本、確定性)處理訊號明確的問題,規則判不出來的才問一次
gpt-4o-mini(結構化 JSON,temperature 0)。router 自己是新的失效點,所以 eval/eval_router.py
用現成的標籤集(108 題單跳 + 20 題 multi-hop + 手寫的審稿/案例題)量分類準確率,並把
「多步誤判成單跳(會漏掉案例)」跟「單跳誤判成多步(只是變慢)」分開算——兩種錯的代價不對稱。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from config.settings import settings

Intent = Literal["regulation_qa", "case_lookup", "ad_review", "multi_hop"]
INTENTS: tuple[str, ...] = ("regulation_qa", "case_lookup", "ad_review", "multi_hop")
DEFAULT_INTENT = "regulation_qa"

# 規則層只處理訊號很強的情況;eval/eval_router.py 第一版量到規則層 90%、LLM 層 97.7%,
# 錯的幾乎都是規則把「罰款標準怎麼制定」這種法規問題當成案例——所以泛用的罰則字眼
# (罰款/裁罰/處分)不再當成任何一類的證據,一律交給 LLM。
_CASE_STRONG = ("案例", "被罰", "罰過", "紀錄", "實際罰", "開罰過", "處分過", "有沒有人", "有人因")
_PENALTY_WORDS = ("罰款", "罰鍰", "裁罰", "裁處", "處分", "罰則", "罰多少", "開罰")
_REG_STRONG = ("哪條", "哪一條", "哪些規定", "違反", "法源", "規定", "法規", "合法", "合不合法",
               "怎麼標", "如何標", "差在哪", "上限")
_REG_WEAK = ("可以嗎", "需要", "要不要", "算不算", "算是", "是否", "嗎", "呢")
_RELATED_WORDS = ("一起被引用", "一起引用", "關聯條文", "關聯法條", "連動", "常跟哪", "常常跟")
_AD_REVIEW_CUES = ("幫我審", "審稿", "審查這", "這段文案", "這樣寫", "文案可以", "可以這樣寫", "這句話", "這句放",
                   "這句可以", "幫我看這", "檢查這段", "這則廣告", "放在包裝上", "放在網頁", "放在網站")


@dataclass
class RouteDecision:
    intent: str
    source: str            # rules / llm / fallback / forced
    reason: str = ""

    def as_dict(self) -> dict:
        return {"intent": self.intent, "source": self.source, "reason": self.reason}


def route_by_rules(question: str) -> RouteDecision | None:
    """關鍵字規則;訊號不夠明確就回 None 交給 LLM。順序即優先序。"""
    q = question.strip()
    case_strong = any(w in q for w in _CASE_STRONG)
    penalty = any(w in q for w in _PENALTY_WORDS)
    reg_strong = any(w in q for w in _REG_STRONG)
    reg_weak = any(w in q for w in _REG_WEAK)
    if any(w in q for w in _RELATED_WORDS):
        return RouteDecision("multi_hop", "rules", "問到法條之間的關聯,只有 agent 的 search_related_laws 能查")
    if case_strong and reg_strong:
        return RouteDecision("multi_hop", "rules", "同時明確問規定跟過去案例,需要法規 + 案例兩次檢索")
    if any(w in q for w in _AD_REVIEW_CUES):
        return RouteDecision("ad_review", "rules", "使用者拿文案來審")
    if case_strong or penalty:
        return None   # 只提到案例/罰則,是純找案例還是順帶問規定,規則分不出來 → LLM
    if (reg_strong or reg_weak) and ("?" in q or "?" in q):
        return RouteDecision("regulation_qa", "rules", "單純問規定")
    return None


_LLM_PROMPT = """你是食品法規問答系統的路由器。把使用者輸入分成四類之一,只回傳 JSON:
{{"intent": "<類別>", "reason": "<十字內理由>"}}

類別:
- regulation_qa:問某個規定是什麼、能不能做、要不要標示(單一問題,只需查法規)
- case_lookup:只想找過去的裁罰案例、罰款紀錄
- ad_review:使用者貼了一段廣告/文案/標語,想知道能不能這樣寫(輸入本身像文案,不是問句)
- multi_hop:同時要法規依據「和」實際案例/罰款金額,或問法條之間的關聯,需要不只一次檢索

範例:
"天然色素算是食品添加物嗎?需要辦理登錄嗎?" → regulation_qa
"最近有哪些葉黃素廣告被罰?" → case_lookup
"本產品有效改善高血壓,三天見效" → ad_review
"賣益生菌說能提升免疫力違反哪條?有人被罰過嗎、罰多少?" → multi_hop
"食安法第28條常跟哪些條文一起被引用?" → multi_hop

使用者輸入:
\"\"\"{question}\"\"\""""


def route_by_llm(client, question: str) -> RouteDecision:
    resp = client.chat.completions.create(
        model=settings.openai_model,
        temperature=0,
        max_tokens=60,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": _LLM_PROMPT.format(question=question)}],
    )
    try:
        data = json.loads(resp.choices[0].message.content or "")
        intent = data.get("intent")
        if intent in INTENTS:
            return RouteDecision(intent, "llm", str(data.get("reason", ""))[:60])
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    # LLM 回傳壞掉 → 走最便宜、最不會出事的那條(固定管線會自己拒答)
    return RouteDecision(DEFAULT_INTENT, "fallback", "LLM 路由回傳格式無效")


def route(question: str, client=None, use_llm: bool = True) -> RouteDecision:
    decision = route_by_rules(question)
    if decision is not None:
        return decision
    if use_llm and client is not None:
        return route_by_llm(client, question)
    return RouteDecision(DEFAULT_INTENT, "fallback", "規則判不出來且未啟用 LLM 路由")
