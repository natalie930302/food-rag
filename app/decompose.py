"""複合問題的拆解與查詢字串的形態修正。

2026/09 實測發現兩個檢索層的盲點,單跳評估集(每題一問)量不到:

1. 檢索對「問法」極度敏感。同一件事,「減肥廣告違反哪一條?」reranker 0.91、過閘門;
   agent 濃縮成關鍵字「減肥廣告」只剩 0.27;整句三連問「…哪條?罰多少?有案例嗎?」只剩 0.17。
   cross-encoder 是拿「問句 vs 段落」訓練的,關鍵字與多問句都不是它的輸入分佈。
2. 多重問題不論走固定管線還是 agent,送進檢索的都是壞查詢:固定管線送整句,agent 送關鍵字。

所以:
- split_subquestions:先用問號切成子問題(零成本);子問題太短、少了主詞(「罰多少?」)時用一次 LLM
  改寫成各自獨立、可單獨檢索的問句(decompose_with_llm)。
- keyword_to_question:工具端保證——LLM 給關鍵字就補成問句再查,不靠 prompt 請它別給關鍵字。
"""
from __future__ import annotations

import json
import re

from config.settings import settings

_SPLIT = re.compile(r"[?？]+")
_TRIM = " \t\r\n,，;；、。"
_MIN_SUB_LEN = 2          # 短於這個長度的片段不算子問題(尾端的空字串、單獨的「呢」)
_SHORT_SUB_LEN = 8        # 短於這個長度的子問題多半沒主詞(「罰多少」「超過會怎樣」),要 LLM 補
_KEYWORD_MAX_LEN = 12
_QUESTION_MARKS = ("?", "?", "嗎", "呢", "什麼", "甚麼", "哪", "如何", "怎麼", "怎樣", "是否", "多少", "為何", "為什麼", "會不會", "能不能", "可不可以", "要不要")


def split_subquestions(question: str) -> list[str]:
    """按問號切;切不出兩個以上有內容的片段就原句回傳(單一問題)。"""
    parts = [p.strip(_TRIM) for p in _SPLIT.split(question)]
    parts = [p for p in parts if len(p) >= _MIN_SUB_LEN]
    if len(parts) < 2:
        return [question.strip()]
    return [p + "?" for p in parts]


def is_compound(question: str) -> bool:
    return len(split_subquestions(question)) >= 2


def looks_like_keywords(query: str) -> bool:
    """「減肥廣告」「食品添加物 標示」這類:短、沒有任何問句記號。"""
    q = query.strip()
    if not q or len(q) > _KEYWORD_MAX_LEN:
        return False
    return not any(m in q for m in _QUESTION_MARKS)


def keyword_to_question(query: str) -> str:
    """關鍵字 → 檢索器吃得下的問句。不改語意,只補問句外殼。"""
    q = query.strip(_TRIM)
    return f"{q}的相關規定是什麼?"


_DECOMPOSE_PROMPT = """把下面這個一次問了好幾件事的問題,拆成幾個各自獨立、單獨看也知道在問什麼的子問題。
規則:每個子問題都要把主詞補齊(例如「罰多少?」要寫成「減肥廣告違規會罰多少錢?」);不新增原問題沒有的問題;
保持原本的順序;只回傳 JSON:{{"subquestions": ["...", "..."]}}

問題:
\"\"\"{question}\"\"\""""


def decompose_with_llm(client, question: str) -> list[str]:
    """一次 LLM 呼叫把複合問題拆成獨立子問題;失敗就退回規則切法。"""
    fallback = split_subquestions(question)
    try:
        resp = client.chat.completions.create(
            model=settings.openai_model,
            temperature=0,
            max_tokens=300,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": _DECOMPOSE_PROMPT.format(question=question)}],
        )
        data = json.loads(resp.choices[0].message.content or "")
        subs = [str(s).strip() for s in data.get("subquestions", []) if str(s).strip()]
    except (json.JSONDecodeError, AttributeError, TypeError, KeyError):
        return fallback
    if not subs or len(subs) > 6:
        return fallback
    return [s if s.endswith(("?", "?")) else s + "?" for s in subs]


def decompose(client, question: str) -> tuple[list[str], bool]:
    """對外入口:單一問題 → ([原句], False);複合問題 → (子問題列表, 是否用了 LLM)。
    只有在規則切出來的子問題有「太短、可能沒主詞」的情況才花一次 LLM;否則零成本。"""
    subs = split_subquestions(question)
    if len(subs) < 2:
        return subs, False
    if client is not None and any(len(s.rstrip("?？")) < _SHORT_SUB_LEN for s in subs):
        return decompose_with_llm(client, question), True
    return subs, False
