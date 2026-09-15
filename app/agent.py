"""
Tool-calling ReAct agent:比 app/agentic_retrieval.py 更進一步的 agent harness。

跟現有 agentic retrieval 的關係(見 README「Agentic Query Reformulation」章節)——
agentic_retrieval.py 是「確定性重試」:信心不足就一定執行同一個固定動作(LLM 改寫
問題、重查一次),LLM 只負責改寫文字,不負責決定策略。這裡改成讓 LLM 自己看到
每次工具呼叫回傳的信心分數,自主決定下一步要做什麼(換句話說重查、查關聯法條
換方向、還是查案例佐證),用的是業界標準的 OpenAI function-calling tool loop,
而不是寫死的 if/else 分支——這是「agent harness」跟「單次呼叫 LLM」的核心差異:
一個有邊界(max_tool_calls)、可觀測(每一步都記錄成 AgentTrace)、會自己決定
何時該再查一次、何時該承認查不到的執行迴圈。

安全邊界(harness 的核心職責,不是信任 LLM 自律):
  - max_tool_calls 硬性上限,避免 LLM 陷入重複呼叫同一個工具的迴圈,失控燒 API 費用
  - grounded 檢查:迴圈結束時,不管 LLM 說了什麼,只要過程中沒有任何一次
    search_regulations 回傳 confident=True,就強制覆寫成既有的誠實拒答訊息
    (跟 app/main.py 的 /ask 用同一句話)。這是刻意的:不信任 LLM 會乖乖遵守
    system prompt 裡「沒有依據就拒答」的指示,用程式碼再把關一次——prompt
    只是請求,不是保證,corrective RAG 的「誠實拒答」紀律要靠程式碼強制,
    才能在多步驟、LLM 自主決策的 agent 迴圈裡維持跟原本單次檢索一樣的
    anti-hallucination 保證。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from openai import OpenAI

from config.settings import settings
from app.agent_tools import TOOL_SCHEMAS, ToolCallRecord, execute_tool
from app.llm import _load_prompt

NO_EVIDENCE_ANSWER = "目前資料庫裡沒有找到足夠可信的法規依據可以回答這個問題,建議換個問法,或直接洽詢主管機關確認。"

DEFAULT_MAX_TOOL_CALLS = 4


@dataclass
class AgentResult:
    answer: str
    trace: list[ToolCallRecord] = field(default_factory=list)
    tool_calls_used: int = 0
    grounded: bool = False
    hit_tool_call_limit: bool = False


def run_agent(
    db, embed_model, faiss_index, reranker, client: OpenAI,
    question: str, max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    temperature: float = 0.0,
) -> AgentResult:
    system = _load_prompt("system_agent") or (
        "你是食品法規助理,只能透過工具檢索資料庫來回答問題。"
    )
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    trace: list[ToolCallRecord] = []
    grounded = False
    tool_calls_used = 0
    hit_limit = False

    while True:
        # temperature 預設0,不是settings.openai_temperature(0.1)——diagnostic發現
        # 8題hard問題裡有題目重跑2次得到不同的search_regulations查詢字串,導致
        # 命中/沒命中不一致(見README「字面錨定一致性檢查」章節)。這裡先假設根因
        # 是取樣隨機性,用temperature=0驗證是否能消除這個不穩定,而不是直接跳去
        # 用「跑5次取多數決」這種更貴的手段掩蓋問題,而沒有先排除更便宜的根因。
        resp = client.chat.completions.create(
            model=settings.openai_model,
            temperature=temperature,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto" if tool_calls_used < max_tool_calls else "none",
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            answer = msg.content or NO_EVIDENCE_ANSWER
            if not grounded:
                answer = NO_EVIDENCE_ANSWER
            return AgentResult(
                answer=answer, trace=trace, tool_calls_used=tool_calls_used,
                grounded=grounded, hit_tool_call_limit=hit_limit,
            )

        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            if tool_calls_used >= max_tool_calls:
                hit_limit = True
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": json.dumps({"error": "已達工具呼叫上限,請直接根據目前已知資訊回答或誠實拒答。"}),
                })
                continue

            args = json.loads(tc.function.arguments or "{}")
            result_json, record = execute_tool(
                db, embed_model, faiss_index, reranker, tc.function.name, args,
                original_question=question,
            )
            trace.append(record)
            tool_calls_used += 1
            if record.confident:
                grounded = True
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_json})

        if tool_calls_used >= max_tool_calls and not hit_limit:
            hit_limit = True
