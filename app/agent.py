"""
Tool-calling ReAct agent:比 app/agentic_retrieval.py 更進一步的 agent harness。

跟現有 agentic retrieval 的關係(見 docs/RESEARCH_LOG.md「Agentic Query Reformulation」章節)——
agentic_retrieval.py 是「確定性重試」:信心不足就一定執行同一個固定動作(LLM 改寫
問題、重查一次),LLM 只負責改寫文字,不負責決定策略。這裡改成讓 LLM 自己看到
每次工具呼叫回傳的信心分數,自主決定下一步要做什麼(換句話說重查、查關聯法條
換方向、還是查案例佐證),用的是業界標準的 OpenAI function-calling tool loop,
而不是寫死的 if/else 分支——這是「agent harness」跟「單次呼叫 LLM」的核心差異:
一個有邊界、可觀測(每一步都記錄成 AgentTrace)、會自己決定何時該再查一次、
何時該承認查不到的執行迴圈。

harness 的職責是設邊界,不是信任 LLM 自律。四道邊界,每一道都可以用參數關掉——
不是為了讓人關,是為了 eval/eval_harness_ablation.py 能各自量它擋掉了什麼:

  1. 預算(AgentBudget):工具呼叫次數、總 token、牆鐘秒數,任一超過就停止呼叫工具、
     強制作答。次數上限防重複查詢燒 API 費用;token/秒數上限是 production 的標配
  2. grounded 檢查(enforce_grounding):迴圈結束時,只要過程中沒有任何一次
     search_regulations 回傳 confident=True,不管 LLM 說了什麼,一律強制覆寫成
     跟 /ask 同一句誠實拒答訊息。prompt 裡雖然也寫了「沒依據就承認」,但 prompt
     只是請求不是保證——corrective RAG 的「誠實拒答」紀律要靠程式碼再把關一次
  3. 字面錨定 drift 檢查(drift_check):見 app/agent_tools.py
  4. 答案層引用驗證(verify_answer):答案裡引用的每個「第 N 條」必須出現在檢索到的
     chunk 裡,否則帶著回饋重生成一次;還是不行就把沒依據的引用標記在結果裡。
     這是信心閘門管不到的那一層——檢索內容可信,不代表 LLM 引用得對(app/verifier.py)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from openai import OpenAI

from app.agent_tools import TOOL_SCHEMAS, ToolCallRecord, execute_tool
from app.llm import _load_prompt
from app.verifier import feedback_for_regeneration, verify_citations
from config.settings import settings

NO_EVIDENCE_ANSWER = "目前資料庫裡沒有找到足夠可信的法規依據可以回答這個問題,建議換個問法,或直接洽詢主管機關確認。"

DEFAULT_MAX_TOOL_CALLS = 4


@dataclass
class AgentBudget:
    """agent 一次回答可以花的資源上限;None 代表該項不設限。"""
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    max_total_tokens: int | None = 12_000     # prompt + completion 累計
    max_seconds: float | None = 90.0

    def exceeded(self, usage: "AgentUsage") -> str | None:
        """回傳超過的是哪一項(給 trace 用),沒超過回 None。"""
        if usage.tool_calls >= self.max_tool_calls:
            return "tool_calls"
        if self.max_total_tokens is not None and usage.total_tokens >= self.max_total_tokens:
            return "tokens"
        if self.max_seconds is not None and usage.elapsed_s >= self.max_seconds:
            return "seconds"
        return None


@dataclass
class AgentUsage:
    tool_calls: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_s: float = 0.0
    stop_reason: str = "answered"   # answered / tool_calls / tokens / seconds

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict:
        return {
            "tool_calls": self.tool_calls, "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens, "elapsed_s": round(self.elapsed_s, 2),
            "stop_reason": self.stop_reason,
        }


@dataclass
class AgentResult:
    answer: str
    trace: list[ToolCallRecord] = field(default_factory=list)
    tool_calls_used: int = 0
    grounded: bool = False
    hit_tool_call_limit: bool = False
    usage: AgentUsage = field(default_factory=AgentUsage)
    unsupported_citations: list[str] = field(default_factory=list)
    citation_regenerated: bool = False


def _record_usage(usage: AgentUsage, resp) -> None:
    usage.llm_calls += 1
    u = getattr(resp, "usage", None)
    if u is not None:
        usage.prompt_tokens += getattr(u, "prompt_tokens", 0) or 0
        usage.completion_tokens += getattr(u, "completion_tokens", 0) or 0


def _grounded_chunks(trace: list[ToolCallRecord]) -> list[dict]:
    """所有 confident 的 search_regulations 回傳過的 chunk(給引用驗證用)。"""
    out: list[dict] = []
    for rec in trace:
        if rec.name == "search_regulations" and rec.confident:
            out.extend(rec.chunks)
    return out


def run_agent(
    db, embed_model, faiss_index, reranker, client: OpenAI,
    question: str,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    temperature: float = 0.0,
    budget: AgentBudget | None = None,
    enforce_grounding: bool = True,
    drift_check: bool = True,
    verify_answer: bool = True,
    faiss_cases=None,
) -> AgentResult:
    # temperature 預設 0,不是 settings.openai_temperature(0.1)——diagnostic 發現
    # 8 題 hard 問題裡有題目重跑 2 次得到不同的查詢字串,導致命中/沒命中不一致
    # (見 docs/RESEARCH_LOG.md「temperature=0」章節)。
    budget = budget or AgentBudget(max_tool_calls=max_tool_calls)
    system = _load_prompt("system_agent") or (
        "你是食品法規助理,只能透過工具檢索資料庫來回答問題。"
    )
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    trace: list[ToolCallRecord] = []
    usage = AgentUsage()
    grounded = False
    hit_limit = False
    t0 = time.time()

    while True:
        usage.elapsed_s = time.time() - t0
        over = budget.exceeded(usage)
        resp = client.chat.completions.create(
            model=settings.openai_model,
            temperature=temperature,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="none" if over else "auto",
        )
        _record_usage(usage, resp)
        msg = resp.choices[0].message

        if not msg.tool_calls:
            answer = msg.content or NO_EVIDENCE_ANSWER
            usage.elapsed_s = time.time() - t0
            if over:
                usage.stop_reason = over
            if not grounded and enforce_grounding:
                answer = NO_EVIDENCE_ANSWER
            result = AgentResult(
                answer=answer, trace=trace, tool_calls_used=usage.tool_calls,
                grounded=grounded, hit_tool_call_limit=hit_limit, usage=usage,
            )
            if verify_answer and grounded and answer != NO_EVIDENCE_ANSWER:
                _verify_and_maybe_regenerate(client, messages, result, temperature)
            return result

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
            # 這裡只管「次數」:token / 秒數預算是拿來決定要不要再讓 LLM 開工具(上面的
            # tool_choice),LLM 已經要求的工具呼叫本身是本地運算、不花 API 費用,照執行
            if usage.tool_calls >= budget.max_tool_calls:
                hit_limit = True
                usage.stop_reason = "tool_calls"
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": json.dumps({"error": "已達工具呼叫上限,請直接根據目前已知資訊回答或誠實拒答。"}),
                })
                continue

            args = json.loads(tc.function.arguments or "{}")
            result_json, record = execute_tool(
                db, embed_model, faiss_index, reranker, tc.function.name, args,
                original_question=question, drift_check=drift_check, faiss_cases=faiss_cases,
            )
            trace.append(record)
            usage.tool_calls += 1
            if record.confident:
                grounded = True
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_json})

        if usage.tool_calls >= budget.max_tool_calls and not hit_limit:
            hit_limit = True
            usage.stop_reason = "tool_calls"


def _verify_and_maybe_regenerate(client, messages: list[dict], result: AgentResult, temperature: float) -> None:
    """答案層引用驗證:沒依據的條號 → 帶回饋重生成一次;仍然沒依據就標記,不再重試。"""
    chunks = _grounded_chunks(result.trace)
    check = verify_citations(result.answer, chunks)
    if check.supported:
        return
    messages.append({"role": "assistant", "content": result.answer})
    messages.append({"role": "user", "content": feedback_for_regeneration(check)})
    resp = client.chat.completions.create(
        model=settings.openai_model, temperature=temperature, messages=messages,
        tools=TOOL_SCHEMAS, tool_choice="none",
    )
    _record_usage(result.usage, resp)
    result.citation_regenerated = True
    new_answer = resp.choices[0].message.content or result.answer
    recheck = verify_citations(new_answer, chunks)
    result.answer = new_answer
    result.unsupported_citations = [c.label for c in recheck.unsupported]
