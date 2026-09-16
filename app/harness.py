"""
/query 統一的 harness 層:不管走哪條執行路徑(固定管線 / 廣告審稿 / tool-calling agent),
回來的 trace、usage、拒答契約、引用驗證都是同一套。

為什麼要有這一層:原本三個端點各自實作——trace 跟 usage 只有 agent 有、引用驗證只有
/ask 跟 agent 有、拒答句寫在三個地方。對外只有一個入口卻有三種規矩,harness 要各管三份。
這裡把「規矩」收成一份,執行路徑照舊各走各的(單跳走便宜的固定管線,多步才進 agent,
見 app/router.py 的依據)。

  RunContext          一次請求的共用狀態:trace 步驟、LLM usage 累計、牆鐘時間、預算
  TraceStep           每一步做了什麼(名稱、耗時、信心分數、chunk ids……),三條路都用同一種
  NO_EVIDENCE_ANSWER  唯一的拒答句;is_refusal() 判斷一段答案是不是拒答(含 LLM 自己的措辭)
  verify_and_regenerate  答案層引用驗證 + 帶回饋重生成一次,所有會生成答案的路徑共用
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.verifier import feedback_for_regeneration, verify_citations

NO_EVIDENCE_ANSWER = "目前資料庫裡沒有找到足夠可信的法規依據可以回答這個問題,建議換個問法,或直接洽詢主管機關確認。"

# LLM 在沒有 grounded 強制時常用自己的話拒答;這些都算拒答,不算「沒依據卻硬答」
REFUSAL_CUES = ("沒有找到足夠可信", "無法回答", "沒有相關", "無相關", "找不到相關", "不足以回答", "建議洽詢",
                "沒有足夠", "無法提供", "不在資料庫", "資料庫裡沒有", "查無")


def is_refusal(answer: str) -> bool:
    return answer == NO_EVIDENCE_ANSWER or any(c in answer for c in REFUSAL_CUES)


@dataclass
class TraceStep:
    name: str                       # retrieve / retry / generate / verify_citations / tool:search_regulations …
    ms: int = 0
    detail: str = ""
    confident: bool | None = None
    top_score: float | None = None
    chunk_ids: list[int] = field(default_factory=list)
    query_drift_detected: bool = False
    arguments: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "name": self.name, "ms": self.ms, "detail": self.detail, "confident": self.confident,
            "top_score": self.top_score, "chunk_ids": self.chunk_ids,
            "query_drift_detected": self.query_drift_detected, "arguments": self.arguments,
        }


@dataclass
class RunContext:
    max_seconds: float | None = 90.0
    t0: float = field(default_factory=time.time)
    trace: list[TraceStep] = field(default_factory=list)
    llm_calls: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    stop_reason: str = "answered"

    # ---- 時間 ----
    @property
    def elapsed_s(self) -> float:
        return time.time() - self.t0

    def over_budget(self) -> bool:
        return self.max_seconds is not None and self.elapsed_s >= self.max_seconds

    # ---- trace ----
    def step(self, name: str, t_start: float | None = None, **kw) -> TraceStep:
        ms = int((time.time() - t_start) * 1000) if t_start is not None else 0
        s = TraceStep(name=name, ms=ms, **kw)
        self.trace.append(s)
        return s

    # ---- usage ----
    def record_llm(self, resp) -> None:
        """從 OpenAI 回應累計 token;沒有 usage 欄位(假 client)就只計次數。"""
        self.llm_calls += 1
        u = getattr(resp, "usage", None)
        if u is not None:
            self.prompt_tokens += getattr(u, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(u, "completion_tokens", 0) or 0

    def usage(self) -> dict:
        return {
            "llm_calls": self.llm_calls, "tool_calls": self.tool_calls,
            "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "elapsed_s": round(self.elapsed_s, 2), "stop_reason": self.stop_reason,
        }


def verify_and_regenerate(ctx: RunContext, answer: str, chunks, regenerate) -> tuple[str, list[str], bool]:
    """答案層引用驗證(app/verifier.py)。

    answer 引用的每個「第 N 條」必須出現在 chunks 裡;不然呼叫 regenerate(feedback) 重生成一次,
    仍然不符就把沒依據的條號如實回報,不默默放行。回傳 (answer, unsupported_labels, regenerated)。
    拒答句不驗(沒有內容可驗)。
    """
    if is_refusal(answer):
        return answer, [], False
    t = time.time()
    check = verify_citations(answer, chunks)
    regenerated = False
    if not check.supported:
        answer = regenerate(feedback_for_regeneration(check)) or answer
        regenerated = True
        check = verify_citations(answer, chunks)
    unsupported = [c.label for c in check.unsupported]
    ctx.step("verify_citations", t, detail=f"citations={len(check.citations)} unsupported={len(unsupported)} regenerated={regenerated}")
    return answer, unsupported, regenerated
