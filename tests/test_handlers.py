"""app/handlers.py:三條執行路徑都把 trace / usage / 拒答 / 引用驗證交給同一套 harness。

用 fake 取代 deps(模型)、檢索與 LLM,只驗證路徑邏輯與 harness 契約。
"""
from types import SimpleNamespace

from app import handlers as h
from app.agent import AgentResult, AgentUsage
from app.agent_tools import ToolCallRecord
from app.agentic_retrieval import AgenticRetrievalResult
from app.corrective_retrieval import CorrectiveRetrievalResult
from app.harness import NO_EVIDENCE_ANSWER, RunContext
from tests.conftest import make_chunk


def _agentic(chunks, confident=True, retry=False):
    final = CorrectiveRetrievalResult(chunks=chunks, confident=confident, top_score=0.9 if confident else 0.1)
    attempts = [{"question": "q", "confident": confident, "top_score": 0.9 if confident else 0.1, "action": "initial_retrieval"}]
    if retry:
        attempts.append({"question": "q2", "confident": confident, "top_score": 0.9, "action": "reformulated_retry"})
    return AgenticRetrievalResult(final_result=final, attempts=attempts, used_retry=retry)


class FakeLLM:
    """依序吐出答案,並記錄被呼叫幾次。"""
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, client, system, user, ctx=None):
        self.calls.append(user)
        if ctx is not None:
            ctx.record_llm(SimpleNamespace(usage=SimpleNamespace(prompt_tokens=50, completion_tokens=5)))
        return self.answers.pop(0)


# ---- regulation ---------------------------------------------------------------

def test_regulation_refuses_without_llm_call_when_not_confident(fake_deps, monkeypatch):
    monkeypatch.setattr(h, "retrieve_agentic", lambda *a, **k: _agentic([], confident=False))
    llm = FakeLLM([])
    monkeypatch.setattr(h.llm, "call_llm", llm)
    ctx = RunContext()
    r = h.run_regulation(ctx, None, None, "問題?")
    assert r.refused and r.answer == NO_EVIDENCE_ANSWER and r.confident is False
    assert llm.calls == []                                   # 沒依據就不呼叫 LLM
    assert [s.name for s in ctx.trace] == ["retrieve", "refuse"]


def test_regulation_records_retry_step_and_llm_usage(fake_deps, monkeypatch):
    c = make_chunk(1, "依食安法第22條,包裝食品應以中文標示。")
    monkeypatch.setattr(h, "retrieve_agentic", lambda *a, **k: _agentic([c], retry=True))
    monkeypatch.setattr(h.llm, "build_general_prompt", lambda q, ch, cs: ("sys", "usr"))
    llm = FakeLLM(["依食安法第22條,應以中文標示。"])
    monkeypatch.setattr(h.llm, "call_llm", llm)
    ctx = RunContext()
    r = h.run_regulation(ctx, None, None, "問題?")
    assert r.used_retry is True and r.confident is True and not r.refused
    assert [s.name for s in ctx.trace] == ["retrieve", "retry", "generate", "verify_citations"]
    assert ctx.usage()["llm_calls"] == 2                    # 改寫 1 次 + 生成 1 次
    assert r.unsupported_citations == [] and r.citation_regenerated is False


def test_regulation_citation_check_regenerates(fake_deps, monkeypatch):
    c = make_chunk(1, "依食安法第22條,包裝食品應以中文標示。")
    c.primary_law = "食安法第22條"
    monkeypatch.setattr(h, "retrieve_agentic", lambda *a, **k: _agentic([c]))
    monkeypatch.setattr(h.llm, "build_general_prompt", lambda q, ch, cs: ("sys", "usr"))
    llm = FakeLLM(["依食安法第28條,廣告不得誇張。", "依食安法第22條,應以中文標示。"])
    monkeypatch.setattr(h.llm, "call_llm", llm)
    r = h.run_regulation(RunContext(), None, None, "問題?")
    assert r.citation_regenerated is True and r.unsupported_citations == []
    assert "第28條" in llm.calls[1]                          # 第二次呼叫帶了回饋


def test_regulation_case_lookup_forces_case_retrieval(fake_deps, monkeypatch):
    monkeypatch.setattr(h, "retrieve_agentic", lambda *a, **k: _agentic([], confident=False))
    seen = {}
    monkeypatch.setattr(h.retrieval, "retrieve_cases", lambda *a, **k: seen.setdefault("called", True) and [])
    ctx = RunContext()
    h.run_regulation(ctx, None, None, "純粹的問題", include_cases=True)
    assert seen.get("called") and any(s.name == "retrieve_cases" for s in ctx.trace)


# ---- review ---------------------------------------------------------------------

def test_review_strips_verdict_line_and_reports_level(fake_deps, monkeypatch):
    c = make_chunk(1, "食安法第28條:食品廣告不得涉及醫療效能。")
    monkeypatch.setattr(h.retrieval, "retrieve_chunks", lambda *a, **k: [c])
    monkeypatch.setattr(h.llm, "detect_risk_keywords", lambda t: [])
    monkeypatch.setattr(h.llm, "build_review_prompt", lambda *a: ("sys", "usr"))
    llm = FakeLLM(["一、法規說明 依《食品安全衛生管理法》第28條,涉及醫療效能。\n\n**VERDICT:** 高風險"])
    monkeypatch.setattr(h.llm, "call_llm", llm)
    ctx = RunContext()
    r = h.run_review(ctx, None, None, "本產品有效改善高血壓")
    assert r.verdict == "high"
    assert "VERDICT" not in r.answer
    assert r.unsupported_citations == []                   # 審稿也做引用驗證
    assert [s.name for s in ctx.trace] == ["keyword_scan", "retrieve", "retrieve_cases", "generate", "verify_citations", "verdict"]


# ---- agent ----------------------------------------------------------------------

def test_agent_route_converts_tool_trace_and_usage(fake_deps, monkeypatch):
    rec = ToolCallRecord(name="search_regulations", arguments={"query": "q"}, confident=True, top_score=0.9,
                         chunk_ids=[7], result_summary="1 chunks")
    usage = AgentUsage(tool_calls=1, llm_calls=2, prompt_tokens=300, completion_tokens=30, stop_reason="answered")
    result = AgentResult(answer="依第28條。", trace=[rec], tool_calls_used=1, grounded=True, usage=usage)
    monkeypatch.setattr(h, "run_agent", lambda *a, **k: result)
    monkeypatch.setattr(h, "retrieval_chunks_by_ids", lambda db, ids: [make_chunk(7)])
    ctx = RunContext()
    r = h.run_agent_route(ctx, None, None, "多步問題")
    assert [s.name for s in ctx.trace] == ["tool:search_regulations", "generate"]
    assert ctx.trace[0].chunk_ids == [7] and ctx.trace[0].confident is True
    assert ctx.usage()["tool_calls"] == 1 and ctx.usage()["total_tokens"] == 330
    assert r.grounded is True and not r.refused and [c.chunk_id for c in r.sources] == [7]


def test_agent_route_marks_refusal(fake_deps, monkeypatch):
    result = AgentResult(answer=NO_EVIDENCE_ANSWER, trace=[], tool_calls_used=0, grounded=False, usage=AgentUsage())
    monkeypatch.setattr(h, "run_agent", lambda *a, **k: result)
    r = h.run_agent_route(RunContext(), None, None, "多步問題")
    assert r.refused and r.confident is False and r.sources == []
