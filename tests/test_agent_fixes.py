"""2026/09 multi-hop 評估後補的兩道 harness 修正:
  5. search_regulations 工具內建「LLM 改寫重查一次」(tool_retry)
  6. LLM 要作答卻從沒查過法規 → 程式碼自己補查一次(force_regulation)
"""
from types import SimpleNamespace

import pytest

from app import agent as ag
from app import agent_tools as at
from app.agent_tools import ToolCallRecord
from app.corrective_retrieval import CorrectiveRetrievalResult
from tests.conftest import make_chunk
from tests.test_agent import ScriptedClient, _resp, _tool_call


def _result(ids, confident=True, score=0.9):
    return CorrectiveRetrievalResult(chunks=[make_chunk(i) for i in ids], confident=confident, top_score=score if ids else None)


# ---- 5. 工具內建重試 -------------------------------------------------------------------

@pytest.fixture
def gate_with_retry(monkeypatch):
    """第一次查沒信心;改寫後的查詢有信心。"""
    table = {"原始問題?": _result([], confident=False, score=0.2), "正式改寫版": _result([42])}
    queries = []

    def fake_gate(db, embed, index, reranker, question, filters=None, top_k=5, candidate_n=10):
        queries.append(question)
        return table.get(question, _result([], confident=False, score=0.1))

    monkeypatch.setattr(at, "retrieve_with_confidence_gate", fake_gate)
    monkeypatch.setattr(at, "reformulate_query", lambda client, q: "正式改寫版")
    return queries


def test_tool_retries_with_reformulated_query_when_not_confident(gate_with_retry):
    payload, rec = at.execute_tool(None, None, None, None, "search_regulations", {"query": "原始問題?"},
                                   original_question="原始問題?", client=object())
    assert rec.retry_used is True and rec.confident is True and rec.chunk_ids == [42]
    assert gate_with_retry == ["原始問題?", "正式改寫版"]
    assert "重查" in payload


def test_tool_retry_is_skipped_without_client_or_when_disabled(gate_with_retry):
    _, rec = at.execute_tool(None, None, None, None, "search_regulations", {"query": "原始問題?"},
                             original_question="原始問題?", client=None)
    assert rec.retry_used is False and rec.confident is False
    _, rec2 = at.execute_tool(None, None, None, None, "search_regulations", {"query": "原始問題?"},
                              original_question="原始問題?", client=object(), tool_retry=False)
    assert rec2.retry_used is False


def test_tool_retry_not_triggered_when_first_query_confident(monkeypatch):
    monkeypatch.setattr(at, "retrieve_with_confidence_gate", lambda *a, **k: _result([1]))
    monkeypatch.setattr(at, "reformulate_query", lambda c, q: (_ for _ in ()).throw(AssertionError("不該改寫")))
    _, rec = at.execute_tool(None, None, None, None, "search_regulations", {"query": "q"}, original_question="q", client=object())
    assert rec.retry_used is False and rec.confident is True


# ---- 6. 程式碼強制先查法規 ---------------------------------------------------------------

@pytest.fixture
def tool_exec(monkeypatch):
    log = []

    def fake(db, embed, index, reranker, name, args, original_question="", **kw):
        log.append((name, args))
        if name == "search_regulations":
            rec = ToolCallRecord(name=name, arguments=args, confident=True, top_score=0.9, chunk_ids=[22],
                                 chunks=[{"chunk_id": 22, "text": "依食安法第22條,包裝食品應以中文標示。", "primary_law": "食安法第22條"}])
        else:
            rec = ToolCallRecord(name=name, arguments=args, chunk_ids=[7])
        return '{"confident": true}', rec

    monkeypatch.setattr(ag, "execute_tool", fake)
    monkeypatch.setattr(ag, "_load_prompt", lambda name: "s")
    return log


def test_forced_regulation_lookup_when_llm_answers_without_it(tool_exec):
    """LLM 只查案例就想作答 → harness 用原始問題補查法規,再讓它答;結果 grounded。"""
    client = ScriptedClient([
        _resp(tool_calls=[_tool_call(name="search_violation_cases", args='{"query": "案例"}')]),
        _resp(content="依案例看,大概會被罰。"),                  # 沒查法規就要作答 → 被攔下補查
        _resp(content="依食安法第22條,應以中文標示;案例顯示會被罰。"),
    ])
    r = ag.run_agent(None, None, None, None, client, "使用者原話")
    assert [n for n, _ in tool_exec] == ["search_violation_cases", "search_regulations"]
    assert tool_exec[1][1] == {"query": "使用者原話", "forced": True}
    assert r.grounded is True and r.answer.startswith("依食安法第22條")
    assert r.tool_calls_used == 2
    # 補查結果是以 user 訊息塞回對話,第三次呼叫 LLM 時看得到
    assert "系統提醒" in client.kwargs_log[2]["messages"][-1]["content"]


def test_forced_lookup_happens_at_most_once(tool_exec, monkeypatch):
    def never_confident(db, embed, index, reranker, name, args, original_question="", **kw):
        tool_exec.append((name, args))
        return '{"confident": false}', ToolCallRecord(name=name, arguments=args, confident=False, top_score=0.1)

    monkeypatch.setattr(ag, "execute_tool", never_confident)
    client = ScriptedClient([_resp(content="答一"), _resp(content="答二")])
    r = ag.run_agent(None, None, None, None, client, "q")
    assert [n for n, _ in tool_exec] == ["search_regulations"]     # 只補查一次,不無限迴圈
    assert r.answer == ag.NO_EVIDENCE_ANSWER and r.grounded is False


def test_forced_lookup_can_be_disabled_for_ablation(tool_exec):
    client = ScriptedClient([_resp(content="直接答")])
    r = ag.run_agent(None, None, None, None, client, "q", force_regulation=False)
    assert tool_exec == [] and r.answer == ag.NO_EVIDENCE_ANSWER


def test_tool_retry_counts_as_llm_call_in_usage(monkeypatch):
    def fake(db, embed, index, reranker, name, args, original_question="", **kw):
        return '{"confident": true}', ToolCallRecord(name=name, arguments=args, confident=True, top_score=0.9,
                                                    chunk_ids=[1], retry_used=True,
                                                    chunks=[{"chunk_id": 1, "text": "第22條", "primary_law": "食安法第22條"}])

    monkeypatch.setattr(ag, "execute_tool", fake)
    monkeypatch.setattr(ag, "_load_prompt", lambda name: "s")
    client = ScriptedClient([_resp(tool_calls=[_tool_call(args='{"query": "q"}')]), _resp(content="依第22條。")])
    r = ag.run_agent(None, None, None, None, client, "q")
    assert r.usage.llm_calls == 3          # 2 次迴圈 + 1 次工具內改寫
    assert r.trace[0].retry_used is True


def test_client_and_flags_are_forwarded_to_tools(monkeypatch):
    seen = {}

    def fake(db, embed, index, reranker, name, args, original_question="", **kw):
        seen.update(kw)
        return "{}", ToolCallRecord(name=name, arguments=args, confident=True)

    monkeypatch.setattr(ag, "execute_tool", fake)
    monkeypatch.setattr(ag, "_load_prompt", lambda name: "s")
    client = ScriptedClient([_resp(tool_calls=[_tool_call()]), _resp(content="a")])
    ag.run_agent(None, None, None, None, client, "q", tool_retry=False, verify_answer=False)
    assert seen["client"] is client and seen["tool_retry"] is False


def test_forced_lookup_respects_budget(tool_exec):
    """預算已超過時不再補查(over=True → 直接作答/拒答)。"""
    client = ScriptedClient([_resp(content="直接答")])
    budget = ag.AgentBudget(max_tool_calls=0)
    r = ag.run_agent(None, None, None, None, client, "q", budget=budget)
    assert tool_exec == [] and r.usage.stop_reason == "tool_calls"


def test_record_dataclass_has_retry_flag():
    assert ToolCallRecord(name="x", arguments={}).retry_used is False
    assert isinstance(SimpleNamespace(), object)
