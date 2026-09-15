"""app/agent.py 的預算、消融開關與答案層引用驗證。"""
from types import SimpleNamespace

import pytest

from app import agent as ag
from app.agent_tools import ToolCallRecord
from tests.test_agent import ScriptedClient, _resp, _tool_call


def _resp_with_usage(content=None, tool_calls=None, prompt=100, completion=20):
    r = _resp(content, tool_calls)
    r.usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)
    return r


@pytest.fixture
def tool_exec(monkeypatch):
    """confident 的 search_regulations 回傳一個講第 22 條的 chunk。"""
    log = []

    def fake(db, embed, index, reranker, name, args, original_question="", drift_check=True, **kw):
        log.append({"name": name, "args": args, "drift_check": drift_check})
        confident = args.get("query") == "good"
        rec = ToolCallRecord(
            name=name, arguments=args, confident=confident, top_score=0.9 if confident else 0.1,
            chunk_ids=[22] if confident else [],
            chunks=[{"chunk_id": 22, "text": "依食安法第22條,包裝食品應以中文標示。", "primary_law": "食安法第22條"}] if confident else [],
        )
        return "{}", rec

    monkeypatch.setattr(ag, "execute_tool", fake)
    monkeypatch.setattr(ag, "_load_prompt", lambda name: "s")
    return log


# ---- 預算 -----------------------------------------------------------------

def test_usage_accumulates_tokens_and_llm_calls(tool_exec):
    client = ScriptedClient([
        _resp_with_usage(tool_calls=[_tool_call(args='{"query": "good"}')], prompt=300, completion=30),
        _resp_with_usage(content="依食安法第22條,應標示。", prompt=500, completion=50),
    ])
    r = ag.run_agent(None, None, None, None, client, "q")
    assert r.usage.llm_calls == 2
    assert r.usage.prompt_tokens == 800 and r.usage.completion_tokens == 80
    assert r.usage.total_tokens == 880
    assert r.usage.stop_reason == "answered"
    assert r.usage.as_dict()["total_tokens"] == 880


def test_token_budget_stops_tool_use(tool_exec):
    """token 累計超過上限 → 下一輪 tool_choice=none、stop_reason=tokens。"""
    client = ScriptedClient([
        _resp_with_usage(tool_calls=[_tool_call(args='{"query": "bad"}')], prompt=900, completion=50),
        _resp_with_usage(content="放棄", prompt=100, completion=10),
    ])
    budget = ag.AgentBudget(max_tool_calls=4, max_total_tokens=500, max_seconds=None)
    r = ag.run_agent(None, None, None, None, client, "q", budget=budget)
    assert client.kwargs_log[1]["tool_choice"] == "none"
    assert r.usage.stop_reason == "tokens"
    assert r.tool_calls_used == 1


def test_seconds_budget_marks_stop_reason(tool_exec, monkeypatch):
    ticks = iter([0.0, 0.0, 200.0, 200.0, 200.0, 200.0, 200.0])
    monkeypatch.setattr(ag.time, "time", lambda: next(ticks))
    client = ScriptedClient([
        _resp_with_usage(tool_calls=[_tool_call(args='{"query": "bad"}')]),
        _resp_with_usage(content="放棄"),
    ])
    r = ag.run_agent(None, None, None, None, client, "q", budget=ag.AgentBudget(max_seconds=90.0))
    assert r.usage.stop_reason == "seconds"
    assert client.kwargs_log[-1]["tool_choice"] == "none"


def test_max_tool_calls_kwarg_still_works(tool_exec):
    client = ScriptedClient([
        _resp_with_usage(tool_calls=[_tool_call(args='{"query": "bad"}', call_id="a")]),
        _resp_with_usage(tool_calls=[_tool_call(args='{"query": "bad"}', call_id="b")]),
        _resp_with_usage(content="放棄"),
    ])
    r = ag.run_agent(None, None, None, None, client, "q", max_tool_calls=1)
    assert r.tool_calls_used == 1 and r.hit_tool_call_limit
    assert r.usage.stop_reason == "tool_calls"


# ---- 消融開關 ---------------------------------------------------------------

def test_enforce_grounding_off_keeps_llm_answer(tool_exec):
    client = ScriptedClient([
        _resp_with_usage(tool_calls=[_tool_call(args='{"query": "bad"}')]),
        _resp_with_usage(content="我猜是第99條"),
    ])
    r = ag.run_agent(None, None, None, None, client, "q", enforce_grounding=False)
    assert r.grounded is False
    assert r.answer == "我猜是第99條"          # 關掉邊界 → 幻覺放行(消融用)


def test_drift_check_flag_is_forwarded_to_tools(tool_exec):
    client = ScriptedClient([_resp_with_usage(tool_calls=[_tool_call()]), _resp_with_usage(content="a")])
    ag.run_agent(None, None, None, None, client, "q", drift_check=False, verify_answer=False)
    assert tool_exec[0]["drift_check"] is False


# ---- 答案層引用驗證 -----------------------------------------------------------

def test_supported_citation_passes_without_regeneration(tool_exec):
    client = ScriptedClient([
        _resp_with_usage(tool_calls=[_tool_call(args='{"query": "good"}')]),
        _resp_with_usage(content="依食安法第22條,應以中文標示。"),
    ])
    r = ag.run_agent(None, None, None, None, client, "q")
    assert r.citation_regenerated is False and r.unsupported_citations == []
    assert r.usage.llm_calls == 2


def test_unsupported_citation_triggers_one_regeneration(tool_exec):
    client = ScriptedClient([
        _resp_with_usage(tool_calls=[_tool_call(args='{"query": "good"}')]),
        _resp_with_usage(content="依食安法第28條,廣告不得誇張。"),      # 檢索到的是第22條,亂引第28條
        _resp_with_usage(content="依食安法第22條,應以中文標示。"),      # 重生成後修正
    ])
    r = ag.run_agent(None, None, None, None, client, "q")
    assert r.citation_regenerated is True
    assert r.unsupported_citations == []
    assert r.answer.startswith("依食安法第22條")
    # 重生成那一輪:帶了回饋、不開工具
    last = client.kwargs_log[-1]
    assert last["tool_choice"] == "none"
    assert "第28條" in last["messages"][-1]["content"]


def test_still_unsupported_after_regeneration_is_flagged_not_hidden(tool_exec):
    client = ScriptedClient([
        _resp_with_usage(tool_calls=[_tool_call(args='{"query": "good"}')]),
        _resp_with_usage(content="依食安法第28條。"),
        _resp_with_usage(content="還是依食安法第28條跟第45條。"),
    ])
    r = ag.run_agent(None, None, None, None, client, "q")
    assert r.citation_regenerated is True
    assert r.unsupported_citations == ["食安法第28條", "食安法第45條"]


def test_verify_answer_off_skips_check(tool_exec):
    client = ScriptedClient([
        _resp_with_usage(tool_calls=[_tool_call(args='{"query": "good"}')]),
        _resp_with_usage(content="依食安法第28條。"),
    ])
    r = ag.run_agent(None, None, None, None, client, "q", verify_answer=False)
    assert r.citation_regenerated is False and r.usage.llm_calls == 2


def test_no_verification_on_refusal(tool_exec):
    client = ScriptedClient([
        _resp_with_usage(tool_calls=[_tool_call(args='{"query": "bad"}')]),
        _resp_with_usage(content="依第28條……"),
    ])
    r = ag.run_agent(None, None, None, None, client, "q")
    assert r.answer == ag.NO_EVIDENCE_ANSWER and r.citation_regenerated is False
