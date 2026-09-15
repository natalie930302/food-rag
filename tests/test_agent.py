"""app/agent.py tool-calling harness 的邊界測試。

用腳本化的假 OpenAI client:每次 chat.completions.create() 依序吐出預先排好的回應,
同時記錄被傳進去的參數(tool_choice、temperature),驗證 harness 真的在管邊界——
不是靠 LLM 自律。
"""
import json
from types import SimpleNamespace

import pytest

from app import agent as ag
from app.agent_tools import ToolCallRecord


def _tool_call(name="search_regulations", args='{"query": "q"}', call_id="c1"):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=args))


def _resp(content=None, tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class ScriptedClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.kwargs_log = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs_log.append(kwargs)
        if not self._responses:
            raise AssertionError("LLM 被呼叫的次數超過腳本預期")
        return self._responses.pop(0)


@pytest.fixture
def tool_exec(monkeypatch):
    """execute_tool 假實作:query == "good" 才算 confident。"""
    log = []

    def fake(db, embed, index, reranker, name, args, original_question="", **kw):
        log.append((name, args))
        confident = args.get("query") == "good"
        rec = ToolCallRecord(name=name, arguments=args, confident=confident,
                             top_score=0.9 if confident else 0.1, chunk_ids=[1] if confident else [],
                             chunks=[{"chunk_id": 1, "text": "第28條 廣告不得誇張", "primary_law": "食安法第28條"}] if confident else [])
        return '{"confident": %s}' % str(confident).lower(), rec

    monkeypatch.setattr(ag, "execute_tool", fake)
    monkeypatch.setattr(ag, "_load_prompt", lambda name: "system prompt")
    return log


def test_ungrounded_answer_is_overridden(tool_exec):
    """LLM 沒查到任何 confident 結果卻硬答 → 程式碼強制覆寫成誠實拒答。"""
    client = ScriptedClient([
        _resp(tool_calls=[_tool_call(args='{"query": "bad"}')]),
        _resp(content="我覺得應該是依食安法第99條..."),
    ])
    r = ag.run_agent(None, None, None, None, client, "問題")
    assert r.grounded is False
    assert r.answer == ag.NO_EVIDENCE_ANSWER
    assert r.tool_calls_used == 1


def test_grounded_answer_is_kept(tool_exec):
    client = ScriptedClient([
        _resp(tool_calls=[_tool_call(args='{"query": "good"}')]),
        _resp(content="依食安法第28條..."),
    ])
    r = ag.run_agent(None, None, None, None, client, "問題")
    assert r.grounded is True
    assert r.answer == "依食安法第28條..."
    assert [t.confident for t in r.trace] == [True]


def test_no_tool_call_at_all_is_not_grounded(tool_exec):
    client = ScriptedClient([_resp(content="不用查我就知道")])
    r = ag.run_agent(None, None, None, None, client, "問題")
    assert r.grounded is False and r.answer == ag.NO_EVIDENCE_ANSWER
    assert tool_exec == []


def test_max_tool_calls_is_enforced_by_code(tool_exec):
    """LLM 一直想查 → 到上限後工具不再執行、tool_choice 變成 none、最後強制作答。"""
    limit = 2
    client = ScriptedClient([
        _resp(tool_calls=[_tool_call(args='{"query": "bad"}', call_id="c1")]),
        _resp(tool_calls=[_tool_call(args='{"query": "bad"}', call_id="c2")]),
        _resp(tool_calls=[_tool_call(args='{"query": "bad"}', call_id="c3")]),   # 超過上限,不該執行
        _resp(content="放棄"),
    ])
    r = ag.run_agent(None, None, None, None, client, "問題", max_tool_calls=limit)

    assert r.tool_calls_used == limit
    assert len(tool_exec) == limit
    assert r.hit_tool_call_limit is True
    # 上限到了之後,送給 LLM 的請求必須關掉工具
    assert [k["tool_choice"] for k in client.kwargs_log] == ["auto", "auto", "none", "none"]
    assert r.answer == ag.NO_EVIDENCE_ANSWER       # 全程沒 confident → 仍然覆寫


def test_over_limit_tool_call_gets_error_message_not_execution(tool_exec):
    client = ScriptedClient([
        _resp(tool_calls=[_tool_call(args='{"query": "good"}', call_id="c1"),
                          _tool_call(args='{"query": "good"}', call_id="c2")]),  # 同一輪兩個呼叫
        _resp(content="答"),
    ])
    r = ag.run_agent(None, None, None, None, client, "問題", max_tool_calls=1)
    assert len(tool_exec) == 1
    assert r.grounded is True
    # 第二個呼叫拿到的是「已達上限」的 tool 訊息,而不是真的執行
    tool_msgs = [m for m in client.kwargs_log[-1]["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert "上限" in json.loads(tool_msgs[1]["content"])["error"]


def test_decision_temperature_defaults_to_zero(tool_exec):
    client = ScriptedClient([_resp(content="x")])
    ag.run_agent(None, None, None, None, client, "問題")
    assert client.kwargs_log[0]["temperature"] == 0.0


def test_original_question_is_passed_for_drift_check(monkeypatch):
    seen = {}

    def fake(db, embed, index, reranker, name, args, original_question="", **kw):
        seen["oq"] = original_question
        return "{}", ToolCallRecord(name=name, arguments=args, confident=True)

    monkeypatch.setattr(ag, "execute_tool", fake)
    monkeypatch.setattr(ag, "_load_prompt", lambda name: "s")
    client = ScriptedClient([_resp(tool_calls=[_tool_call()]), _resp(content="a")])
    ag.run_agent(None, None, None, None, client, "使用者原話")
    assert seen["oq"] == "使用者原話"
