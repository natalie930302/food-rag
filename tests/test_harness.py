"""app/harness.py:所有執行路徑共用的 trace / usage / 拒答契約 / 引用驗證。"""
from types import SimpleNamespace

from app import harness as hz
from tests.conftest import make_chunk


def test_trace_steps_record_name_and_fields():
    ctx = hz.RunContext()
    ctx.step("retrieve", confident=True, top_score=0.93, chunk_ids=[1, 2], detail="x")
    ctx.step("generate")
    assert [s.name for s in ctx.trace] == ["retrieve", "generate"]
    d = ctx.trace[0].as_dict()
    assert d["confident"] is True and d["chunk_ids"] == [1, 2] and d["top_score"] == 0.93


def test_usage_accumulates_from_responses_and_manual_counters():
    ctx = hz.RunContext()
    ctx.record_llm(SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10)))
    ctx.record_llm(SimpleNamespace())                     # 假 client 沒有 usage → 只計次數
    ctx.tool_calls += 2
    u = ctx.usage()
    assert u["llm_calls"] == 2 and u["tool_calls"] == 2
    assert u["prompt_tokens"] == 100 and u["total_tokens"] == 110
    assert u["stop_reason"] == "answered"


def test_time_budget(monkeypatch):
    ticks = iter([10.0, 100.0])
    monkeypatch.setattr(hz.time, "time", lambda: next(ticks))
    ctx = hz.RunContext(max_seconds=50.0, t0=0.0)
    assert ctx.over_budget() is False      # 10 s
    assert ctx.over_budget() is True       # 100 s
    assert hz.RunContext(max_seconds=None).over_budget() is False


def test_is_refusal_recognises_contract_and_llm_wording():
    assert hz.is_refusal(hz.NO_EVIDENCE_ANSWER)
    assert hz.is_refusal("很抱歉,目前資料庫裡沒有找到足夠可信的法規依據。")
    assert hz.is_refusal("查無相關規定,建議洽詢主管機關。")
    assert not hz.is_refusal("依食安法第22條,應以中文標示。")


def test_verify_supported_answer_passes_without_regeneration():
    ctx = hz.RunContext()
    chunks = [make_chunk(1, "依食安法第22條,包裝食品應以中文標示。")]
    answer, unsupported, regenerated = hz.verify_and_regenerate(
        ctx, "依食安法第22條,應以中文標示。", chunks, regenerate=lambda fb: (_ for _ in ()).throw(AssertionError("不該重生成")))
    assert unsupported == [] and regenerated is False
    assert ctx.trace[-1].name == "verify_citations"


def test_verify_regenerates_once_and_flags_leftovers():
    ctx = hz.RunContext()
    c = make_chunk(1, "依食安法第22條,包裝食品應以中文標示。")
    c.primary_law = "食安法第22條"
    calls = []

    def regenerate(feedback):
        calls.append(feedback)
        return "修正後:依食安法第22條;另外第45條也相關。"     # 第45條仍然沒依據

    answer, unsupported, regenerated = hz.verify_and_regenerate(ctx, "依食安法第28條,廣告不得誇張。", [c], regenerate)
    assert regenerated is True and len(calls) == 1 and "第28條" in calls[0]
    assert answer.startswith("修正後")
    assert unsupported == ["食安法第45條"]              # 不默默放行,如實回報


def test_verify_skips_refusals():
    ctx = hz.RunContext()
    answer, unsupported, regenerated = hz.verify_and_regenerate(ctx, hz.NO_EVIDENCE_ANSWER, [], regenerate=None)
    assert answer == hz.NO_EVIDENCE_ANSWER and unsupported == [] and regenerated is False
    assert ctx.trace == []
