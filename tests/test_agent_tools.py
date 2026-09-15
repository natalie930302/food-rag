"""app/agent_tools.py:工具執行與「字面錨定一致性檢查」的介入條件。"""
import json

import pytest

from app import agent_tools as at
from app.corrective_retrieval import CorrectiveRetrievalResult
from tests.conftest import make_chunk


def _result(ids, confident=True, score=0.9):
    return CorrectiveRetrievalResult(
        chunks=[make_chunk(i) for i in ids], confident=confident, top_score=score if ids else None,
    )


@pytest.fixture
def gate(monkeypatch):
    """用「查詢字串 → 結果」對照表取代信心閘門檢索,並記錄被查了哪些字串。"""
    table: dict[str, CorrectiveRetrievalResult] = {}
    queries: list[str] = []

    def fake(db, embed, index, reranker, question, filters=None, top_k=5, candidate_n=10):
        queries.append(question)
        return table[question]

    monkeypatch.setattr(at, "retrieve_with_confidence_gate", fake)
    return table, queries


def test_drift_detected_switches_to_literal_result(gate):
    table, queries = gate
    table["關鍵字 濃縮"] = _result([1, 2], score=0.978)       # LLM 改寫:高信心但答錯
    table["原始問題?"] = _result([7, 2], score=0.974)         # 字面問題:top-1 不同

    payload, rec = at.execute_tool(
        None, None, None, None, "search_regulations", {"query": "關鍵字 濃縮"},
        original_question="原始問題?",
    )

    assert rec.query_drift_detected is True
    assert rec.chunk_ids == [7, 2]                            # 改用字面問題的結果
    assert "不一致" in json.loads(payload)["note"]
    assert queries == ["關鍵字 濃縮", "原始問題?"]


def test_no_intervention_when_top1_agrees(gate):
    table, _ = gate
    table["改寫"] = _result([1, 2])
    table["原始"] = _result([1, 9])
    _, rec = at.execute_tool(None, None, None, None, "search_regulations", {"query": "改寫"}, original_question="原始")
    assert rec.query_drift_detected is False and rec.chunk_ids == [1, 2]


def test_no_intervention_when_literal_not_confident(gate):
    """字面問題本身沒信心 → 保留 LLM 改寫版(這是 agentic retry 原本有效的情境)。"""
    table, _ = gate
    table["改寫"] = _result([1])
    table["原始"] = _result([], confident=False, score=0.1)
    _, rec = at.execute_tool(None, None, None, None, "search_regulations", {"query": "改寫"}, original_question="原始")
    assert rec.query_drift_detected is False and rec.chunk_ids == [1]


def test_literal_query_skipped_when_identical(gate):
    table, queries = gate
    table["同一句"] = _result([1])
    _, rec = at.execute_tool(None, None, None, None, "search_regulations", {"query": "同一句"}, original_question="同一句")
    assert queries == ["同一句"]                              # 不多做一次無意義的檢索
    assert rec.query_drift_detected is False


def test_low_confidence_payload_warns_llm(gate):
    table, _ = gate
    table["q"] = _result([], confident=False, score=0.2)
    payload, rec = at.execute_tool(None, None, None, None, "search_regulations", {"query": "q"})
    data = json.loads(payload)
    assert data["confident"] is False and data["chunks"] == []
    assert "不可當作回答依據" in data["note"]
    assert rec.confident is False


def test_law_article_filter_passthrough(monkeypatch):
    seen = {}

    def fake(db, embed, index, reranker, question, filters=None, top_k=5, candidate_n=10):
        seen["filters"] = filters
        return _result([1])

    monkeypatch.setattr(at, "retrieve_with_confidence_gate", fake)
    at.execute_tool(None, None, None, None, "search_regulations", {"query": "q", "law_article": "食安法第28條"})
    assert seen["filters"] == {"law_article": "食安法第28條"}


def test_unknown_tool_raises():
    with pytest.raises(ValueError):
        at.execute_tool(None, None, None, None, "not_a_tool", {})


def test_tool_schemas_are_valid_function_definitions():
    names = {t["function"]["name"] for t in at.TOOL_SCHEMAS}
    assert names == {"search_regulations", "search_violation_cases", "search_related_laws"}
    for t in at.TOOL_SCHEMAS:
        params = t["function"]["parameters"]
        assert set(params["required"]) <= set(params["properties"])


def test_violation_cases_use_the_cases_index_not_the_chunk_index(monkeypatch):
    """回歸測試:search_violation_cases 曾經拿法規 chunk 的 FAISS 索引去查案例(命中 0/17)。"""
    seen = {}

    def fake_cases(db, model, faiss_index, text, top_k=3):
        seen["index"] = faiss_index
        return []

    monkeypatch.setattr(at, "retrieve_cases", fake_cases)
    chunk_index, cases_index = object(), object()
    at.execute_tool(None, None, chunk_index, None, "search_violation_cases", {"query": "q"}, faiss_cases=cases_index)
    assert seen["index"] is cases_index
