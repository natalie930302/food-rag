"""app/corrective_retrieval.py 信心閘門的邏輯測試(不載入任何模型)。"""
import pytest

from app import corrective_retrieval as cr
from tests.conftest import FakeReranker, make_chunk


@pytest.fixture
def patch_retrieval(monkeypatch):
    """把 dense retrieval 跟 entity boost 換成可控的假函式。"""
    state = {"candidates": [], "boosted_ids": [], "boosted_chunks": {}}

    monkeypatch.setattr(cr, "retrieve_chunks", lambda *a, **k: list(state["candidates"]))
    monkeypatch.setattr(cr, "find_entity_boosted_chunk_ids", lambda q: list(state["boosted_ids"]))
    monkeypatch.setattr(
        cr, "fetch_chunks_by_ids",
        lambda db, ids: [state["boosted_chunks"][i] for i in ids],
    )
    return state


def test_rejects_when_top_score_below_threshold(patch_retrieval):
    patch_retrieval["candidates"] = [make_chunk(1, "a"), make_chunk(2, "b")]
    reranker = FakeReranker({"a": cr.CONFIDENCE_THRESHOLD - 0.01, "b": 0.1})

    r = cr.retrieve_with_confidence_gate(None, None, None, reranker, "q")

    assert r.confident is False
    assert r.chunks == []                      # 不把低相關內容塞給 LLM
    assert r.top_score == pytest.approx(cr.CONFIDENCE_THRESHOLD - 0.01)


def test_accepts_at_threshold_and_sorts_by_reranker_score(patch_retrieval):
    # dense retrieval 的順序是 1,2,3;reranker 認為 3 > 1 > 2
    patch_retrieval["candidates"] = [make_chunk(1, "a"), make_chunk(2, "b"), make_chunk(3, "c")]
    reranker = FakeReranker({"a": 0.7, "b": 0.3, "c": cr.CONFIDENCE_THRESHOLD + 0.4})

    r = cr.retrieve_with_confidence_gate(None, None, None, reranker, "q", top_k=2)

    assert r.confident is True
    assert [c.chunk_id for c in r.chunks] == [3, 1]   # 依 reranker 分數排序,且截到 top_k
    assert r.top_score == pytest.approx(cr.CONFIDENCE_THRESHOLD + 0.4)


def test_empty_candidates_is_not_confident(patch_retrieval):
    reranker = FakeReranker({})
    r = cr.retrieve_with_confidence_gate(None, None, None, reranker, "q")
    assert r.confident is False and r.chunks == [] and r.top_score is None
    assert reranker.calls == []                # 沒候選就不該呼叫 reranker


def test_entity_boost_adds_candidates_without_duplicates(patch_retrieval):
    patch_retrieval["candidates"] = [make_chunk(1, "a"), make_chunk(2, "b")]
    patch_retrieval["boosted_ids"] = [2, 9]    # 2 已在候選池,9 是新加的
    patch_retrieval["boosted_chunks"] = {9: make_chunk(9, "boosted")}
    reranker = FakeReranker({"a": 0.6, "b": 0.6, "boosted": 0.99})

    r = cr.retrieve_with_confidence_gate(None, None, None, reranker, "q")

    scored_texts = [doc for _, doc in reranker.calls[0]]
    assert scored_texts.count("b") == 1        # 不重複
    assert "boosted" in scored_texts
    assert r.chunks[0].chunk_id == 9           # boost 進來的候選一樣要經過 reranker 排序


def test_entity_boost_does_not_bypass_gate(patch_retrieval):
    """entity boost 只擴充候選池,不繞過信心閘門:boost 到不相關的 chunk 一樣被擋。"""
    patch_retrieval["candidates"] = []
    patch_retrieval["boosted_ids"] = [9]
    patch_retrieval["boosted_chunks"] = {9: make_chunk(9, "irrelevant")}
    reranker = FakeReranker({"irrelevant": 0.01})

    r = cr.retrieve_with_confidence_gate(None, None, None, reranker, "q")
    assert r.confident is False and r.chunks == []


def test_reranker_sees_question_paired_with_each_candidate(patch_retrieval):
    patch_retrieval["candidates"] = [make_chunk(1, "a"), make_chunk(2, "b")]
    reranker = FakeReranker({"a": 0.9, "b": 0.9})
    cr.retrieve_with_confidence_gate(None, None, None, reranker, "我的問題")
    assert reranker.calls[0] == [["我的問題", "a"], ["我的問題", "b"]]
