"""複合問題(一句多問)的處理:拆解、路由、工具端查詢修正、案例信心、部分回答不算拒答。

2026/09 使用者實測「減肥廣告違反哪一條?罰多少?有沒有實際案例?」整題被拒答後補的一組邊界。
"""
import json
from types import SimpleNamespace

import pytest

from app import agent_tools as at
from app import decompose as dc
from app import handlers as h
from app import harness as hz
from app import router as rt
from app.corrective_retrieval import CorrectiveRetrievalResult
from app.harness import RunContext
from app.retrieval import RetrievedCase
from tests.conftest import make_chunk

# ---------- decompose ----------

def test_split_subquestions_by_question_marks():
    assert dc.split_subquestions("減肥廣告違反哪一條?罰多少?有沒有實際案例?") == ["減肥廣告違反哪一條?", "罰多少?", "有沒有實際案例?"]
    assert dc.split_subquestions("真空包裝豆干要符合什麼規定?") == ["真空包裝豆干要符合什麼規定?"]
    assert dc.split_subquestions("全形問號？也要切？") == ["全形問號?", "也要切?"]
    assert dc.is_compound("A規定?B規定?") and not dc.is_compound("只有一個問題?")


def test_keyword_detection_and_conversion():
    assert dc.looks_like_keywords("減肥廣告")
    assert dc.looks_like_keywords("食品添加物 標示")
    assert not dc.looks_like_keywords("減肥廣告違反哪一條?")
    assert not dc.looks_like_keywords("食品添加物需要登錄嗎")
    assert dc.keyword_to_question("減肥廣告") == "減肥廣告的相關規定是什麼?"


def test_decompose_uses_llm_only_when_a_subquestion_is_too_short():
    calls = []

    class FakeClient:
        chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: (
            calls.append(kw) or SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=json.dumps({"subquestions": ["減肥廣告違反食安法哪一條", "減肥廣告違規會罰多少錢?"]})))]))))

    subs, used = dc.decompose(FakeClient(), "減肥廣告違反哪一條?罰多少?")
    assert used and subs == ["減肥廣告違反食安法哪一條?", "減肥廣告違規會罰多少錢?"]   # 補了問號、補了主詞
    assert len(calls) == 1

    calls.clear()
    subs, used = dc.decompose(FakeClient(), "天然色素算是食品添加物嗎?天然色素需要辦理登錄嗎?")
    assert not used and calls == [] and len(subs) == 2                               # 子問題都有主詞、夠長 → 零成本

    subs, used = dc.decompose(None, "只有一個問題?")
    assert subs == ["只有一個問題?"] and not used


def test_decompose_falls_back_to_rule_split_on_bad_llm_output():
    class BadClient:
        chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))])))
    subs, used = dc.decompose(BadClient(), "減肥廣告違反哪一條?罰多少?")
    assert used and subs == ["減肥廣告違反哪一條?", "罰多少?"]


# ---------- router ----------

@pytest.mark.parametrize("question, intent", [
    ("天然色素算是食品添加物嗎?需要辦理登錄嗎?", "regulation_qa"),          # 子問題都只問規定 → 規則可判
    ("減肥廣告違反哪一條?罰多少?有沒有實際案例?", "multi_hop"),            # 明確要案例
    ("宣稱降血糖會依哪條處罰?跟第28條有關的其他條文有哪些?", "multi_hop"),  # 關聯條文
])
def test_rules_on_compound_questions(question, intent):
    d = rt.route_by_rules(question)
    assert d is not None and d.intent == intent


@pytest.mark.parametrize("question", [
    "食品添加物要怎麼標示?沒標會被罰多少錢?",   # 「被罰多少錢」是罰則問題,規則不該當成找案例;複合 → LLM
    "維生素D在膠囊食品的添加上限是多少?超過會怎樣?",
])
def test_rules_defer_mixed_compound_questions_to_llm(question):
    assert rt.route_by_rules(question) is None


def test_penalty_amount_alone_is_not_a_case_signal():
    d = rt.route_by_rules("沒依規定標示會被罰多少錢?")
    assert d is None or d.intent != "multi_hop"


# ---------- 工具端:關鍵字 → 問句、案例信心 ----------

def test_search_regulations_converts_keywords_to_a_question(monkeypatch):
    queries = []

    def fake(db, embed, index, reranker, question, filters=None, top_k=5, candidate_n=10):
        queries.append(question)
        return CorrectiveRetrievalResult(chunks=[make_chunk(1)], confident=True, top_score=0.9)

    monkeypatch.setattr(at, "retrieve_with_confidence_gate", fake)
    payload, rec = at.execute_tool(None, None, None, None, "search_regulations", {"query": "減肥廣告"},
                                   original_question="減肥廣告違反哪一條?罰多少?", drift_check=False)
    assert queries == ["減肥廣告的相關規定是什麼?"]
    data = json.loads(payload)
    assert data["query_used"] == "減肥廣告的相關規定是什麼?" and "關鍵字" in data["note_query"]
    assert rec.arguments["query_used"] == "減肥廣告的相關規定是什麼?" and "[關鍵字→問句]" in rec.result_summary


def test_search_regulations_retry_uses_subquestion_when_original_is_compound(monkeypatch):
    """原始問題是複合句時,改寫重試的底稿是工具這次的子問句,不是整句(整句本身就是壞查詢)。"""
    seen = {}

    def fake(db, embed, index, reranker, question, filters=None, top_k=5, candidate_n=10):
        return CorrectiveRetrievalResult(chunks=[], confident=False, top_score=0.2)

    def fake_reformulate(client, base):
        seen["base"] = base
        return base + "(改寫)"

    monkeypatch.setattr(at, "retrieve_with_confidence_gate", fake)
    monkeypatch.setattr(at, "reformulate_query", fake_reformulate)
    at.execute_tool(None, None, None, None, "search_regulations", {"query": "減肥廣告違反食安法哪一條?"},
                    original_question="減肥廣告違反哪一條?罰多少?有案例嗎?", client=object(), drift_check=False)
    assert seen["base"] == "減肥廣告違反食安法哪一條?"


def _case(score):
    return RetrievedCase(id=1, year=114, month=1, date=None, product="p", company="c", violation="v",
                         penalty_twd=40000, law_cited="第28條第1項", article_no=28, score=score)


@pytest.mark.parametrize("score, confident", [(0.66, True), (0.58, True), (0.50, False), (None, False)])
def test_violation_cases_confidence_follows_similarity_threshold(monkeypatch, score, confident):
    monkeypatch.setattr(at, "retrieve_cases", lambda *a, **k: [_case(score)])
    payload, rec = at.execute_tool(None, None, None, None, "search_violation_cases", {"query": "q"}, faiss_cases=object())
    data = json.loads(payload)
    assert data["confident"] is confident and rec.confident is confident
    assert ("note" in data) is (not confident)


# ---------- harness:部分回答不是拒答 ----------

def test_partial_answer_is_not_a_refusal():
    assert not hz.is_refusal("依食安法第28條第1項處罰。資料庫中沒有找到與豆干相關的案例,建議洽詢主管機關確認。" * 2)
    assert hz.is_refusal(hz.NO_EVIDENCE_ANSWER + " 建議換個問法。")
    assert hz.is_refusal("查無相關規定。")


# ---------- 固定管線:拆子問題各自檢索、合併 ----------

def _agentic(chunks, confident=True):
    r = SimpleNamespace(chunks=chunks, confident=confident, top_score=0.9 if confident else 0.2)
    return SimpleNamespace(final_result=r, used_retry=False,
                           attempts=[{"question": "x", "confident": confident, "top_score": r.top_score}])


def test_regulation_handler_decomposes_and_merges(fake_deps, monkeypatch):
    asked = []

    def fake_retrieve(db, em, fi, rr, client, question, filters=None, top_k=5):
        asked.append(question)
        if "上限" in question:
            return _agentic([make_chunk(1), make_chunk(2)])
        return _agentic([make_chunk(2), make_chunk(3)])

    monkeypatch.setattr(h, "retrieve_agentic", fake_retrieve)
    prompts = {}

    def fake_prompt(q, ch, cs):
        prompts["chunks"] = [c.chunk_id for c in ch]
        return "sys", "usr"

    monkeypatch.setattr(h.llm, "build_general_prompt", fake_prompt)
    monkeypatch.setattr(h.llm, "call_llm", lambda client, system, user, ctx=None: (prompts.setdefault("user", user), "答")[1])

    ctx = RunContext()
    res = h.run_regulation(ctx, None, None, "維生素D添加上限是多少?超過了會怎樣?", top_k=8)
    assert asked == ["維生素D添加上限是多少?", "超過了會怎樣?"]
    assert prompts["chunks"] == [1, 2, 3]                                  # 合併、去重、保持順序
    assert "逐一回答" in prompts["user"] and "1. 維生素D添加上限是多少?" in prompts["user"]
    assert [s.name for s in ctx.trace][:3] == ["decompose", "retrieve", "retrieve"]
    assert res.confident and not res.refused


def test_regulation_handler_answers_when_only_one_subquestion_is_confident(fake_deps, monkeypatch):
    def fake_retrieve(db, em, fi, rr, client, question, filters=None, top_k=5):
        return _agentic([make_chunk(1)]) if "上限" in question else _agentic([], confident=False)

    monkeypatch.setattr(h, "retrieve_agentic", fake_retrieve)
    monkeypatch.setattr(h.llm, "build_general_prompt", lambda q, ch, cs: ("sys", "usr"))
    monkeypatch.setattr(h.llm, "call_llm", lambda *a, **k: "上限 800 IU。資料庫中沒有找到與此相關的資料。")
    res = h.run_regulation(RunContext(), None, None, "維生素D添加上限是多少?外太空可以賣嗎?", top_k=8)
    assert res.confident and not res.refused and [c.chunk_id for c in res.sources] == [1]


# ---------- 引用驗證的證據集合要包含關聯法條與有信心的案例 ----------

def test_related_laws_and_confident_cases_count_as_citation_evidence(monkeypatch):
    from app import agent as ag
    monkeypatch.setattr(at, "get_co_cited_laws", lambda db, a: [{"related_law_name": "食安法", "related_article": "45", "co_count": 9}])
    monkeypatch.setattr(at, "count_chunks_for_law", lambda db, a: 228)
    _, rel = at.execute_tool(None, None, None, None, "search_related_laws", {"article_full": "28"})
    monkeypatch.setattr(at, "retrieve_cases", lambda *a, **k: [_case(0.66)])
    _, case = at.execute_tool(None, None, None, None, "search_violation_cases", {"query": "q"}, faiss_cases=object())
    evidence = ag._grounded_chunks([rel, case])
    texts = " ".join(c["text"] for c in evidence)
    assert rel.confident is False                                             # 共現統計不能單獨撐起 grounded(OOD 題會被放行)
    assert "食安法第45條" in texts and "第28條第1項" in texts                  # 但條號仍是引用驗證的證據
    monkeypatch.setattr(at, "retrieve_cases", lambda *a, **k: [_case(0.40)])
    _, weak = at.execute_tool(None, None, None, None, "search_violation_cases", {"query": "q"}, faiss_cases=object())
    assert ag._grounded_chunks([weak]) == []                                   # 沒信心的案例不算證據
