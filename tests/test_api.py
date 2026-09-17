"""API 端點契約測試。

對外功能端點只有 /query 與 /health。/health 與 /query 的請求驗證不需要模型;
需要索引的部分在 data/index/ 不存在時 skip。
"""
import pytest
from fastapi.testclient import TestClient

from config.settings import settings

skip_no_index = pytest.mark.skipif(
    not settings.db_path.exists(),
    reason="data/index/ 未建立(或正在重建),跑過 make ingest 後再執行",
)


def _client():
    from app.main import app
    return TestClient(app)


def test_old_endpoints_are_gone():
    c = _client()
    for path in ("/ask", "/ask_agent", "/review"):
        assert c.post(path, json={"question": "x", "ad_text": "x"}).status_code in (404, 405)
    assert c.get("/stats").status_code == 404
    assert c.get("/failed").status_code == 404


def test_query_validates_request_before_touching_models():
    c = _client()
    assert c.post("/query", json={}).status_code == 422                                   # 缺 question
    assert c.post("/query", json={"question": "x", "force_intent": "nope"}).status_code == 422
    assert c.post("/query", json={"question": "x", "max_seconds": 1}).status_code == 422  # 低於下限 5


def test_api_prefix_is_an_alias_for_the_same_routes():
    """後端直接供應前端 build 時,前端打的是 /api/...;必須跟 /... 行為一致(不是 404/405)。"""
    c = _client()
    assert c.post("/api/query", json={}).status_code == 422
    assert c.post("/api/query", json={"question": "x", "force_intent": "nope"}).status_code == 422
    assert c.get("/api/stats").status_code == 404      # 舊端點在別名下一樣不存在


def test_openapi_lists_only_two_feature_endpoints():
    paths = _client().get("/openapi.json").json()["paths"]
    feature = {p for p in paths if p in ("/query", "/health")}
    assert feature == {"/query", "/health"}
    assert "/laws/{article}/related" in paths and "/files/{file_path}" in paths   # 資料端點


@skip_no_index
def test_health_reports_checks_and_index_stats():
    r = _client().get("/health")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] in ("ok", "degraded")
    assert set(d["checks"]) == {"index", "cases_index", "embed_model", "reranker", "openai_key"}
    assert d["index"]["total_chunks"] > 0 and "by_kind" in d["index"]
    assert isinstance(d["failed_files"], list)


def test_query_stream_emits_steps_then_result(monkeypatch):
    """/query/stream:每個 ctx.step 一個 step 事件,最後一個 result 事件;順序要對。"""
    from app import main as mn
    from app import schemas as sc

    def fake_run(req, client, ctx):
        ctx.step("route", detail="regulation_qa (rules)")
        ctx.step("retrieve", detail="x")
        return sc.QueryResponse(
            answer="答", route=sc.RouteInfo(intent="regulation_qa", source="rules", reason="", handler="regulation", router_ms=1),
            trace=[sc.TraceStep(**s.as_dict()) for s in ctx.trace], usage=sc.Usage(**ctx.usage()),
            meta=sc.QueryMeta(model="m", total_chunks_searched=0, retrieval_ms=0, llm_ms=0, refused=False),
        )

    monkeypatch.setattr(mn, "_run_query", fake_run)
    monkeypatch.setattr(mn, "_client", lambda key: object())
    c = _client()
    with c.stream("POST", "/query/stream", json={"question": "q"}) as resp:
        assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/event-stream")
        body = "".join(resp.iter_text())
    events = [blk.split("\n", 1)[0].removeprefix("event: ") for blk in body.strip().split("\n\n")]
    assert events == ["step", "step", "result"]
    assert '"name": "retrieve"' in body and '"answer": "答"' in body


def test_query_stream_reports_errors_as_event(monkeypatch):
    from app import main as mn

    def boom(req, client, ctx):
        raise RuntimeError("模型沒載")

    monkeypatch.setattr(mn, "_run_query", boom)
    monkeypatch.setattr(mn, "_client", lambda key: object())
    with _client().stream("POST", "/query/stream", json={"question": "q"}) as resp:
        body = "".join(resp.iter_text())
    assert "event: error" in body and "模型沒載" in body
