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
