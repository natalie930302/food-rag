"""API 端點測試。

這些測試需要 data/index/ 已有索引才能跑。
若沒有,會被 skip。
"""

import pytest

# 條件 skip
from config.settings import settings

skip_no_index = pytest.mark.skipif(
    not settings.db_path.exists(),
    reason="data/index/ 未建立,跑過 make ingest 後再執行",
)


@skip_no_index
def test_stats():
    from fastapi.testclient import TestClient

    from app.main import app
    client = TestClient(app)
    r = client.get("/stats")
    assert r.status_code == 200
    data = r.json()
    assert "total_chunks" in data
    assert "by_kind" in data


@skip_no_index
def test_failed_files():
    from fastapi.testclient import TestClient

    from app.main import app
    client = TestClient(app)
    r = client.get("/failed")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
