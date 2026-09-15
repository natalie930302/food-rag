"""app/entity_boost.py:關鍵詞→chunk 對照查找。"""
import sqlite3

from app import entity_boost as eb


def test_substring_match_and_dedup(monkeypatch):
    monkeypatch.setattr(eb, "_entity_to_chunks", {"紅麴膠囊": [14031, 7], "膠囊": [7, 8]})
    assert eb.find_entity_boosted_chunk_ids("紅麴膠囊怎麼歸類?") == [14031, 7, 8]


def test_no_match_returns_empty(monkeypatch):
    monkeypatch.setattr(eb, "_entity_to_chunks", {"紅麴膠囊": [14031]})
    assert eb.find_entity_boosted_chunk_ids("所得稅怎麼申報?") == []


def test_missing_entity_map_degrades_to_noop(monkeypatch, tmp_path):
    monkeypatch.setattr(eb, "_entity_to_chunks", None)
    monkeypatch.setattr(eb, "_ENTITY_MAP_PATH", tmp_path / "missing.json")
    assert eb.find_entity_boosted_chunk_ids("紅麴膠囊") == []


def test_fetch_chunks_by_ids_roundtrip():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("""CREATE TABLE chunks (id INTEGER PRIMARY KEY, text TEXT, primary_law TEXT,
                  subtopic TEXT, document TEXT, kind TEXT, source_path TEXT,
                  is_ocr INTEGER, has_table INTEGER)""")
    db.execute("INSERT INTO chunks VALUES (5,'五','食安法第8條',NULL,'d','qa','p',0,1)")
    db.execute("INSERT INTO chunks VALUES (6,'六',NULL,NULL,'d','guide','p',1,0)")

    chunks = eb.fetch_chunks_by_ids(db, [6, 5, 999])
    by_id = {c.chunk_id: c for c in chunks}

    assert set(by_id) == {5, 6}                 # 不存在的 id 靜默略過
    assert by_id[5].has_table is True and by_id[5].is_ocr is False
    assert by_id[6].is_ocr is True and by_id[6].primary_law is None
    assert eb.fetch_chunks_by_ids(db, []) == []
