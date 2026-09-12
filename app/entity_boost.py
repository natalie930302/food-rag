"""
針對 README 記錄的 Agentic RAG 誠實負向結果(問題:答案是嵌在較大範疇定義
裡的一個列舉例子,例如「紅麴膠囊」只是「營養補充食品」定義段落裡的一個
例子,dense retrieval 抓不準)所做的局部修補。

不重寫 chunker、不重跑 ingest,只針對 scripts/extract_chunk_entities.py
量測出的窄範圍問題(0.73%~0.63%的chunks)做一個輕量的關鍵詞查找層:
如果查詢字串包含某個從定義段落列舉例子裡抽出來的具體名詞,就把對應的
chunk 額外加進候選池,交給既有的 cross-encoder reranker 跟信心閘門去
判斷該不該用——這一層只負責「不要漏掉候選」,相關性判斷仍然交給下游
既有機制,不會繞過品質把關。
"""
from __future__ import annotations

import json
from pathlib import Path

from app.retrieval import RetrievedChunk

_ENTITY_MAP_PATH = Path(__file__).parent.parent / "data" / "index" / "chunk_entities.json"
_entity_to_chunks: dict[str, list[int]] | None = None


def _load_entity_map() -> dict:
    global _entity_to_chunks
    if _entity_to_chunks is None:
        if _ENTITY_MAP_PATH.exists():
            with open(_ENTITY_MAP_PATH, encoding="utf-8") as f:
                _entity_to_chunks = json.load(f)
        else:
            _entity_to_chunks = {}
    return _entity_to_chunks


def find_entity_boosted_chunk_ids(query: str) -> list[int]:
    """查詢字串裡如果包含任何已知的列舉實體詞,回傳對應的 chunk id。"""
    entity_map = _load_entity_map()
    matched_ids: list[int] = []
    for entity, chunk_ids in entity_map.items():
        if entity in query:
            for cid in chunk_ids:
                if cid not in matched_ids:
                    matched_ids.append(cid)
    return matched_ids


def fetch_chunks_by_ids(db, chunk_ids: list[int]) -> list[RetrievedChunk]:
    if not chunk_ids:
        return []
    placeholders = ",".join(["?"] * len(chunk_ids))
    rows = db.execute(
        f"""SELECT id, text, primary_law, subtopic, document, kind,
                   source_path, is_ocr, has_table
            FROM chunks WHERE id IN ({placeholders})""",
        chunk_ids,
    ).fetchall()
    return [
        RetrievedChunk(
            chunk_id=r["id"], text=r["text"], primary_law=r["primary_law"],
            subtopic=r["subtopic"], document=r["document"], kind=r["kind"],
            source_path=r["source_path"], is_ocr=bool(r["is_ocr"]),
            has_table=bool(r["has_table"]), score=0.0,
        )
        for r in rows
    ]
