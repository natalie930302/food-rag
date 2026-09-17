"""檢索層:SQL 預過濾 + FAISS 向量搜尋 + 違規案例查詢。

核心函式:
  - retrieve_chunks() : 主要法規/指引/QA 檢索
  - retrieve_cases()  : 違規案例檢索
  - get_co_cited_laws(): 法條關聯統計
"""
import sqlite3
from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedChunk:
    chunk_id: int
    text: str
    primary_law: str | None
    subtopic: str | None
    document: str | None
    kind: str | None
    source_path: str | None
    is_ocr: bool
    has_table: bool
    score: float


@dataclass
class RetrievedCase:
    id: int
    year: int
    month: int
    date: str | None
    product: str | None
    company: str | None
    violation: str
    penalty_twd: int | None
    law_cited: str | None
    article_no: int | None
    score: float | None = None


def _embed_query(model: SentenceTransformer, text: str) -> np.ndarray:
    """編碼單一查詢,normalize 後 reshape 給 FAISS。"""
    vec = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
    return vec.astype(np.float32)


def _get_candidate_embedding_ids(
    db: sqlite3.Connection, filters: dict | None
) -> list[int] | None:
    """根據 filters 查 SQLite,回傳符合的 embedding_id 清單。
    回 None = 不限制,FAISS 全庫搜尋。
    """
    if not filters:
        return None

    where: list[str] = []
    params: list = []

    # law_article 走多對多 JOIN(網狀關聯)
    if "law_article" in filters and filters["law_article"]:
        law = filters["law_article"]
        # 從字串抽條號
        import re
        m = re.search(r"第(\d+(?:之[一二三四五]|之\d+)?)條", law)
        if m:
            article_full = m.group(1)
            sql = """
                SELECT DISTINCT c.embedding_id
                FROM chunks c
                LEFT JOIN chunk_laws cl ON c.id = cl.chunk_id
                WHERE (c.primary_law = ? OR cl.article_full = ?)
                  AND c.embedding_id IS NOT NULL
            """
            rows = db.execute(sql, [law, article_full]).fetchall()
        else:
            rows = db.execute(
                "SELECT embedding_id FROM chunks WHERE primary_law = ?", [law]
            ).fetchall()
        return [r["embedding_id"] for r in rows]

    # 其他純 metadata 篩選
    if "category" in filters and filters["category"]:
        where.append("category = ?")
        params.append(filters["category"])
    if "subtopic" in filters and filters["subtopic"]:
        where.append("subtopic LIKE ?")
        params.append(f"%{filters['subtopic']}%")
    if "kind" in filters and filters["kind"]:
        where.append("kind = ?")
        params.append(filters["kind"])

    if not where:
        return None

    sql = "SELECT embedding_id FROM chunks WHERE " + " AND ".join(where)
    rows = db.execute(sql, params).fetchall()
    return [r["embedding_id"] for r in rows]


def _fetch_chunks_by_embedding_ids(
    db: sqlite3.Connection, eids: list[int], scores: list[float]
) -> list[RetrievedChunk]:
    """依 embedding_id 撈出 chunk metadata,組合分數。"""
    if not eids:
        return []
    placeholders = ",".join(["?"] * len(eids))
    sql = f"""
        SELECT id, text, primary_law, subtopic, document, kind,
               source_path, is_ocr, has_table, embedding_id
        FROM chunks WHERE embedding_id IN ({placeholders})
    """
    rows = db.execute(sql, eids).fetchall()
    # 建立 embedding_id → score map
    score_map = {eid: scores[i] for i, eid in enumerate(eids)}

    out = []
    for r in rows:
        out.append(RetrievedChunk(
            chunk_id=r["id"],
            text=r["text"],
            primary_law=r["primary_law"],
            subtopic=r["subtopic"],
            document=r["document"],
            kind=r["kind"],
            source_path=r["source_path"],
            is_ocr=bool(r["is_ocr"]),
            has_table=bool(r["has_table"]),
            score=score_map.get(r["embedding_id"], 0.0),
        ))
    # 按分數從高到低
    out.sort(key=lambda x: -x.score)
    return out


def retrieve_chunks(
    db: sqlite3.Connection,
    model: SentenceTransformer,
    faiss_index: faiss.Index,
    question: str,
    filters: dict | None = None,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """主檢索:用 metadata 預過濾,再做向量搜尋。"""
    if faiss_index.ntotal == 0:
        return []

    query_vec = _embed_query(model, question)
    candidate_eids = _get_candidate_embedding_ids(db, filters)

    if candidate_eids is None:
        # 全庫搜尋
        D, I = faiss_index.search(query_vec, top_k)
    elif not candidate_eids:
        return []
    else:
        # 用 IDSelector 限定範圍
        sel = faiss.IDSelectorArray(np.array(candidate_eids, dtype=np.int64))
        params = faiss.SearchParameters(sel=sel)
        D, I = faiss_index.search(query_vec, top_k, params=params)

    eids = [int(x) for x in I[0] if x >= 0]
    scores = [float(s) for s in D[0][: len(eids)]]
    return _fetch_chunks_by_embedding_ids(db, eids, scores)


# 案例向量相似度(cosine)門檻:FAISS 永遠回傳最近鄰,「查到 3 筆」不等於「有相關案例」。
# 2026/09 實測相關問題 0.60–0.66、無關問題(捷運票價、天氣)0.45–0.52。過門檻的案例才算回答依據。
CASE_GROUNDING_THRESHOLD = 0.58


def cases_are_confident(cases: list) -> bool:
    top = cases[0].score if cases and cases[0].score is not None else None
    return top is not None and top >= CASE_GROUNDING_THRESHOLD


def retrieve_cases(
    db: sqlite3.Connection,
    model: SentenceTransformer,
    faiss_index: faiss.Index,
    text: str,
    top_k: int = 3,
) -> list[RetrievedCase]:
    """案例檢索:用文字 embed 找最相似的違規案例。"""
    if faiss_index.ntotal == 0:
        return []

    query_vec = _embed_query(model, text)
    D, I = faiss_index.search(query_vec, top_k)

    eids = [int(x) for x in I[0] if x >= 0]
    scores = [float(s) for s in D[0][: len(eids)]]
    if not eids:
        return []

    placeholders = ",".join(["?"] * len(eids))
    rows = db.execute(
        f"""SELECT id, year, month, date, product, company, violation,
                   penalty_twd, law_cited, article_no, embedding_id
            FROM violations WHERE embedding_id IN ({placeholders})""",
        eids,
    ).fetchall()
    score_map = {eid: scores[i] for i, eid in enumerate(eids)}

    out = []
    for r in rows:
        out.append(RetrievedCase(
            id=r["id"],
            year=r["year"], month=r["month"],
            date=r["date"], product=r["product"], company=r["company"],
            violation=r["violation"], penalty_twd=r["penalty_twd"],
            law_cited=r["law_cited"], article_no=r["article_no"],
            score=score_map.get(r["embedding_id"], 0.0),
        ))
    out.sort(key=lambda x: -(x.score or 0))
    return out


def get_co_cited_laws(db: sqlite3.Connection, article_full: str, limit: int = 10):
    """法條關聯:此條被引用的 chunks 還引用了哪些其他法條。"""
    sql = """
        SELECT cl2.law_name AS related_law_name,
               cl2.article_full AS related_article,
               COUNT(DISTINCT cl1.chunk_id) AS co_count
        FROM chunk_laws cl1
        JOIN chunk_laws cl2 ON cl1.chunk_id = cl2.chunk_id
        WHERE cl1.article_full = ?
          AND NOT (cl2.law_name = cl1.law_name AND cl2.article_full = cl1.article_full)
        GROUP BY related_law_name, related_article
        ORDER BY co_count DESC
        LIMIT ?
    """
    return db.execute(sql, [article_full, limit]).fetchall()


def count_chunks_for_law(db: sqlite3.Connection, article_full: str) -> int:
    row = db.execute(
        """SELECT COUNT(DISTINCT chunk_id) AS n
           FROM chunk_laws WHERE article_full = ?""",
        [article_full],
    ).fetchone()
    return row["n"] if row else 0
