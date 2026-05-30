"""Step 2:把 JSONL → SQLite + FAISS。

流程:
  1. 讀 chunks.jsonl + violations.jsonl
  2. 用 BGE-M3 編碼所有 text
  3. 建 SQLite schema 並寫入 metadata、法條多對多、案例、失敗檔
  4. 建 FAISS IndexFlatIP 並寫入向量
"""

import json
import sqlite3
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config.settings import settings, ensure_dirs


SCHEMA_SQL = """
DROP TABLE IF EXISTS chunks;
DROP TABLE IF EXISTS chunk_laws;
DROP TABLE IF EXISTS violations;
DROP TABLE IF EXISTS failed_files;
DROP TABLE IF EXISTS extracted_tables;

CREATE TABLE chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    text          TEXT NOT NULL,
    category      TEXT,
    primary_law   TEXT,
    subtopic      TEXT,
    document      TEXT,
    kind          TEXT,
    is_ocr        INTEGER DEFAULT 0,
    has_table     INTEGER DEFAULT 0,
    char_len      INTEGER,
    chunk_strategy TEXT,
    source_path   TEXT NOT NULL,
    embedding_id  INTEGER UNIQUE NOT NULL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_chunks_primary_law ON chunks(primary_law);
CREATE INDEX idx_chunks_subtopic ON chunks(subtopic);
CREATE INDEX idx_chunks_kind ON chunks(kind);
CREATE INDEX idx_chunks_category ON chunks(category);
CREATE INDEX idx_chunks_embedding_id ON chunks(embedding_id);

CREATE TABLE chunk_laws (
    chunk_id      INTEGER NOT NULL,
    law_name      TEXT NOT NULL,
    article_no    INTEGER NOT NULL,
    article_full  TEXT NOT NULL,
    paragraph     TEXT,
    role          TEXT NOT NULL,
    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
);
CREATE INDEX idx_cl_law ON chunk_laws(law_name, article_no);
CREATE INDEX idx_cl_chunk ON chunk_laws(chunk_id);
CREATE INDEX idx_cl_role ON chunk_laws(role);

CREATE TABLE violations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    year          INTEGER NOT NULL,
    month         INTEGER NOT NULL,
    date          TEXT,
    product       TEXT,
    channel       TEXT,
    violation     TEXT NOT NULL,
    company       TEXT,
    penalty_twd   INTEGER,
    law_cited     TEXT,
    article_no    INTEGER,
    source_file   TEXT,
    embedding_id  INTEGER UNIQUE NOT NULL
);
CREATE INDEX idx_viol_article ON violations(article_no);
CREATE INDEX idx_viol_company ON violations(company);
CREATE INDEX idx_viol_penalty ON violations(penalty_twd);
CREATE INDEX idx_viol_date ON violations(year, month);

CREATE TABLE failed_files (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path     TEXT NOT NULL,
    file_name       TEXT NOT NULL,
    file_ext        TEXT,
    page_count      INTEGER,
    failure_reason  TEXT NOT NULL,
    inferred_law    TEXT,
    inferred_topic  TEXT,
    note            TEXT,
    detected_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_failed_reason ON failed_files(failure_reason);

"""


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def init_db(db_path: Path) -> sqlite3.Connection:
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def encode_texts(model: SentenceTransformer, texts: list[str],
                  batch_size: int, desc: str) -> np.ndarray:
    """批次編碼。回傳 normalized (n, dim) float32 array(用於 IP 內積)。"""
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32)


def build_faiss(vectors: np.ndarray) -> faiss.Index:
    """IndexFlatIP:對 normalize 過的向量做內積 = 餘弦相似度。"""
    if vectors.shape[0] == 0:
        # 空索引也要建,避免 API 載入失敗
        return faiss.IndexFlatIP(1024)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def main():
    ensure_dirs()

    # === 讀 JSONL ===
    chunks_path = settings.processed_dir / "chunks.jsonl"
    violations_path = settings.processed_dir / "violations.jsonl"
    failed_path = settings.processed_dir / "failed_files.jsonl"
    chunks = load_jsonl(chunks_path)
    violations = load_jsonl(violations_path)
    failed = load_jsonl(failed_path)

    print(f"讀取結果:")
    print(f"  chunks    : {len(chunks)}")
    print(f"  violations: {len(violations)}")
    print(f"  failed    : {len(failed)}")

    # 去除完全相同文字的重複 chunk(保留第一個出現的)
    seen_texts: set[str] = set()
    deduped: list[dict] = []
    for c in chunks:
        t = c["text"].strip()
        if t not in seen_texts:
            seen_texts.add(t)
            deduped.append(c)
    if len(deduped) < len(chunks):
        print(f"  去重後 chunks: {len(deduped)}(移除 {len(chunks) - len(deduped)} 筆重複)")
    chunks = deduped

    if not chunks and not violations:
        print("\n沒有任何 chunk 或案例可入庫,先跑 step1_parse.py")
        sys.exit(1)

    # === 載入 embedding 模型 ===
    print(f"\n載入 embedding 模型: {settings.embed_model}(裝置:{settings.embed_device})")
    print("(第一次跑會下載 ~2.3 GB,請耐心等候)")
    model = SentenceTransformer(settings.embed_model, device=settings.embed_device)

    # === 編碼 chunks ===
    print(f"\n[1/2] 編碼 chunks ({len(chunks)} 筆)...")
    chunk_texts = [c["text"] for c in chunks]
    chunk_vecs = encode_texts(model, chunk_texts, settings.embed_batch_size, "chunks")

    # === 編碼 violations(用違規情節欄位) ===
    print(f"\n[2/2] 編碼 violations ({len(violations)} 筆)...")
    viol_texts = [v["violation"] for v in violations]
    viol_vecs = encode_texts(model, viol_texts, settings.embed_batch_size, "violations")

    # === 寫 SQLite ===
    print(f"\n寫入 SQLite: {settings.db_path}")
    conn = init_db(settings.db_path)
    cur = conn.cursor()

    # chunks(embedding_id = FAISS 內部索引 = i)
    for i, c in enumerate(tqdm(chunks, desc="寫 chunks")):
        cur.execute(
            """INSERT INTO chunks (
                text, category, primary_law, subtopic, document, kind,
                is_ocr, has_table, char_len, chunk_strategy, source_path, embedding_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                c["text"], c.get("category"), c.get("primary_law"),
                c.get("subtopic"), c.get("document"), c.get("kind"),
                1 if c.get("is_ocr") else 0,
                1 if c.get("has_table") else 0,
                c.get("char_len"), c.get("chunk_strategy"),
                c.get("source_path"), i,
            ),
        )
        chunk_id = cur.lastrowid
        for law in c.get("law_refs", []):
            cur.execute(
                """INSERT INTO chunk_laws (
                    chunk_id, law_name, article_no, article_full, paragraph, role
                ) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    chunk_id, law["law_name"], law["article_no"],
                    law["article_full"], law.get("paragraph"), law["role"],
                ),
            )

    # violations
    for i, v in enumerate(tqdm(violations, desc="寫 violations")):
        cur.execute(
            """INSERT INTO violations (
                year, month, date, product, channel, violation,
                company, penalty_twd, law_cited, article_no,
                source_file, embedding_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                v["year"], v["month"], v.get("date"), v.get("product"),
                v.get("channel"), v["violation"], v.get("company"),
                v.get("penalty_twd"), v.get("law_cited"), v.get("article_no"),
                v.get("source_file"), i,
            ),
        )

    # failed_files
    for f in failed:
        cur.execute(
            """INSERT INTO failed_files (
                source_path, file_name, file_ext, page_count,
                failure_reason, inferred_law, inferred_topic, note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                f.get("source_path"), f.get("file_name"), f.get("file_ext"),
                f.get("page_count"), f.get("failure_reason"),
                f.get("inferred_law"), f.get("inferred_topic"), f.get("note"),
            ),
        )

    conn.commit()
    conn.close()

    # === 寫 FAISS ===
    print(f"\n寫入 FAISS: {settings.faiss_chunks_path}")
    chunk_index = build_faiss(chunk_vecs)
    faiss.write_index(chunk_index, str(settings.faiss_chunks_path))

    print(f"寫入 FAISS: {settings.faiss_cases_path}")
    case_index = build_faiss(viol_vecs)
    faiss.write_index(case_index, str(settings.faiss_cases_path))

    print("\n" + "=" * 60)
    print("INGEST Step 2 完成")
    print("=" * 60)
    print(f"  SQLite              : {settings.db_path}")
    print(f"  FAISS (chunks)      : {settings.faiss_chunks_path} ({chunk_vecs.shape})")
    print(f"  FAISS (cases)       : {settings.faiss_cases_path} ({viol_vecs.shape})")
    print(f"\n啟動 API:  make run\n")


if __name__ == "__main__":
    main()
