"""FastAPI 入口。

啟動:
  uvicorn app.main:app --reload --port 8000
"""
import logging
import re
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from config.settings import settings
from app import deps, retrieval, llm, schemas


logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("food-rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API 啟動,預載資源中...")
    deps.preload_all()
    chunks_idx = deps.get_faiss_chunks()
    cases_idx = deps.get_faiss_cases()
    logger.info(f"FAISS chunks: {chunks_idx.ntotal} vectors")
    logger.info(f"FAISS cases : {cases_idx.ntotal} vectors")
    logger.info(f"OpenAI 模型 : {settings.openai_model}")
    logger.info("Ready.")
    yield
    logger.info("API 關閉")


app = FastAPI(
    title="食品法規 RAG API",
    description="食藥署法規 + 北市違規案例的問答 API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ 端點 ============

@app.post("/ask", response_model=schemas.AskResponse)
def ask(req: schemas.AskRequest):
    db = deps.get_db()
    try:
        t0 = time.time()
        chunks = retrieval.retrieve_chunks(
            db=db,
            model=deps.get_embed_model(),
            faiss_index=deps.get_faiss_chunks(),
            question=req.question,
            filters=req.filters,
            top_k=req.top_k,
        )
        cases = []
        if req.include_cases or any(
            kw in req.question for kw in ("罰", "案例", "違規", "裁處")
        ):
            cases = retrieval.retrieve_cases(
                db=db,
                model=deps.get_embed_model(),
                faiss_index=deps.get_faiss_cases(),
                text=req.question,
                top_k=3,
            )
        retrieval_ms = int((time.time() - t0) * 1000)

        # 呼叫 LLM
        t1 = time.time()
        system, user = llm.build_general_prompt(req.question, chunks, cases)
        answer = llm.call_llm(deps.get_openai_client(), system, user)
        llm_ms = int((time.time() - t1) * 1000)

        return schemas.AskResponse(
            answer=answer,
            sources=[
                schemas.SourceChunk(
                    chunk_id=c.chunk_id, text=c.text,
                    primary_law=c.primary_law, subtopic=c.subtopic,
                    document=c.document, kind=c.kind,
                    source_path=c.source_path,
                    is_ocr=c.is_ocr, has_table=c.has_table,
                    score=c.score,
                ) for c in chunks
            ],
            related_cases=[
                schemas.ViolationCase(
                    id=c.id, year=c.year, month=c.month, date=c.date,
                    product=c.product, company=c.company, violation=c.violation,
                    penalty_twd=c.penalty_twd, law_cited=c.law_cited,
                    article_no=c.article_no, score=c.score,
                ) for c in cases
            ],
            meta=schemas.QueryMeta(
                retrieval_ms=retrieval_ms,
                llm_ms=llm_ms,
                model=settings.openai_model,
                total_chunks_searched=deps.get_faiss_chunks().ntotal,
            ),
        )
    finally:
        db.close()


@app.post("/review", response_model=schemas.ReviewResponse)
def review(req: schemas.ReviewRequest):
    """廣告審稿:固定三段式輸出。"""
    db = deps.get_db()
    try:
        t0 = time.time()
        matched = llm.detect_risk_keywords(req.ad_text)

        # 檢索:用整段廣告文案
        chunks = retrieval.retrieve_chunks(
            db=db,
            model=deps.get_embed_model(),
            faiss_index=deps.get_faiss_chunks(),
            question=req.ad_text,
            filters={"law_article": "食安法第28條"},  # 廣告主要看 28 條
            top_k=req.top_k,
        )
        # 並行查相似案例
        cases = retrieval.retrieve_cases(
            db=db,
            model=deps.get_embed_model(),
            faiss_index=deps.get_faiss_cases(),
            text=req.ad_text,
            top_k=3,
        )
        retrieval_ms = int((time.time() - t0) * 1000)

        # 呼叫 LLM
        t1 = time.time()
        system, user = llm.build_review_prompt(req.ad_text, chunks, cases, matched)
        answer = llm.call_llm(deps.get_openai_client(), system, user)
        llm_ms = int((time.time() - t1) * 1000)

        verdict = llm.infer_verdict(matched, answer)
        answer = re.sub(r"\n*VERDICT:\s*(高風險|有疑慮|合規)\s*$", "", answer).strip()

        sources = [
            schemas.SourceChunk(
                chunk_id=c.chunk_id, text=c.text,
                primary_law=c.primary_law, subtopic=c.subtopic,
                document=c.document, kind=c.kind,
                source_path=c.source_path,
                is_ocr=c.is_ocr, has_table=c.has_table,
                score=c.score,
            ) for c in chunks
        ]
        case_list = [
            schemas.ViolationCase(
                id=c.id, year=c.year, month=c.month, date=c.date,
                product=c.product, company=c.company, violation=c.violation,
                penalty_twd=c.penalty_twd, law_cited=c.law_cited,
                article_no=c.article_no, score=c.score,
            ) for c in cases
        ]

        return schemas.ReviewResponse(
            verdict=verdict,
            answer=answer,
            evidence=schemas.ReviewEvidence(
                matched_keywords=matched, laws=sources, cases=case_list,
            ),
            meta=schemas.QueryMeta(
                retrieval_ms=retrieval_ms, llm_ms=llm_ms,
                model=settings.openai_model,
                total_chunks_searched=deps.get_faiss_chunks().ntotal,
            ),
        )
    finally:
        db.close()


@app.get("/stats", response_model=schemas.StatsResponse)
def stats():
    db = deps.get_db()
    try:
        total_chunks = db.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        total_violations = db.execute("SELECT COUNT(*) AS n FROM violations").fetchone()["n"]
        failed_count = db.execute("SELECT COUNT(*) AS n FROM failed_files").fetchone()["n"]

        by_kind = dict(db.execute(
            "SELECT kind, COUNT(*) AS n FROM chunks GROUP BY kind"
        ).fetchall())
        by_category = dict(db.execute(
            "SELECT category, COUNT(*) AS n FROM chunks GROUP BY category"
        ).fetchall())
        by_law = dict(db.execute(
            """SELECT primary_law, COUNT(*) AS n FROM chunks
               WHERE primary_law IS NOT NULL
               GROUP BY primary_law ORDER BY n DESC LIMIT 20"""
        ).fetchall())

        ocr_count = db.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE is_ocr = 1"
        ).fetchone()["n"]
        table_count = db.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE has_table = 1"
        ).fetchone()["n"]

        return schemas.StatsResponse(
            total_chunks=total_chunks,
            total_violations=total_violations,
            failed_files_count=failed_count,
            by_kind={k or "(none)": v for k, v in by_kind.items()},
            by_category={k or "(none)": v for k, v in by_category.items()},
            by_primary_law={k or "(none)": v for k, v in by_law.items()},
            ocr_chunks=ocr_count,
            chunks_with_tables=table_count,
        )
    finally:
        db.close()


@app.get("/failed", response_model=list[schemas.FailedFile])
def failed_files(limit: int = 200):
    db = deps.get_db()
    try:
        rows = db.execute(
            "SELECT * FROM failed_files ORDER BY id LIMIT ?", [limit]
        ).fetchall()
        return [
            schemas.FailedFile(
                file_name=r["file_name"],
                source_path=r["source_path"],
                file_ext=r["file_ext"],
                page_count=r["page_count"],
                failure_reason=r["failure_reason"],
                inferred_law=r["inferred_law"],
                inferred_topic=r["inferred_topic"],
                note=r["note"],
            ) for r in rows
        ]
    finally:
        db.close()


@app.get("/laws/{article}/related", response_model=schemas.LawRelatedResponse)
def law_related(article: str):
    """法條關聯查詢。article 範例:'28' 或 '15之一'。"""
    db = deps.get_db()
    try:
        rows = retrieval.get_co_cited_laws(db, article)
        total = retrieval.count_chunks_for_law(db, article)
        co = [
            schemas.RelatedLaw(
                law_name=r["related_law_name"],
                article_full=r["related_article"],
                co_occurrence=r["co_count"],
            ) for r in rows
        ]
        return schemas.LawRelatedResponse(
            law=article, total_chunks=total, co_cited_laws=co,
        )
    finally:
        db.close()


INLINE_TYPES = {
    ".pdf": "application/pdf",
    ".txt": "text/plain; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

@app.get("/files/{file_path:path}")
def serve_file(file_path: str):
    """在瀏覽器直接顯示 data/raw/ 下的原始檔案（PDF/圖片/文字 inline；其他才下載）。"""
    full_path = (settings.raw_dir / file_path).resolve()
    if not str(full_path).startswith(str(settings.raw_dir.resolve())):
        raise HTTPException(status_code=403, detail="Access denied")
    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    media_type = INLINE_TYPES.get(full_path.suffix.lower())
    if media_type:
        return FileResponse(full_path, media_type=media_type,
                            headers={"Content-Disposition": "inline"})
    return FileResponse(full_path, filename=full_path.name)


