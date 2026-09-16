"""FastAPI 入口。

對外只有兩個功能端點:
  POST /query    所有問答 / 審稿 / 多步問題。先路由(app/router.py)再分派到 app/handlers.py
                 的三條路徑,三條路徑共用同一套 harness(app/harness.py):同一種 trace、usage、
                 拒答契約、引用驗證
  GET  /health   系統狀態:模型/索引載好了沒、裝置、索引統計、失敗檔清單

資料端點(給前端法條關聯頁與原始檔連結用,不是功能):
  GET  /laws/{article}/related
  GET  /files/{path}

啟動:uvicorn app.main:app --reload --port 8000
"""
import logging
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

from app import deps, handlers, retrieval, schemas
from app.harness import RunContext
from app.router import route
from config.settings import PROJECT_ROOT, settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("food-rag")
_STARTED_AT = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API 啟動,預載資源中...")
    deps.preload_all()
    logger.info(f"FAISS chunks: {deps.get_faiss_chunks().ntotal} vectors")
    logger.info(f"FAISS cases : {deps.get_faiss_cases().ntotal} vectors")
    logger.info(f"OpenAI 模型 : {settings.openai_model}")
    logger.info(f"reranker    : {settings.reranker_device} fp16={settings.reranker_fp16}")
    logger.info("Ready.")
    yield
    logger.info("API 關閉")


app = FastAPI(
    title="食品法規 RAG API",
    description="食藥署法規 + 北市違規案例的問答 API。功能端點只有 /query 與 /health。",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _client(x_openai_key: str | None) -> OpenAI:
    if x_openai_key:
        return OpenAI(api_key=x_openai_key, base_url=settings.openai_base_url or None)
    return deps.get_openai_client()


def _sources(chunks) -> list[schemas.SourceChunk]:
    return [
        schemas.SourceChunk(
            chunk_id=c.chunk_id, text=c.text, primary_law=c.primary_law, subtopic=c.subtopic,
            document=c.document, kind=c.kind, source_path=c.source_path,
            is_ocr=c.is_ocr, has_table=c.has_table, score=c.score,
        ) for c in chunks
    ]


def _cases(cases) -> list[schemas.ViolationCase]:
    return [
        schemas.ViolationCase(
            id=c.id, year=c.year, month=c.month, date=c.date, product=c.product, company=c.company,
            violation=c.violation, penalty_twd=c.penalty_twd, law_cited=c.law_cited,
            article_no=c.article_no, score=c.score,
        ) for c in cases
    ]


# ============ 功能端點 ============

@app.post("/query", response_model=schemas.QueryResponse)
def query(req: schemas.QueryRequest, x_openai_key: str | None = Header(default=None)):
    """單一入口。

    1. 路由:規則層零成本判定,判不出來才問一次 LLM;呼叫端也可以 force_intent 指定
    2. 分派:regulation_qa / case_lookup → 固定管線;ad_review → 審稿;multi_hop → tool-calling agent
       (單跳問題上 agent 跟固定管線一樣準但慢一倍,所以只有多步才用,見 README §3、§6)
    3. harness:不管走哪條,回來都是同一種 trace / usage / meta,拒答與引用驗證同一套
    """
    ctx = RunContext(max_seconds=req.max_seconds)
    client = _client(x_openai_key)

    t = time.time()
    if req.force_intent:
        intent, source, reason = req.force_intent, "forced", ""
    else:
        d = route(req.question, client=client, use_llm=req.use_llm_router)
        intent, source, reason = d.intent, d.source, d.reason
        if d.source == "llm":
            ctx.llm_calls += 1
    ctx.step("route", t, detail=f"{intent} ({source}) {reason}")
    router_ms = int((time.time() - t) * 1000)

    db = deps.get_db()
    try:
        if intent == "ad_review":
            handler = "review"
            r = handlers.run_review(ctx, db, client, req.question, top_k=min(req.top_k, 20))
        elif intent == "multi_hop":
            handler = "agent"
            r = handlers.run_agent_route(ctx, db, client, req.question, max_tool_calls=req.max_tool_calls)
        else:
            handler = "regulation"
            r = handlers.run_regulation(ctx, db, client, req.question, top_k=req.top_k,
                                        include_cases=(intent == "case_lookup"))
    finally:
        db.close()

    return schemas.QueryResponse(
        answer=r.answer,
        route=schemas.RouteInfo(intent=intent, source=source, reason=reason, handler=handler, router_ms=router_ms),
        sources=_sources(r.sources),
        related_cases=_cases(r.cases),
        verdict=r.verdict,
        matched_keywords=r.matched_keywords,
        trace=[schemas.TraceStep(**s.as_dict()) for s in ctx.trace],
        usage=schemas.Usage(**ctx.usage()),
        meta=schemas.QueryMeta(
            model=settings.openai_model,
            total_chunks_searched=deps.get_faiss_chunks().ntotal,
            retrieval_ms=r.retrieval_ms, llm_ms=r.llm_ms,
            confident=r.confident, used_retry=r.used_retry, refused=r.refused,
            unsupported_citations=r.unsupported_citations, citation_regenerated=r.citation_regenerated,
            grounded=r.grounded, hit_tool_call_limit=r.hit_tool_call_limit,
        ),
    )


@app.get("/health", response_model=schemas.HealthResponse)
def health():
    """系統狀態。任一關鍵資源沒載好就回 degraded(HTTP 仍 200,讓前端能顯示是哪一項)。"""
    checks = {
        "index": settings.db_path.exists() and settings.faiss_chunks_path.exists(),
        "cases_index": settings.faiss_cases_path.exists(),
        "embed_model": deps.get_embed_model.cache_info().currsize > 0,
        "reranker": deps.get_reranker.cache_info().currsize > 0,
        "openai_key": bool(settings.openai_api_key),
    }
    db = deps.get_db()
    try:
        def count(sql: str) -> int:
            return db.execute(sql).fetchone()[0]

        index = schemas.IndexStats(
            total_chunks=count("SELECT COUNT(*) FROM chunks"),
            total_violations=count("SELECT COUNT(*) FROM violations"),
            failed_files_count=count("SELECT COUNT(*) FROM failed_files"),
            by_kind={k or "(none)": v for k, v in db.execute("SELECT kind, COUNT(*) FROM chunks GROUP BY kind").fetchall()},
            by_category={k or "(none)": v for k, v in db.execute("SELECT category, COUNT(*) FROM chunks GROUP BY category").fetchall()},
            by_primary_law={k or "(none)": v for k, v in db.execute(
                "SELECT primary_law, COUNT(*) FROM chunks WHERE primary_law IS NOT NULL GROUP BY primary_law ORDER BY 2 DESC LIMIT 20"
            ).fetchall()},
            ocr_chunks=count("SELECT COUNT(*) FROM chunks WHERE is_ocr = 1"),
            chunks_with_tables=count("SELECT COUNT(*) FROM chunks WHERE has_table = 1"),
        )
        failed = [
            schemas.FailedFile(
                file_name=r["file_name"], source_path=r["source_path"], file_ext=r["file_ext"],
                page_count=r["page_count"], failure_reason=r["failure_reason"],
                inferred_law=r["inferred_law"], inferred_topic=r["inferred_topic"], note=r["note"],
            ) for r in db.execute("SELECT * FROM failed_files ORDER BY id LIMIT 200").fetchall()
        ]
    finally:
        db.close()

    return schemas.HealthResponse(
        status="ok" if all(checks.values()) else "degraded",
        checks=checks,
        embed_device=settings.embed_device,
        reranker_device=settings.reranker_device,
        reranker_fp16=settings.reranker_fp16,
        gpu_available=torch.cuda.is_available(),
        model=settings.openai_model,
        uptime_s=round(time.time() - _STARTED_AT, 1),
        index=index,
        failed_files=failed,
    )


# ============ 資料端點 ============

@app.get("/laws/{article}/related", response_model=schemas.LawRelatedResponse)
def law_related(article: str):
    """法條關聯查詢。article 範例:'28' 或 '15之一'。"""
    db = deps.get_db()
    try:
        rows = retrieval.get_co_cited_laws(db, article)
        total = retrieval.count_chunks_for_law(db, article)
        return schemas.LawRelatedResponse(
            law=article, total_chunks=total,
            co_cited_laws=[
                schemas.RelatedLaw(law_name=r["related_law_name"], article_full=r["related_article"],
                                   co_occurrence=r["co_count"]) for r in rows
            ],
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
    """在瀏覽器直接顯示 data/raw/ 下的原始檔案(PDF/圖片/文字 inline;其他才下載)。"""
    full_path = (settings.raw_dir / file_path).resolve()
    if not str(full_path).startswith(str(settings.raw_dir.resolve())):
        raise HTTPException(status_code=403, detail="Access denied")
    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    media_type = INLINE_TYPES.get(full_path.suffix.lower())
    if media_type:
        return FileResponse(full_path, media_type=media_type, headers={"Content-Disposition": "inline"})
    return FileResponse(full_path, filename=full_path.name)


_UI_DIST = PROJECT_ROOT / "ui_dist"
if not _UI_DIST.exists():
    _UI_DIST = PROJECT_ROOT.parent / "food-rag-ui" / "dist"
if _UI_DIST.exists():
    app.mount("/", StaticFiles(directory=_UI_DIST, html=True), name="ui")
