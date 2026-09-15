"""FastAPI 入口。

啟動:
  uvicorn app.main:app --reload --port 8000
"""
import logging
import re
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config.settings import settings, PROJECT_ROOT
from app import deps, retrieval, llm, schemas
from app.agentic_retrieval import retrieve_agentic
from app.agent import run_agent


logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("food-rag")

_LABELING_KEYWORDS = ("標示", "標籤", "包裝", "基因改造", "GMO", "有機", "素食", "過敏原", "營養標示", "成分")
_AD_KEYWORDS = ("廣告", "宣稱", "宣傳", "文案", "行銷", "標榜", "聲稱")

# 廣告常見功效動詞，出現即視同廣告問題 → 路由至第28條
_AD_CLAIM_VERBS = (
    "阻斷", "燃燒", "排出", "溶解", "抑制吸收",
    "瘦", "燃脂", "消脂", "去脂", "塑身", "纖體",
    "壯陽", "豐胸", "增高",
    "治療", "治癒", "根治", "抗癌", "防癌",
    "降血糖", "降血壓", "降血脂", "降膽固醇",
    "降低血糖", "降低血壓", "降低血脂", "降低膽固醇",
    "控制血糖", "穩定血糖", "改善血糖",
    "醫療效能", "疾病症狀", "適合糖尿病",
)


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
def ask(req: schemas.AskRequest, x_openai_key: str | None = Header(default=None)):
    db = deps.get_db()
    try:
        t0 = time.time()
        _em = deps.get_embed_model()
        _fi = deps.get_faiss_chunks()
        is_ad    = any(kw in req.question for kw in _AD_KEYWORDS)
        is_claim = any(kw in req.question for kw in _AD_CLAIM_VERBS)
        is_label = any(kw in req.question for kw in _LABELING_KEYWORDS)

        if (is_ad or is_claim) and is_label:
            # 廣告/宣稱 + 標示 → 全庫（22/25/28條都需要）
            filters = None
        elif is_ad or is_claim:
            # 廣告或功效宣稱 → 鎖定第28條
            filters = {"law_article": "食安法第28條"}
        else:
            filters = req.filters

        client = OpenAI(api_key=x_openai_key) if x_openai_key else deps.get_openai_client()

        agentic_result = retrieve_agentic(
            db=db, embed_model=_em, faiss_index=_fi,
            reranker=deps.get_reranker(), openai_client=client,
            question=req.question, filters=filters, top_k=req.top_k,
        )
        confident = agentic_result.final_result.confident
        chunks = agentic_result.final_result.chunks if confident else []

        cases = []
        if req.include_cases or is_ad or is_claim or any(
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

        # 信心不足時,誠實回報找不到可信依據,不硬答(Corrective RAG,見README)
        # 也不呼叫 LLM——省下一次呼叫的延遲跟費用,且避免LLM在沒有可信依據時自己腦補答案
        t1 = time.time()
        if not confident:
            answer = "目前資料庫裡沒有找到足夠可信的法規依據可以回答這個問題,建議換個問法,或直接洽詢主管機關確認。"
        else:
            system, user = llm.build_general_prompt(req.question, chunks, cases)
            answer = llm.call_llm(client, system, user)
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
                confident=confident,
                used_retry=agentic_result.used_retry,
            ),
        )
    finally:
        db.close()


@app.post("/ask_agent", response_model=schemas.AgentAskResponse)
def ask_agent(req: schemas.AgentAskRequest, x_openai_key: str | None = Header(default=None)):
    """跟 /ask 用同一套檢索/信心閘門邏輯,差別是讓 LLM 自己決定要不要重查、
    查哪個工具,而不是走 /ask 裡固定的「一次重試」流程。見 app/agent.py 說明。
    """
    db = deps.get_db()
    try:
        t0 = time.time()
        client = OpenAI(api_key=x_openai_key) if x_openai_key else deps.get_openai_client()
        result = run_agent(
            db=db, embed_model=deps.get_embed_model(), faiss_index=deps.get_faiss_chunks(),
            reranker=deps.get_reranker(), client=client,
            question=req.question, max_tool_calls=req.max_tool_calls,
        )
        total_ms = int((time.time() - t0) * 1000)

        return schemas.AgentAskResponse(
            answer=result.answer,
            trace=[
                schemas.AgentToolCall(
                    name=r.name, arguments=r.arguments, confident=r.confident,
                    top_score=r.top_score, chunk_ids=r.chunk_ids,
                    result_summary=r.result_summary,
                    query_drift_detected=r.query_drift_detected,
                ) for r in result.trace
            ],
            meta=schemas.QueryMeta(
                # 檢索跟LLM決策在迴圈裡交錯執行,拆不出各自的耗時,全部算進llm_ms
                retrieval_ms=0, llm_ms=total_ms,
                model=settings.openai_model,
                total_chunks_searched=deps.get_faiss_chunks().ntotal,
                confident=result.grounded,
            ),
            tool_calls_used=result.tool_calls_used,
            grounded=result.grounded,
            hit_tool_call_limit=result.hit_tool_call_limit,
        )
    finally:
        db.close()


@app.post("/review", response_model=schemas.ReviewResponse)
def review(req: schemas.ReviewRequest, x_openai_key: str | None = Header(default=None)):
    """廣告審稿:固定三段式輸出。"""
    db = deps.get_db()
    try:
        t0 = time.time()
        matched = llm.detect_risk_keywords(req.ad_text)

        # 兩段式：先保證撈到食安法第28條，再補全庫填滿 top_k
        _em = deps.get_embed_model()
        _fi = deps.get_faiss_chunks()
        guaranteed = retrieval.retrieve_chunks(
            db=db, model=_em, faiss_index=_fi,
            question=req.ad_text,
            filters={"law_article": "食安法第28條"},
            top_k=req.top_k,
        )
        supplement = retrieval.retrieve_chunks(
            db=db, model=_em, faiss_index=_fi,
            question=req.ad_text,
            filters=None,
            top_k=req.top_k,
        )
        # 28條優先佔前排，剩餘名額才給 supplement（不按分數淘汰保底的28條）
        seen: set[int] = {c.chunk_id for c in guaranteed}
        remaining = max(0, req.top_k - len(guaranteed))
        extra = sorted(
            [c for c in supplement if c.chunk_id not in seen],
            key=lambda x: -x.score,
        )[:remaining]
        chunks = list(guaranteed) + extra

        # 查相似違規案例
        cases = retrieval.retrieve_cases(
            db=db,
            model=deps.get_embed_model(),
            faiss_index=deps.get_faiss_cases(),
            text=req.ad_text,
            top_k=5,
        )
        retrieval_ms = int((time.time() - t0) * 1000)

        # 呼叫 LLM
        t1 = time.time()
        system, user = llm.build_review_prompt(req.ad_text, chunks, cases, matched)
        client = OpenAI(api_key=x_openai_key) if x_openai_key else deps.get_openai_client()
        answer = llm.call_llm(client, system, user)
        llm_ms = int((time.time() - t1) * 1000)

        verdict = llm.infer_verdict(matched, answer, req.ad_text)
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


_UI_DIST = PROJECT_ROOT / "ui_dist"
if not _UI_DIST.exists():
    _UI_DIST = PROJECT_ROOT.parent / "food-rag-ui" / "dist"
if _UI_DIST.exists():
    app.mount("/", StaticFiles(directory=_UI_DIST, html=True), name="ui")


