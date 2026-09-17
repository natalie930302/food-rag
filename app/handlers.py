"""
/query 的三條執行路徑。每條都拿同一個 RunContext(app/harness.py)記 trace 與 usage,
回傳同一種 HandlerResult——對外只有一個入口、一種回傳格式、一套邊界。

  run_regulation  法規問答:(複合問題先拆子問題)→ dense → entity boost → rerank → 信心閘門 → (改寫重試)
                  → 各子問題結果合併 → 生成 → 引用驗證
  run_review      廣告審稿:關鍵字掃描 → 保證撈第 28 條 + 全庫補 → 案例 → 生成三段報告 → 引用驗證 → verdict
  run_agent       多步:tool-calling agent(app/agent.py),trace 由工具紀錄轉成同一種 TraceStep

三條路徑的核心邏輯跟原本的 /ask、/review、/ask_agent 完全一樣,只是把「記錄」跟「檢查」
搬到 harness 層統一做。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from app import deps, llm, retrieval
from app.agent import run_agent
from app.agentic_retrieval import retrieve_agentic
from app.decompose import decompose
from app.harness import NO_EVIDENCE_ANSWER, RunContext, is_refusal, verify_and_regenerate
from app.retrieval import RetrievedCase, RetrievedChunk

# 跟舊 /ask 一樣的路由關鍵字:廣告/宣稱問題鎖定食安法第 28 條;提到罰則/案例就順帶查案例
_LABELING_KEYWORDS = ("標示", "標籤", "包裝", "基因改造", "GMO", "有機", "素食", "過敏原", "營養標示", "成分")
_AD_KEYWORDS = ("廣告", "宣稱", "宣傳", "文案", "行銷", "標榜", "聲稱")
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
_CASE_TRIGGERS = ("罰", "案例", "違規", "裁處")


@dataclass
class HandlerResult:
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    cases: list[RetrievedCase] = field(default_factory=list)
    confident: bool | None = None
    used_retry: bool | None = None
    refused: bool = False
    unsupported_citations: list[str] = field(default_factory=list)
    citation_regenerated: bool = False
    verdict: str | None = None
    matched_keywords: list[str] = field(default_factory=list)
    grounded: bool | None = None
    hit_tool_call_limit: bool | None = None
    retrieval_ms: int = 0
    llm_ms: int = 0


def _filters_for(question: str) -> dict | None:
    is_ad = any(kw in question for kw in _AD_KEYWORDS)
    is_claim = any(kw in question for kw in _AD_CLAIM_VERBS)
    is_label = any(kw in question for kw in _LABELING_KEYWORDS)
    if (is_ad or is_claim) and is_label:
        return None                                   # 廣告 + 標示 → 全庫(22/25/28 條都要)
    if is_ad or is_claim:
        return {"law_article": "食安法第28條"}
    return None


def _wants_cases(question: str) -> bool:
    return any(kw in question for kw in _AD_KEYWORDS + _AD_CLAIM_VERBS + _CASE_TRIGGERS)


def _retrieve_one(ctx: RunContext, db, em, fi, client, question: str, top_k: int, label: str = ""):
    """一個問句走一次「檢索 → 信心閘門 → 信心不足就 LLM 改寫重查」,把每一步記進 trace。"""
    t = time.time()
    r = retrieve_agentic(db, em, fi, deps.get_reranker(), client, question,
                         filters=_filters_for(question), top_k=top_k)
    first = r.attempts[0]
    ctx.step("retrieve", t, confident=first["confident"], top_score=first["top_score"],
             chunk_ids=[c.chunk_id for c in r.final_result.chunks] if not r.used_retry else [],
             detail=(label + " " if label else "") + "dense + entity boost + rerank + 信心閘門")
    if r.used_retry:
        ctx.llm_calls += 1                             # reformulate_query 呼叫了一次 LLM
        second = r.attempts[1]
        ctx.step("retry", detail=(label + " " if label else "") + f"LLM 改寫 → 「{second['question']}」",
                 confident=second["confident"], top_score=second["top_score"],
                 chunk_ids=[c.chunk_id for c in r.final_result.chunks])
    return r


def _case_as_chunk(c: RetrievedCase) -> RetrievedChunk:
    """案例當引用驗證的證據:答案引用案例的法條(「第28條第1項」)不該被判成瞎掰。"""
    return RetrievedChunk(
        chunk_id=-c.id, text=f"{c.year}年{c.month}月 {c.company or ''}「{c.product or ''}」違反{c.law_cited or ''},"
                             f"裁處 {c.penalty_twd or 0} 元。{(c.violation or '')[:200]}",
        primary_law=c.law_cited, subtopic=None, document=None, kind="case", source_path=None,
        is_ocr=False, has_table=False, score=c.score or 0.0,
    )


def run_regulation(ctx: RunContext, db, client, question: str, top_k: int = 8, include_cases: bool = False) -> HandlerResult:
    em, fi = deps.get_embed_model(), deps.get_faiss_chunks()
    t0 = time.time()

    # 複合問題(一句多問)整句送檢索會失敗(2026/09 實測 reranker 0.17 vs 單問 0.91):
    # 先拆成子問題各自檢索,結果合併;任一子問題過閘門就生成,答案逐部分標示有無依據。
    subs, used_llm = decompose(client, question)
    if used_llm:
        ctx.llm_calls += 1
    if len(subs) >= 2:
        ctx.step("decompose", t0, detail=" | ".join(subs), arguments={"subquestions": subs, "llm": used_llm})

    used_retry = False
    confident = False
    chunks: list[RetrievedChunk] = []
    seen: set[int] = set()
    per_sub = max(3, top_k // len(subs)) if len(subs) >= 2 else top_k
    for i, sub in enumerate(subs):
        r = _retrieve_one(ctx, db, em, fi, client, sub, per_sub, label=f"子問題{i + 1}" if len(subs) >= 2 else "")
        used_retry = used_retry or r.used_retry
        if r.final_result.confident:
            confident = True
            for c in r.final_result.chunks:
                if c.chunk_id not in seen:
                    seen.add(c.chunk_id)
                    chunks.append(c)

    cases: list[RetrievedCase] = []
    cases_confident = False
    if include_cases or any(_wants_cases(sub) for sub in subs):
        t = time.time()
        cases = retrieval.retrieve_cases(db, em, deps.get_faiss_cases(), question, top_k=3)
        cases_confident = retrieval.cases_are_confident(cases)
        ctx.step("retrieve_cases", t, chunk_ids=[c.id for c in cases], confident=cases_confident,
                 top_score=cases[0].score if cases else None)
    retrieval_ms = int((time.time() - t0) * 1000)

    # 「有哪些業者因為宣稱減肥被罰?罰了多少?」:法規那一跳信心不足,但案例查到了且相似度過門檻——
    # 案例本身就是回答依據,不能因為法規沒過閘門就整題拒答(2026/09 使用者實測抓到;agent 路徑同一規則)
    t1 = time.time()
    if not confident and not cases_confident:
        ctx.step("refuse", detail="法規與案例都信心不足,不呼叫 LLM")
        return HandlerResult(answer=NO_EVIDENCE_ANSWER, cases=cases, confident=False, used_retry=used_retry,
                             refused=True, retrieval_ms=retrieval_ms)
    if not cases_confident:
        cases = []                                     # 沒信心的案例不進 prompt,免得 LLM 拿無關案例湊答案

    system, user = llm.build_general_prompt(question, chunks, cases)
    if not confident:
        user += ("\n\n注意:這次沒有檢索到可信的法規條文,只有上列違規案例是可靠依據。"
                 "請只根據案例回答(廠商、年月、罰鍰、引用的法條),不要自行補充案例裡沒有的法規內容。")
    if len(subs) >= 2:
        listed = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(subs))
        user += ("\n\n這個問題包含多個子問題:\n" + listed +
                 "\n請依序逐一回答每個子問題。某個子問題在 context 裡沒有依據時,只針對那一部分寫"
                 "「資料庫中沒有找到與此相關的資料」,其餘子問題照答;不要因為一部分沒依據就整題拒答。")
    answer = llm.call_llm(client, system, user, ctx=ctx)
    ctx.step("generate", t1)
    evidence = list(chunks) + [_case_as_chunk(c) for c in cases]
    answer, unsupported, regenerated = verify_and_regenerate(
        ctx, answer, evidence, lambda fb: llm.call_llm(client, system, user + "\n\n" + fb, ctx=ctx))
    return HandlerResult(
        answer=answer, sources=chunks, cases=cases, confident=True, used_retry=used_retry,
        refused=is_refusal(answer), unsupported_citations=unsupported, citation_regenerated=regenerated,
        retrieval_ms=retrieval_ms, llm_ms=int((time.time() - t1) * 1000),
    )


def run_review(ctx: RunContext, db, client, ad_text: str, top_k: int = 5) -> HandlerResult:
    t0 = time.time()
    matched = llm.detect_risk_keywords(ad_text)
    ctx.step("keyword_scan", t0, detail=f"{len(matched)} 個高風險詞", arguments={"matched": matched})

    em, fi = deps.get_embed_model(), deps.get_faiss_chunks()
    t = time.time()
    guaranteed = retrieval.retrieve_chunks(db, em, fi, ad_text, filters={"law_article": "食安法第28條"}, top_k=top_k)
    supplement = retrieval.retrieve_chunks(db, em, fi, ad_text, filters=None, top_k=top_k)
    seen = {c.chunk_id for c in guaranteed}
    extra = sorted([c for c in supplement if c.chunk_id not in seen], key=lambda x: -x.score)[: max(0, top_k - len(guaranteed))]
    chunks = list(guaranteed) + extra
    ctx.step("retrieve", t, chunk_ids=[c.chunk_id for c in chunks], detail="保證第 28 條 + 全庫補位")

    t = time.time()
    cases = retrieval.retrieve_cases(db, em, deps.get_faiss_cases(), ad_text, top_k=5)
    ctx.step("retrieve_cases", t, chunk_ids=[c.id for c in cases])
    retrieval_ms = int((time.time() - t0) * 1000)

    t1 = time.time()
    system, user = llm.build_review_prompt(ad_text, chunks, cases, matched)
    answer = llm.call_llm(client, system, user, ctx=ctx)
    ctx.step("generate", t1)
    answer, unsupported, regenerated = verify_and_regenerate(
        ctx, answer, chunks, lambda fb: llm.call_llm(client, system, user + "\n\n" + fb, ctx=ctx))
    verdict = llm.infer_verdict(matched, answer, ad_text)
    answer = re.sub(r"\n*\**VERDICT\**\s*[:：]?\s*\**\s*(高風險|有疑慮|合規)\**\s*$", "", answer).strip()
    ctx.step("verdict", detail=verdict)
    return HandlerResult(
        answer=answer, sources=chunks, cases=cases, verdict=verdict, matched_keywords=matched,
        unsupported_citations=unsupported, citation_regenerated=regenerated,
        retrieval_ms=retrieval_ms, llm_ms=int((time.time() - t1) * 1000),
    )


def run_agent_route(ctx: RunContext, db, client, question: str, max_tool_calls: int = 4) -> HandlerResult:
    t0 = time.time()
    r = run_agent(db, deps.get_embed_model(), deps.get_faiss_chunks(), deps.get_reranker(), client,
                  question, max_tool_calls=max_tool_calls, faiss_cases=deps.get_faiss_cases())
    for rec in r.trace:
        detail = rec.result_summary + (" [工具內改寫重查]" if rec.retry_used else "")
        if rec.arguments.get("forced"):
            detail += " [harness 強制補查法規]"
        ctx.step(f"tool:{rec.name}", confident=rec.confident, top_score=rec.top_score, chunk_ids=rec.chunk_ids,
                 query_drift_detected=rec.query_drift_detected, arguments=rec.arguments, detail=detail)
    ctx.tool_calls += r.usage.tool_calls
    ctx.llm_calls += r.usage.llm_calls
    ctx.prompt_tokens += r.usage.prompt_tokens
    ctx.completion_tokens += r.usage.completion_tokens
    if r.usage.stop_reason != "answered":
        ctx.stop_reason = r.usage.stop_reason
    ctx.step("generate", detail=f"grounded={r.grounded} regenerated={r.citation_regenerated}")

    chunk_ids = [cid for rec in r.trace if rec.name == "search_regulations" and rec.confident for cid in rec.chunk_ids]
    sources = retrieval_chunks_by_ids(db, chunk_ids) if chunk_ids else []
    return HandlerResult(
        answer=r.answer, sources=sources, confident=r.grounded, refused=is_refusal(r.answer),
        unsupported_citations=r.unsupported_citations, citation_regenerated=r.citation_regenerated,
        grounded=r.grounded, hit_tool_call_limit=r.hit_tool_call_limit,
        llm_ms=int((time.time() - t0) * 1000),
    )


def retrieval_chunks_by_ids(db, chunk_ids: list[int]) -> list[RetrievedChunk]:
    from app.entity_boost import fetch_chunks_by_ids
    seen: list[int] = []
    for cid in chunk_ids:
        if cid not in seen:
            seen.append(cid)
    by_id = {c.chunk_id: c for c in fetch_chunks_by_ids(db, seen)}
    return [by_id[cid] for cid in seen if cid in by_id]
