"""API 請求與回應的 Pydantic schemas。

對外只有兩個功能端點:POST /query(所有問答/審稿/多步)與 GET /health(系統狀態)。
/laws/{article}/related 與 /files/... 是資料端點,給前端的法條關聯頁與原始檔連結用。
"""
from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal["regulation_qa", "case_lookup", "ad_review", "multi_hop"]


# ============ 請求 ============

class QueryRequest(BaseModel):
    """單一入口:先路由(app/router.py)再分派,三條路徑共用同一套 harness(app/harness.py)。"""
    question: str = Field(..., min_length=1, description="使用者輸入:問題,或一段待審的廣告文案")
    force_intent: Intent | None = Field(
        default=None, description="跳過路由、直接指定走哪條(前端的審稿分頁、評估腳本用)")
    use_llm_router: bool = Field(
        default=True, description="規則判不出來時是否用 LLM 分類;False 則一律走固定管線")
    top_k: int = Field(default=8, ge=1, le=20)
    max_tool_calls: int = Field(default=4, ge=1, le=8, description="multi_hop 路徑的工具呼叫上限")
    max_seconds: float = Field(default=90.0, ge=5, le=600, description="整次請求的時間預算")


# ============ 共用 ============

class SourceChunk(BaseModel):
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


class ViolationCase(BaseModel):
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


class RouteInfo(BaseModel):
    intent: Intent
    source: Literal["rules", "llm", "fallback", "forced"] = Field(
        description="rules=關鍵字規則零成本判定;llm=規則判不出來、問了一次 gpt-4o-mini;fallback=LLM 回傳無效或未啟用;forced=呼叫端指定")
    reason: str = ""
    handler: Literal["regulation", "review", "agent"]
    router_ms: int


class TraceStep(BaseModel):
    """三條執行路徑共用的步驟紀錄。固定管線:retrieve / retry / retrieve_cases / generate /
    verify_citations / refuse;審稿:keyword_scan / retrieve / … / verdict;agent:tool:<工具名> / generate。"""
    name: str
    ms: int = 0
    detail: str = ""
    confident: bool | None = None
    top_score: float | None = None
    chunk_ids: list[int] = []
    query_drift_detected: bool = False
    arguments: dict = {}


class Usage(BaseModel):
    llm_calls: int
    tool_calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    elapsed_s: float
    stop_reason: str = Field(description="answered / tool_calls / tokens / seconds")


class QueryMeta(BaseModel):
    model: str
    total_chunks_searched: int
    retrieval_ms: int
    llm_ms: int
    confident: bool | None = Field(default=None, description="信心閘門是否通過(審稿路徑不適用 → null)")
    used_retry: bool | None = Field(default=None, description="固定管線是否觸發了 LLM 改寫重試")
    refused: bool = Field(description="是否為拒答(程式碼契約:沒有可信依據就不硬答)")
    unsupported_citations: list[str] = Field(
        default=[], description="答案引用、但檢索內容裡沒出現的條號(已重生成一次仍未修正);空清單代表全部有依據")
    citation_regenerated: bool = False
    grounded: bool | None = Field(default=None, description="agent 路徑:迴圈裡是否有任何一次 confident=True 的檢索")
    hit_tool_call_limit: bool | None = None


class QueryResponse(BaseModel):
    """所有路徑同一種回傳。用不到的欄位留空,不會因為走的路不同而長得不一樣。"""
    answer: str
    route: RouteInfo
    sources: list[SourceChunk] = []
    related_cases: list[ViolationCase] = []
    verdict: Literal["low", "medium", "high"] | None = Field(default=None, description="只有審稿路徑有")
    matched_keywords: list[str] = []
    trace: list[TraceStep]
    usage: Usage
    meta: QueryMeta


# ============ /health ============

class FailedFile(BaseModel):
    file_name: str
    source_path: str
    file_ext: str | None
    page_count: int | None
    failure_reason: str
    inferred_law: str | None
    inferred_topic: str | None
    note: str | None


class IndexStats(BaseModel):
    total_chunks: int
    total_violations: int
    failed_files_count: int
    by_kind: dict
    by_category: dict
    by_primary_law: dict
    ocr_chunks: int
    chunks_with_tables: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    checks: dict = Field(description="每項檢查 true/false:index / embed_model / reranker / cases_index / openai_key")
    embed_device: str
    reranker_device: str
    reranker_fp16: bool
    gpu_available: bool
    model: str
    uptime_s: float
    index: IndexStats
    failed_files: list[FailedFile] = Field(default=[], description="無法處理的檔案清單(最多 200 筆)")


# ============ 資料端點 ============

class RelatedLaw(BaseModel):
    law_name: str
    article_full: str
    co_occurrence: int


class LawRelatedResponse(BaseModel):
    law: str
    total_chunks: int
    co_cited_laws: list[RelatedLaw]
