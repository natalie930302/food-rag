"""API 請求與回應的 Pydantic schemas。"""
from typing import Literal

from pydantic import BaseModel, Field

# ============ 請求 ============

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="使用者問題")
    filters: dict | None = Field(
        default=None,
        description="可選 metadata 篩選,如 {'law_article':'食安法第28條','subtopic':'GHP'}",
    )
    top_k: int = Field(default=8, ge=1, le=20)
    include_cases: bool = Field(
        default=False, description="是否並行查詢相關違規案例")


class ReviewRequest(BaseModel):
    ad_text: str = Field(..., min_length=1, description="待審查的廣告文案")
    top_k: int = Field(default=5, ge=1, le=20)


class AgentAskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="使用者問題")
    max_tool_calls: int = Field(
        default=4, ge=1, le=8,
        description="agent 迴圈裡最多能呼叫幾次工具,防止 LLM 陷入重複查詢的迴圈",
    )


class QueryRequest(BaseModel):
    """單一入口:先路由再分派(見 app/router.py)。"""
    question: str = Field(..., min_length=1, description="使用者輸入:問題或一段待審文案")
    force_intent: Literal["regulation_qa", "case_lookup", "ad_review", "multi_hop"] | None = Field(
        default=None, description="跳過路由、直接指定走哪條(除錯/評估用)")
    use_llm_router: bool = Field(
        default=True, description="規則判不出來時是否用 LLM 分類;False 則一律走固定管線")
    top_k: int = Field(default=8, ge=1, le=20)
    max_tool_calls: int = Field(default=4, ge=1, le=8)


# ============ 回應 ============

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


class QueryMeta(BaseModel):
    retrieval_ms: int
    llm_ms: int
    model: str
    total_chunks_searched: int
    confident: bool | None = Field(
        default=None,
        description="Corrective RAG信心閘門判斷:檢索結果夠不夠可信。/ask才會有值,/review不使用這道機制",
    )
    used_retry: bool | None = Field(
        default=None,
        description="是否觸發了Agentic RAG的query reformulation重試",
    )
    unsupported_citations: list[str] | None = Field(
        default=None,
        description="答案層引用驗證:答案裡引用、但檢索內容裡沒出現的條號(已重生成一次仍未修正)。空清單代表全部有依據",
    )
    citation_regenerated: bool | None = Field(
        default=None,
        description="答案是否因為引用驗證失敗而重新生成過一次",
    )


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    related_cases: list[ViolationCase] = []
    meta: QueryMeta


class ReviewEvidence(BaseModel):
    matched_keywords: list[str]
    laws: list[SourceChunk]
    cases: list[ViolationCase]


class ReviewResponse(BaseModel):
    verdict: Literal["low", "medium", "high"]
    answer: str
    evidence: ReviewEvidence
    meta: QueryMeta


class AgentToolCall(BaseModel):
    name: str
    arguments: dict
    confident: bool | None
    top_score: float | None
    chunk_ids: list[int]
    result_summary: str
    query_drift_detected: bool = Field(
        default=False,
        description="LLM改寫的查詢跟原始問題字面文字檢索結果top-1不一致,已改用字面文字結果",
    )


class AgentAskResponse(BaseModel):
    answer: str
    trace: list[AgentToolCall] = Field(
        description="這次回答呼叫了哪些工具、每次呼叫的信心分數,供除錯與評估用",
    )
    meta: QueryMeta
    tool_calls_used: int
    grounded: bool = Field(
        description="迴圈裡有沒有任何一次工具呼叫回傳confident=True。"
                     "false代表answer是程式碼強制覆寫的誠實拒答訊息,不是LLM亂答",
    )
    hit_tool_call_limit: bool
    usage: dict = Field(
        default_factory=dict,
        description="預算使用量:tool_calls / llm_calls / prompt_tokens / completion_tokens / elapsed_s / stop_reason",
    )


class FailedFile(BaseModel):
    file_name: str
    source_path: str
    file_ext: str | None
    page_count: int | None
    failure_reason: str
    inferred_law: str | None
    inferred_topic: str | None
    note: str | None


class StatsResponse(BaseModel):
    total_chunks: int
    total_violations: int
    failed_files_count: int
    by_kind: dict
    by_category: dict
    by_primary_law: dict
    ocr_chunks: int
    chunks_with_tables: int


class RelatedLaw(BaseModel):
    law_name: str
    article_full: str
    co_occurrence: int


class LawRelatedResponse(BaseModel):
    law: str
    total_chunks: int
    co_cited_laws: list[RelatedLaw]


class RouteInfo(BaseModel):
    intent: Literal["regulation_qa", "case_lookup", "ad_review", "multi_hop"]
    source: Literal["rules", "llm", "fallback", "forced"] = Field(
        description="rules=關鍵字規則零成本判定;llm=規則判不出來、問了一次 gpt-4o-mini;fallback=LLM 回傳無效或未啟用")
    reason: str = ""
    handler: str = Field(description="實際分派到的端點邏輯:/ask、/review 或 /ask_agent")
    router_ms: int


class QueryResponse(BaseModel):
    """/query 的統一回傳:三種 handler 的欄位聯集,用不到的留空。"""
    answer: str
    route: RouteInfo
    sources: list[SourceChunk] = []
    related_cases: list[ViolationCase] = []
    verdict: Literal["low", "medium", "high"] | None = Field(default=None, description="只有 ad_review 有")
    matched_keywords: list[str] = []
    trace: list[AgentToolCall] = Field(default=[], description="只有 multi_hop(agent)有")
    tool_calls_used: int | None = None
    grounded: bool | None = None
    usage: dict = {}
    meta: QueryMeta
