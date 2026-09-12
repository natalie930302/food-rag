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
