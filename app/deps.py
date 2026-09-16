"""依賴注入:把昂貴的資源(model、index、db)在啟動時載入一次,後續共用。

FastAPI 透過 Depends() 取得這些單例。
"""
import sqlite3
from functools import lru_cache

import faiss
from openai import OpenAI
from sentence_transformers import CrossEncoder, SentenceTransformer

from config.settings import settings


@lru_cache(maxsize=1)
def get_embed_model() -> SentenceTransformer:
    """單例載入 BGE-M3。"""
    return SentenceTransformer(settings.embed_model, device=settings.embed_device)


@lru_cache(maxsize=1)
def get_faiss_chunks() -> faiss.Index:
    if not settings.faiss_chunks_path.exists():
        raise FileNotFoundError(
            f"找不到 {settings.faiss_chunks_path},請先跑 `make ingest`"
        )
    return faiss.read_index(str(settings.faiss_chunks_path))


@lru_cache(maxsize=1)
def get_faiss_cases() -> faiss.Index:
    if not settings.faiss_cases_path.exists():
        raise FileNotFoundError(
            f"找不到 {settings.faiss_cases_path},請先跑 `make ingest`"
        )
    return faiss.read_index(str(settings.faiss_cases_path))


def get_db() -> sqlite3.Connection:
    """每個請求一個 SQLite connection(SQLite 非執行緒安全)。"""
    if not settings.db_path.exists():
        raise FileNotFoundError(
            f"找不到 {settings.db_path},請先跑 `make ingest`"
        )
    conn = sqlite3.connect(str(settings.db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """單例載入 cross-encoder reranker,用於 corrective/agentic retrieval。

    2026/09 從 bge-reranker-base 換成 bge-reranker-v2-m3:
    早期以 32 題比較 rank-1 準確率從 0.688 提升到 0.875(該腳本已由 eval/eval_reranking.py 取代),
    見 eval/results_reranker_comparison.json、eval/results_rerank.json 與 README §1。
    """
    kwargs: dict = {}
    if settings.reranker_fp16:
        import torch
        kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
    return CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device=settings.reranker_device, **kwargs)


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY 未設定,請編輯 .env")
    return OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url or None)


def preload_all():
    """API 啟動時主動載入所有資源(暖機)。"""
    get_embed_model()
    get_faiss_chunks()
    get_faiss_cases()
    get_reranker()
    if settings.openai_api_key:
        get_openai_client()
