"""
Corrective RAG(CRAG)概念的簡化實作,參考 Yan et al., "Corrective Retrieval
Augmented Generation"(arXiv:2401.15884, 2024)。

CRAG 原始做法是用一個獨立的輕量評分模型,把每筆檢索結果分成
Correct(直接用)/ Incorrect(丟棄,改用網路搜尋等其他來源補救)/
Ambiguous(混合使用)三類,而不是像傳統RAG一樣不管檢索品質好壞,
一律把top-k結果原封不動塞給LLM生成答案——這是近年RAG系統從「只顧
檢索有沒有東西」進化到「檢索到的東西到底可不可信」的關鍵轉變,
背後動機是：低品質的檢索結果會誤導LLM生成看似合理但錯誤的答案
(hallucination的一種常見成因)。

這裡簡化成:用現有的 cross-encoder reranker(eval/eval_reranking.py
已經驗證過能提升排序品質)的分數當作「評分模型」,分數低於閾值就判定
這批候選都不夠相關,回傳「沒有足夠可信的檢索結果」的訊號,而不是
硬塞低相關的內容給LLM。這是CRAG精神的簡化版,不是CRAG論文的完整
實作(沒有網路搜尋fallback、沒有knowledge refinement),但核心的
「檢索結果品質把關」邏輯是一致的。
"""
from dataclasses import dataclass

from app.retrieval import RetrievedChunk, retrieve_chunks

# 閾值是用 eval/tune_confidence_threshold.py 對24題真實in-domain問題跟6題
# out-of-domain問題的reranker分數分布實測校準出來的,不是隨便猜的數字:
# in-domain分數落在 0.994~1.000(24題全部),out-of-domain分數落在
# 0.0006~0.8056(6題),兩組有清楚間隔(gap=0.1884),取中點約0.90。
# 見 eval/threshold_tuning_results.json 完整數字。
CONFIDENCE_THRESHOLD = 0.90


@dataclass
class CorrectiveRetrievalResult:
    chunks: list[RetrievedChunk]
    confident: bool
    top_score: float | None


def retrieve_with_confidence_gate(
    db, embed_model, faiss_index, reranker, question: str,
    filters: dict | None = None, top_k: int = 5, candidate_n: int = 10,
) -> CorrectiveRetrievalResult:
    """先用dense retrieval撈候選,cross-encoder重排,分數太低就回傳「不確定」。"""
    candidates = retrieve_chunks(db, embed_model, faiss_index, question, filters, top_k=candidate_n)

    if not candidates:
        return CorrectiveRetrievalResult(chunks=[], confident=False, top_score=None)

    pairs = [[question, c.text] for c in candidates]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    top_score = float(reranked[0][1])

    if top_score < CONFIDENCE_THRESHOLD:
        return CorrectiveRetrievalResult(chunks=[], confident=False, top_score=top_score)

    top_chunks = [c for c, _ in reranked[:top_k]]
    return CorrectiveRetrievalResult(chunks=top_chunks, confident=True, top_score=top_score)
