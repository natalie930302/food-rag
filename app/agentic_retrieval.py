"""
把 corrective_retrieval.py 的「被動信心把關」升級成「主動採取行動」——這是
Corrective RAG 原論文比較完整的精神,也是「Agentic RAG」這個2025-2026最主流
研究方向的最小可行版本:系統不只是判斷檢索結果可不可信,信心不足時還會
自己決定下一步該怎麼做(這裡是:用LLM把問題換一種說法重新表述,再檢索一次),
而不是被動地直接回報「不知道」。

跟完整的 Agentic RAG 文獻(見 arXiv:2501.09136 Agentic RAG survey)比,這裡只做了
最基礎的一種 agent 行為(query reformulation + retry),沒有工具選擇、沒有
multi-agent協作、沒有iterative reasoning迴圈——是誠實的「最小可行版本」,
用來驗證這個方向本身有沒有用,不是宣稱做了完整的agentic系統。

已知限制(誠實記錄):信心分數是根據reranker對「檢索到的內容」打分,如果
retrieval找到了一個語意相近但實際上答錯的chunk,分數可能還是很高
(confidently wrong)——這種情況目前的機制偵測不到,retry不會被觸發。這是
confidence-based方法本身的限制,不是這次改動能解決的問題,在README裡有
記錄實測到的真實案例。
"""
from dataclasses import dataclass

from openai import OpenAI
from app.corrective_retrieval import retrieve_with_confidence_gate, CorrectiveRetrievalResult
from config.settings import settings

REFORMULATE_PROMPT = """你是食品法規檢索系統的輔助工具。以下使用者問題,用檢索系統目前的搜尋方式找不到足夠相關的結果。
請把這個問題換一種說法重新表述,保留原本的意圖,但改用更貼近政府公告/法規QA常見的正式用語跟句型。
只回傳改寫後的問題本身,不要加任何說明或標點符號以外的文字。

原問題:{question}
改寫後的問題:"""


@dataclass
class AgenticRetrievalResult:
    final_result: CorrectiveRetrievalResult
    attempts: list[dict]
    used_retry: bool


def reformulate_query(client: OpenAI, question: str) -> str:
    resp = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": REFORMULATE_PROMPT.format(question=question)}],
        temperature=0.3,
        max_tokens=100,
    )
    return resp.choices[0].message.content.strip()


def retrieve_agentic(
    db, embed_model, faiss_index, reranker, openai_client, question: str,
    filters: dict | None = None, top_k: int = 5, max_retries: int = 1,
) -> AgenticRetrievalResult:
    """先照原本的方式檢索,信心不夠的話,主動用LLM改寫問題重試。"""
    result = retrieve_with_confidence_gate(db, embed_model, faiss_index, reranker, question, filters, top_k)
    attempts = [{"question": question, "confident": result.confident, "top_score": result.top_score, "action": "initial_retrieval"}]

    if result.confident or max_retries <= 0:
        return AgenticRetrievalResult(final_result=result, attempts=attempts, used_retry=False)

    reformulated = reformulate_query(openai_client, question)
    retry_result = retrieve_with_confidence_gate(db, embed_model, faiss_index, reranker, reformulated, filters, top_k)
    attempts.append({"question": reformulated, "confident": retry_result.confident, "top_score": retry_result.top_score, "action": "reformulated_retry"})

    final = retry_result if retry_result.confident else result
    return AgenticRetrievalResult(final_result=final, attempts=attempts, used_retry=True)
