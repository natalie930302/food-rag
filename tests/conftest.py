"""測試共用的假物件。

核心檢索/agent 邏輯的測試不載入 BGE-M3、reranker、FAISS 或 OpenAI——這些
是 app/deps.py 的職責,已經在 eval/ 用真實模型驗證過。這裡只驗證「邏輯」:
信心閘門的判斷、候選池合併、drift 檢查的介入條件、agent harness 的邊界。
"""
from __future__ import annotations

from app.retrieval import RetrievedChunk


def make_chunk(chunk_id: int, text: str = "", score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, text=text or f"chunk-{chunk_id}", primary_law="食安法第28條",
        subtopic=None, document="doc", kind="qa", source_path="x", is_ocr=False,
        has_table=False, score=score,
    )


class FakeReranker:
    """用「文字 → 分數」對照表取代 cross-encoder;沒列到的給 0。"""

    def __init__(self, scores_by_text: dict[str, float]):
        self.scores_by_text = scores_by_text
        self.calls: list[list[list[str]]] = []

    def predict(self, pairs):
        self.calls.append(pairs)
        return [self.scores_by_text.get(doc, 0.0) for _, doc in pairs]
