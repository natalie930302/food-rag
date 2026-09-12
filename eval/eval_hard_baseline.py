import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.deps import get_embed_model, get_faiss_chunks, get_db
from app.corrective_retrieval import retrieve_with_confidence_gate
from sentence_transformers import CrossEncoder

with open(Path(__file__).parent / "hard_questions.json", encoding="utf-8") as f:
    hard_qs = json.load(f)

model = get_embed_model()
index = get_faiss_chunks()
db = get_db()
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)

for q in hard_qs:
    result = retrieve_with_confidence_gate(db, model, index, reranker, q["question"])
    ids = [c.chunk_id for c in result.chunks] if result.confident else []
    hit = q["gold_chunk_id"] in ids
    print(f"confident={result.confident}  score={result.top_score:.4f}  hit={hit}  {q['question']}")
