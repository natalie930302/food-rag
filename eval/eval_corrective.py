"""
驗證 app/corrective_retrieval.py 的信心閾值(0.90,eval/tune_confidence_threshold.py
實測校準)有沒有真的達到目的:
  1. 24題真實in-domain問題,加了信心閘門後,還能不能維持原本的檢索品質(不能
     因為加了把關機制反而誤傷本來答得對的問題)
  2. 6題跟食品法規無關的問題,信心閘門能不能正確判斷「這些不該自信回答」,
     而不是硬檢索出幾個分數最高但其實不相關的chunk硬充數
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.deps import get_embed_model, get_faiss_chunks, get_db
from app.corrective_retrieval import retrieve_with_confidence_gate
from sentence_transformers import CrossEncoder

with open(Path(__file__).parent / "eval_questions.json", encoding="utf-8") as f:
    in_domain_qs = json.load(f)
with open(Path(__file__).parent / "adversarial_questions.json", encoding="utf-8") as f:
    out_domain_qs = json.load(f)

print("載入模型與索引...")
model = get_embed_model()
index = get_faiss_chunks()
db = get_db()
reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512)

print("\n=== In-domain 問題(應該要 confident=True,且撈到正確 gold chunk) ===")
in_domain_correct_and_confident = 0
in_domain_wrongly_rejected = 0
for q in in_domain_qs:
    result = retrieve_with_confidence_gate(db, model, index, reranker, q["question"])
    if not result.confident:
        in_domain_wrongly_rejected += 1
        print(f"  ⚠️ 誤判為不確定 (score={result.top_score:.4f}): {q['question']}")
        continue
    ids = [c.chunk_id for c in result.chunks]
    if q["gold_chunk_id"] in ids:
        in_domain_correct_and_confident += 1

print(f"\n維持信心且答對: {in_domain_correct_and_confident}/{len(in_domain_qs)}")
print(f"誤判為不確定(false negative,不該發生): {in_domain_wrongly_rejected}/{len(in_domain_qs)}")

print("\n=== Out-of-domain 問題(應該要 confident=False,正確拒答) ===")
out_domain_correctly_rejected = 0
for q in out_domain_qs:
    result = retrieve_with_confidence_gate(db, model, index, reranker, q["question"])
    status = "✅ 正確拒答" if not result.confident else "❌ 誤判為有信心"
    score_str = f"{result.top_score:.4f}" if result.top_score is not None else "N/A"
    print(f"  {status} (score={score_str}): {q['question']}")
    if not result.confident:
        out_domain_correctly_rejected += 1

print(f"\n正確拒答: {out_domain_correctly_rejected}/{len(out_domain_qs)}")

summary = {
    "in_domain_correct_and_confident": in_domain_correct_and_confident,
    "in_domain_total": len(in_domain_qs),
    "in_domain_wrongly_rejected": in_domain_wrongly_rejected,
    "out_domain_correctly_rejected": out_domain_correctly_rejected,
    "out_domain_total": len(out_domain_qs),
}
with open(Path(__file__).parent / "results_corrective.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("\n已儲存至 eval/results_corrective.json")
