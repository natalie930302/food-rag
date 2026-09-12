"""
實測校準 corrective_retrieval.py 用的信心閾值,不是憑空猜數字。

做法:對24題真實in-domain問題(eval_questions.json)跟6題明顯跟食品法規
無關的out-of-domain問題(adversarial_questions.json),各自跑一次dense
retrieval + cross-encoder rerank,記錄top-1的reranker分數分布。理想情況
是兩組分數有清楚的間隔(in-domain分數普遍高、out-of-domain分數普遍低),
閾值就選在間隔中間;如果沒有清楚間隔,也要誠實報告,不能硬掰一個閾值。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.deps import get_embed_model, get_faiss_chunks, get_db
from app.retrieval import retrieve_chunks
from sentence_transformers import CrossEncoder

with open(Path(__file__).parent / "eval_questions.json", encoding="utf-8") as f:
    in_domain_qs = json.load(f)
with open(Path(__file__).parent / "adversarial_questions.json", encoding="utf-8") as f:
    out_domain_qs = json.load(f)

print("載入模型與索引...")
model = get_embed_model()
index = get_faiss_chunks()
db = get_db()
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)


def top_scores(questions):
    scores = []
    for q in questions:
        candidates = retrieve_chunks(db, model, index, q["question"], filters=None, top_k=10)
        if not candidates:
            scores.append(None)
            continue
        pairs = [[q["question"], c.text] for c in candidates]
        s = reranker.predict(pairs)
        scores.append(float(max(s)))
    return scores


in_domain_scores = top_scores(in_domain_qs)
out_domain_scores = top_scores(out_domain_qs)

print("\n=== In-domain(24題真實食品法規問題)top-1 reranker分數 ===")
for q, s in zip(in_domain_qs, in_domain_scores):
    print(f"  {s:.4f}  {q['question']}")
print(f"\n  min={min(in_domain_scores):.4f}  max={max(in_domain_scores):.4f}  "
      f"avg={sum(in_domain_scores)/len(in_domain_scores):.4f}")

print("\n=== Out-of-domain(6題跟食品法規無關的問題)top-1 reranker分數 ===")
for q, s in zip(out_domain_qs, out_domain_scores):
    print(f"  {s:.4f}  {q['question']}")
print(f"\n  min={min(out_domain_scores):.4f}  max={max(out_domain_scores):.4f}  "
      f"avg={sum(out_domain_scores)/len(out_domain_scores):.4f}")

gap = min(in_domain_scores) - max(out_domain_scores)
print(f"\n=== 間隔分析 ===")
print(f"in-domain最低分: {min(in_domain_scores):.4f}")
print(f"out-of-domain最高分: {max(out_domain_scores):.4f}")
if gap > 0:
    suggested = (min(in_domain_scores) + max(out_domain_scores)) / 2
    print(f"兩組分數有清楚間隔(gap={gap:.4f}),建議閾值(取中點): {suggested:.4f}")
else:
    print(f"⚠️ 兩組分數有重疊(gap={gap:.4f} < 0),沒有一個閾值能完美切開兩組,"
          f"這是誠實的實測結果,不是bug——見README討論")

with open(Path(__file__).parent / "threshold_tuning_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "in_domain_scores": in_domain_scores,
        "out_domain_scores": out_domain_scores,
        "in_domain_min": min(in_domain_scores),
        "out_domain_max": max(out_domain_scores),
        "gap": gap,
    }, f, ensure_ascii=False, indent=2)
print("\n已儲存至 eval/threshold_tuning_results.json")
