"""
實測校準 corrective_retrieval.py 用的信心閾值,不是憑空猜數字。

做法:對 in-domain 評估題跟 out-of-domain 問題(adversarial_questions.json:
所得稅、Unity、貓咪疫苗這類明顯無關的,加上 2026/09 新增的幾題「近域」陷阱——
藥局成藥、化妝品標示、餐廳員工加班費、寵物飼料標示——刻意挑跟食品法規擦邊、
更容易被誤判有信心的問題),各自跑一次 dense retrieval + cross-encoder rerank,
記錄 top-1 的 reranker 分數分布。

理想情況是兩組分數有清楚間隔,閾值就選在間隔中間;如果沒有清楚間隔(評估集
擴大到 100 題後很可能發生),就誠實報告重疊範圍,並改用「總錯誤數最少」的
閾值——同時列出這個閾值下各自的誤判數,不硬掰一個看起來乾淨的數字。

用法:python eval/tune_confidence_threshold.py [--questions eval_questions.json]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import CrossEncoder

from app.deps import get_db, get_embed_model, get_faiss_chunks
from app.retrieval import retrieve_chunks
from eval.common import HERE, load_questions, save_json

ap = argparse.ArgumentParser()
ap.add_argument("--questions", default=None)
args = ap.parse_args()

in_domain_qs = load_questions(args.questions)
out_domain_qs = load_questions(HERE / "adversarial_questions.json")

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
            scores.append(0.0)
            continue
        s = reranker.predict([[q["question"], c.text] for c in candidates])
        scores.append(float(max(s)))
    return scores


in_scores = top_scores(in_domain_qs)
out_scores = top_scores(out_domain_qs)

print(f"\n=== In-domain({len(in_scores)} 題)top-1 reranker 分數 ===")
for q, s in sorted(zip(in_domain_qs, in_scores), key=lambda x: x[1])[:10]:
    print(f"  {s:.4f}  [{q['source']}] {q['question']}")
print(f"  ...(只列最低 10 題)  min={min(in_scores):.4f}  median={sorted(in_scores)[len(in_scores)//2]:.4f}")

print(f"\n=== Out-of-domain({len(out_scores)} 題)top-1 reranker 分數 ===")
for q, s in sorted(zip(out_domain_qs, out_scores), key=lambda x: -x[1]):
    print(f"  {s:.4f}  {q['question']}")

gap = min(in_scores) - max(out_scores)
print("\n=== 間隔分析 ===")
print(f"in-domain 最低分: {min(in_scores):.4f}   out-of-domain 最高分: {max(out_scores):.4f}   gap={gap:+.4f}")

# 掃所有候選閾值,找總錯誤數(in-domain 誤拒 + out-of-domain 誤放)最少的
candidates = sorted(set(in_scores) | set(out_scores))
best = None
for t in candidates:
    false_reject = sum(1 for s in in_scores if s < t)
    false_accept = sum(1 for s in out_scores if s >= t)
    total = false_reject + false_accept
    if best is None or total < best["errors"]:
        best = {"threshold": t, "errors": total, "false_reject": false_reject, "false_accept": false_accept}

if gap > 0:
    suggested = (min(in_scores) + max(out_scores)) / 2
    print(f"兩組分數有清楚間隔,建議閾值取中點: {suggested:.4f}(此時兩邊誤判都是 0)")
else:
    suggested = best["threshold"]
    print("⚠️ 兩組分數有重疊,沒有一個閾值能完美切開——這是誠實的實測結果,不是 bug。")
    print(f"總錯誤最少的閾值: {suggested:.4f} → in-domain 誤拒 {best['false_reject']}/{len(in_scores)},"
          f" out-of-domain 誤放 {best['false_accept']}/{len(out_scores)}")

save_json("threshold_tuning_results.json", {
    "n_in_domain": len(in_scores), "n_out_domain": len(out_scores),
    "in_domain_scores": in_scores, "out_domain_scores": out_scores,
    "in_domain_min": min(in_scores), "out_domain_max": max(out_scores), "gap": gap,
    "suggested_threshold": suggested, "min_error_threshold": best,
})
