"""
驗證 app/corrective_retrieval.py 的信心閾值(tune_confidence_threshold.py 實測校準)
有沒有真的達到目的:
  1. in-domain 問題,加了信心閘門後,還能不能維持原本的檢索品質(不能因為加了
     把關機制反而誤傷本來答得對的問題)
  2. out-of-domain 問題,信心閘門能不能正確判斷「這些不該自信回答」,而不是硬
     檢索出幾個分數最高但其實不相關的 chunk 硬充數

用法:python eval/eval_corrective.py [--questions eval_questions.json]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.corrective_retrieval import CONFIDENCE_THRESHOLD, retrieve_with_confidence_gate
from app.deps import get_db, get_embed_model, get_faiss_chunks, get_reranker
from eval.common import HERE, load_questions, save_json
from eval.stats import bootstrap_ci, fmt_ci

ap = argparse.ArgumentParser()
ap.add_argument("--questions", default=None)
args = ap.parse_args()

in_domain_qs = load_questions(args.questions)
out_domain_qs = load_questions(HERE / "adversarial_questions.json")

print("載入模型與索引...")
model = get_embed_model()
index = get_faiss_chunks()
db = get_db()
reranker = get_reranker()
print(f"信心閾值 = {CONFIDENCE_THRESHOLD}")

print("\n=== In-domain(應該要 confident=True,且撈到正確 gold chunk)===")
in_records = []
for q in in_domain_qs:
    r = retrieve_with_confidence_gate(db, model, index, reranker, q["question"])
    ids = [c.chunk_id for c in r.chunks]
    rec = {
        "question": q["question"], "gold": q["gold_chunk_id"], "source": q["source"],
        "confident": r.confident, "top_score": r.top_score,
        "hit": r.confident and q["gold_chunk_id"] in ids,
    }
    in_records.append(rec)
    if not r.confident:
        print(f"  ⚠️ 誤判為不確定 (score={r.top_score:.4f}) [{q['source']}] {q['question']}")
    elif not rec["hit"]:
        print(f"  ❌ 有信心但沒撈到 gold (score={r.top_score:.4f}) [{q['source']}] {q['question']}")

print("\n=== Out-of-domain(應該要 confident=False,正確拒答)===")
out_records = []
for q in out_domain_qs:
    r = retrieve_with_confidence_gate(db, model, index, reranker, q["question"])
    score_str = f"{r.top_score:.4f}" if r.top_score is not None else "N/A"
    status = "✅ 正確拒答" if not r.confident else "❌ 誤判為有信心"
    print(f"  {status} (score={score_str}): {q['question']}")
    out_records.append({"question": q["question"], "confident": r.confident, "top_score": r.top_score,
                        "correctly_rejected": not r.confident})


def summarize(records, key):
    vals = [float(r[key]) for r in records]
    point, lo, hi = bootstrap_ci(vals)
    return {"count": int(sum(vals)), "total": len(vals), "rate": point, "ci95": [lo, hi]}


summary = {
    "threshold": CONFIDENCE_THRESHOLD,
    "in_domain": {
        "all": summarize(in_records, "hit"),
        **{src: summarize([r for r in in_records if r["source"] == src], "hit")
           for src in sorted({r["source"] for r in in_records})},
        "wrongly_rejected": sum(1 for r in in_records if not r["confident"]),
        "confident_but_miss": sum(1 for r in in_records if r["confident"] and not r["hit"]),
    },
    "out_domain": summarize(out_records, "correctly_rejected"),
}

print("\n=== 總結 ===")
for grp, s in summary["in_domain"].items():
    if isinstance(s, dict):
        print(f"  in-domain[{grp}] 維持信心且答對: {s['count']}/{s['total']} = {fmt_ci(s['rate'], *s['ci95'])}")
print(f"  in-domain 誤拒(false negative): {summary['in_domain']['wrongly_rejected']}/{len(in_records)}")
print(f"  in-domain 有信心但答錯(confidently wrong,閘門偵測不到): "
      f"{summary['in_domain']['confident_but_miss']}/{len(in_records)}")
o = summary["out_domain"]
print(f"  out-of-domain 正確拒答: {o['count']}/{o['total']} = {fmt_ci(o['rate'], *o['ci95'])}")

save_json("results_corrective.json", {"summary": summary, "in_domain": in_records, "out_domain": out_records})
