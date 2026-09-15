"""
路由器(app/router.py)的分類準確率。router 是 /query 新增的失效點,所以要單獨量。

標籤集(不用另外標,現成的就有):
  - eval_questions_v2.json  108 題 → regulation_qa
  - hard_questions.json       8 題 → regulation_qa
  - multihop_questions.json  20 題 → multi_hop
  - router_questions.json    30 題手寫:ad_review / case_lookup / 額外的 regulation_qa、multi_hop

報告三層:
  1. 整體與各類別準確率、混淆矩陣
  2. 規則層覆蓋率(多少題不用問 LLM)與規則層準確率;LLM 層準確率(只算規則判不出來的)
  3. 代價不對稱的錯誤分開列:multi_hop 被判成單跳 = 會漏掉案例/關聯法條(有害);
     單跳被判成 multi_hop = 只是變慢、變貴(無害但浪費)

用法:python eval/eval_router.py [--no-llm]
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.deps import get_openai_client
from app.router import INTENTS, route
from eval.common import HERE, load_questions, save_json
from eval.stats import bootstrap_ci, fmt_ci

ap = argparse.ArgumentParser()
ap.add_argument("--no-llm", action="store_true", help="只量規則層")
args = ap.parse_args()

labelled = []
for q in load_questions(HERE / "eval_questions_v2.json") + load_questions(HERE / "hard_questions.json"):
    labelled.append({"question": q["question"], "intent": "regulation_qa", "set": "single-hop"})
for q in load_questions(HERE / "multihop_questions.json"):
    labelled.append({"question": q["question"], "intent": "multi_hop", "set": "multi-hop"})
for q in load_questions(HERE / "router_questions.json"):
    labelled.append({"question": q["question"], "intent": q["intent"], "set": "router-set"})

client = None if args.no_llm else get_openai_client()

records = []
for item in labelled:
    d = route(item["question"], client=client, use_llm=not args.no_llm)
    records.append({**item, "predicted": d.intent, "source": d.source, "reason": d.reason,
                    "correct": d.intent == item["intent"]})

n = len(records)
acc, lo, hi = bootstrap_ci([float(r["correct"]) for r in records])
print(f"=== 路由準確率(n={n})=== {sum(r['correct'] for r in records)}/{n} = {fmt_ci(acc, lo, hi)}")

confusion = defaultdict(Counter)
for r in records:
    confusion[r["intent"]][r["predicted"]] += 1
header = "gold / pred"
print(f"\n{header:<16}" + "".join(f"{p:<16}" for p in INTENTS) + "recall")
per_intent = {}
for g in INTENTS:
    row = confusion[g]
    total = sum(row.values())
    rec = row[g] / total if total else float("nan")
    per_intent[g] = {"n": total, "recall": rec}
    print(f"{g:<16}" + "".join(f"{row[p]:<16}" for p in INTENTS) + (f"{rec:.3f}" if total else "—"))

by_source = defaultdict(list)
for r in records:
    by_source[r["source"]].append(r["correct"])
print("\n=== 分層 ===")
for src, vals in by_source.items():
    print(f"  {src:<9} {len(vals):>4} 題({len(vals)/n:.0%}) 準確率 {sum(vals)}/{len(vals)} = {sum(vals)/len(vals):.3f}")

harmful = [r for r in records if r["intent"] == "multi_hop" and r["predicted"] != "multi_hop"]
wasteful = [r for r in records if r["intent"] != "multi_hop" and r["predicted"] == "multi_hop"]
print(f"\n=== 代價不對稱的錯誤 ===\n  有害(multi_hop → 單跳,會漏案例/關聯法條): {len(harmful)}")
for r in harmful:
    print(f"    → {r['predicted']:<14}[{r['source']}] {r['question']}")
print(f"  浪費(單跳/審稿 → multi_hop,只是變慢): {len(wasteful)}")
for r in wasteful:
    print(f"    → [{r['source']}] {r['question']}")
others = [r for r in records if not r["correct"] and r not in harmful and r not in wasteful]
if others:
    print("  其他錯誤:")
    for r in others:
        print(f"    {r['intent']} → {r['predicted']:<14}[{r['source']}] {r['question']}")

save_json("results_router.json", {
    "n": n, "accuracy": {"point": acc, "ci95": [lo, hi]},
    "per_intent": per_intent,
    "confusion": {g: dict(confusion[g]) for g in INTENTS},
    "by_source": {s: {"n": len(v), "accuracy": sum(v) / len(v)} for s, v in by_source.items()},
    "harmful_errors": len(harmful), "wasteful_errors": len(wasteful),
    "per_question": records,
})
