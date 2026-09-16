"""
從 /query 端到端評估:統一入口後,harness 只有一套,評估也只要一支。

直接呼叫 app.main.query()(不起 HTTP server),對三組有標籤的題目跑完整流程——
路由 → 執行 → harness(trace / usage / 拒答 / 引用驗證)——然後依「實際走到的路由」分組報告:

  eval_questions_v2.json  108 題單跳(有 gold chunk → 量 hit)
  hard_questions.json       8 題(有 gold)
  multihop_questions.json  20 題(量 reg_hit,案例/關聯法條在 eval_multihop.py 已量過)
  router_questions.json    30 題(審稿/案例/邊界;沒有 gold,只量路由、拒答率、引用驗證)
  adversarial_questions.json 20 題 OOD(應拒答)

每題記錄:route、handler、hit(有 gold 時)、refused、unsupported_citations、usage、延遲。
成本:約 186 題,大多走固定管線(1 次 LLM),multi-hop 走 agent(2~4 次);gpt-4o-mini 十幾元台幣。

用法:python eval/eval_query_e2e.py [--limit N]
"""
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import deps
from app.main import query
from app.schemas import QueryRequest
from eval.common import HERE, load_questions, save_json
from eval.stats import bootstrap_ci, fmt_ci

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=None, help="每組最多取幾題(快速煙霧測試用)")
args = ap.parse_args()

sets = {
    "single-hop": load_questions(HERE / "eval_questions_v2.json"),
    "hard": load_questions(HERE / "hard_questions.json"),
    "multi-hop": load_questions(HERE / "multihop_questions.json"),
    "router-set": load_questions(HERE / "router_questions.json"),
    "ood": load_questions(HERE / "adversarial_questions.json"),
}
if args.limit:
    sets = {k: v[: args.limit] for k, v in sets.items()}

deps.preload_all()
records = []
for set_name, qs in sets.items():
    print(f"\n=== {set_name}({len(qs)} 題)===")
    for q in qs:
        t0 = time.time()
        r = query(QueryRequest(question=q["question"]), None)
        elapsed = time.time() - t0
        src_ids = [c.chunk_id for c in r.sources]
        gold = q.get("gold_chunk_id")
        hit = (gold in src_ids) if gold is not None else None
        if set_name == "multi-hop":                      # 多步題:法規 chunk 的 primary_law 有沒有對到
            laws = {c.primary_law for c in r.sources if c.primary_law}
            hit = bool(set(q["gold_laws"]) & laws)
        rec = {
            "set": set_name, "question": q["question"], "expected_intent": q.get("intent"),
            "intent": r.route.intent, "route_source": r.route.source, "handler": r.route.handler,
            "hit": hit, "refused": r.meta.refused, "confident": r.meta.confident,
            "unsupported_citations": r.meta.unsupported_citations, "citation_regenerated": r.meta.citation_regenerated,
            "verdict": r.verdict, "trace": [s.name for s in r.trace],
            "usage": r.usage.model_dump(), "elapsed_s": round(elapsed, 1),
        }
        records.append(rec)
        flag = "" if hit is None else ("✓" if hit else "✗")
        print(f"{flag:<2}{r.route.intent:<14}{r.route.source:<6}refused={str(r.meta.refused):<6}"
              f"{r.usage.llm_calls}llm {r.usage.total_tokens:>5}tok {elapsed:5.1f}s  {q['question'][:38]}")

# ---- 彙整:依題組、依實際路由 ----------------------------------------------------
def summarize(rows):
    hits = [float(x["hit"]) for x in rows if x["hit"] is not None]
    out = {
        "n": len(rows),
        "refused": sum(x["refused"] for x in rows),
        "unsupported_citation_cases": sum(1 for x in rows if x["unsupported_citations"]),
        "citation_regenerated": sum(x["citation_regenerated"] for x in rows),
        "avg_llm_calls": round(sum(x["usage"]["llm_calls"] for x in rows) / len(rows), 2),
        "avg_tokens": round(sum(x["usage"]["total_tokens"] for x in rows) / len(rows)),
        "avg_elapsed_s": round(sum(x["elapsed_s"] for x in rows) / len(rows), 1),
    }
    if hits:
        p, lo, hi = bootstrap_ci(hits)
        out["hit"] = {"count": int(sum(hits)), "total": len(hits), "rate": p, "ci95": [lo, hi]}
    return out


by_set = {k: summarize([x for x in records if x["set"] == k]) for k in sets}
by_handler = defaultdict(list)
for x in records:
    by_handler[x["handler"]].append(x)
by_handler = {k: summarize(v) for k, v in by_handler.items()}
routing_acc = [float(x["intent"] == x["expected_intent"]) for x in records if x["expected_intent"]]

print("\n=== 依題組 ===")
print(f"{'題組':<12}{'n':>4}  {'hit':<28}{'拒答':>5}{'引用未支持':>10}{'LLM/題':>7}{'tok/題':>7}{'秒/題':>6}")
for k, s in by_set.items():
    h = f"{s['hit']['count']}/{s['hit']['total']} = {fmt_ci(s['hit']['rate'], *s['hit']['ci95'])}" if "hit" in s else "—"
    print(f"{k:<12}{s['n']:>4}  {h:<28}{s['refused']:>5}{s['unsupported_citation_cases']:>10}{s['avg_llm_calls']:>7}{s['avg_tokens']:>7}{s['avg_elapsed_s']:>6}")
print("\n=== 依實際路由(handler)===")
for k, s in by_handler.items():
    h = f"{s['hit']['count']}/{s['hit']['total']}" if "hit" in s else "—"
    print(f"{k:<12}{s['n']:>4}  hit {h:<10} 拒答 {s['refused']:<4} 引用未支持 {s['unsupported_citation_cases']:<3} "
          f"LLM {s['avg_llm_calls']}/題  {s['avg_tokens']} tok/題  {s['avg_elapsed_s']} s/題")
if routing_acc:
    print(f"\n路由準確率(有標籤的 {len(routing_acc)} 題):{sum(routing_acc)/len(routing_acc):.3f}")
ood = by_set["ood"]
print(f"OOD 拒答:{ood['refused']}/{ood['n']}")

save_json("results_query_e2e.json", {
    "by_set": by_set, "by_handler": by_handler,
    "routing_accuracy": (sum(routing_acc) / len(routing_acc)) if routing_acc else None,
    "per_question": records,
})
