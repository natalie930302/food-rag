"""複合問題(一句多問)端到端評估。

2026/09 使用者實測發現:108 題單跳、20 題多跳評估集全是「一句一問」的乾淨題目,
真人一口氣問三件事(「減肥廣告違反哪一條?罰多少?有沒有實際案例?」)時,
固定路徑整句送檢索分數掉到門檻下、agent 濃縮成關鍵字也查不到、查到案例卻被 grounded 規則整題丟掉。
這支腳本量的就是這件事:16 題複合問題,每題拆成 2–3 個「部分」,每個部分有自己的判定條件:

  must_any  答案要出現其中任一字串(條號、金額、關鍵詞)
  case      答案要提到實際案例(年月 + 金額/公司)
  none_ok   這一部分資料庫可能真的沒有,誠實寫「沒有找到」也算過

指標:full_hit(所有部分都過)、parts_hit(部分層級的命中率)、整題拒答數、各路由分佈。
不呼叫 HTTP,直接進 app.main.query(),跟 eval_query_e2e.py 同一條路。

用法:python eval/eval_compound.py [--tag before|after] [--out results_compound.json] [--limit N]
"""
import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import deps
from app.main import query
from app.schemas import QueryRequest
from eval.common import HERE, load_questions, save_json
from eval.stats import bootstrap_ci, fmt_ci

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="after")
ap.add_argument("--out", default="results_compound.json")
ap.add_argument("--limit", type=int, default=None)
args = ap.parse_args()

_CASE_MONEY = re.compile(r"\d[\d,]*\s*(元|萬)")
_CASE_WHO = re.compile(r"(\d+\s*年\s*\d+\s*月|\d+年|公司|有限|商行|企業|OO|生技|股份)")
_NONE = ("沒有找到", "未涵蓋", "查無", "沒有相關", "未找到", "無相關")


def part_ok(part: dict, answer: str) -> bool:
    if part.get("none_ok") and any(w in answer for w in _NONE):
        return True
    if part.get("case"):
        return bool(_CASE_MONEY.search(answer) and _CASE_WHO.search(answer))
    return any(w in answer for w in part["must_any"])


qs = load_questions(HERE / "compound_questions.json")
if args.limit:
    qs = qs[: args.limit]

deps.preload_all()
records = []
for q in qs:
    t0 = time.time()
    r = query(QueryRequest(question=q["question"]), None)
    elapsed = time.time() - t0
    parts = [{"name": p["name"], "ok": part_ok(p, r.answer)} for p in q["parts"]]
    refused = bool(r.meta.refused)
    full = (not refused) and all(p["ok"] for p in parts)
    rec = {
        "question": q["question"], "route": r.route.intent, "route_source": r.route.source,
        "refused": refused, "parts": parts, "full_hit": full,
        "trace": [s.name for s in r.trace], "llm_calls": r.usage.llm_calls, "elapsed_s": round(elapsed, 1),
        "answer_preview": r.answer[:300],
    }
    records.append(rec)
    marks = " ".join(("✓" if p["ok"] else "✗") + p["name"] for p in parts)
    print(f"{'✅' if full else '❌'} [{r.route.intent}/{r.route.source}] {q['question'][:34]}  {marks}"
          f"{'  (整題拒答)' if refused else ''}  {elapsed:.0f}s")

n = len(records)
full_hits = [float(x["full_hit"]) for x in records]
p, lo, hi = bootstrap_ci(full_hits)
part_flags = [float(pp["ok"]) for x in records for pp in x["parts"]]
pp_, plo, phi = bootstrap_ci(part_flags)
routes = {}
for x in records:
    routes[x["route"]] = routes.get(x["route"], 0) + 1
summary = {
    "tag": args.tag, "n": n,
    "full_hit": {"count": int(sum(full_hits)), "total": n, "rate": p, "ci95": [lo, hi]},
    "parts_hit": {"count": int(sum(part_flags)), "total": len(part_flags), "rate": pp_, "ci95": [plo, phi]},
    "refused": int(sum(x["refused"] for x in records)),
    "routes": routes,
    "avg_llm_calls": round(sum(x["llm_calls"] for x in records) / n, 2),
    "avg_latency_s": round(sum(x["elapsed_s"] for x in records) / n, 1),
}
print(f"\n=== 複合問題 {n} 題({args.tag})===")
print(f"full_hit  {summary['full_hit']['count']}/{n} = {fmt_ci(p, lo, hi)}")
print(f"parts_hit {summary['parts_hit']['count']}/{len(part_flags)} = {fmt_ci(pp_, plo, phi)}")
print(f"整題拒答 {summary['refused']}/{n};路由 {routes};平均 LLM 呼叫 {summary['avg_llm_calls']};{summary['avg_latency_s']} 秒/題")
save_json(args.out, {"summary": summary, "records": records})
print(f"已儲存至 eval/{args.out}")
