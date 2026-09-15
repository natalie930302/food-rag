"""
評估 tool-calling agent(app/agent.py,含「字面錨定一致性檢查」與 temperature=0):
跟 eval_agentic.py 用同一組問題、同樣的 hit 定義(gold chunk 有沒有出現在最後
被拿去回答的 chunk 清單裡),才能跟固定重試的 baseline 做 paired 比較。

歷史脈絡(見 docs/RESEARCH_LOG.md):最早的 tool-calling agent 在 24 題上是 22/24,
比固定重試(24/24)差,診斷出 query drift 後加一致性檢查回到 24/24,再把決策
temperature 歸零後 hard set 從 6/8 到 7/8。這支腳本現在跑的是最終版本;
results_tool_agent.json 保留最早版本的數字不覆蓋,方便對照。

用法:python eval/eval_tool_agent_drift_check.py [--questions eval_questions.json]
     每題最多 4 次工具呼叫,100 題約 gpt-4o-mini 數十元台幣、延遲約 20-30 秒/題
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent import run_agent
from app.deps import get_db, get_embed_model, get_faiss_chunks, get_openai_client, get_reranker
from eval.common import HERE, load_questions, save_json
from eval.stats import bootstrap_ci, exact_sign_test, fmt_ci

ap = argparse.ArgumentParser()
ap.add_argument("--questions", default=None)
args = ap.parse_args()

main_qs = load_questions(args.questions)
hard_qs = load_questions(HERE / "hard_questions.json")

model = get_embed_model()
index = get_faiss_chunks()
db = get_db()
reranker = get_reranker()
openai_client = get_openai_client()


def last_confident_chunk_ids(trace) -> list[int]:
    for record in reversed(trace):
        if record.name == "search_regulations" and record.confident:
            return record.chunk_ids
    return []


def run_set(questions, name):
    print(f"\n=== {name} ===")
    records = []
    t_total = 0.0
    for q in questions:
        gold = q["gold_chunk_id"]
        t0 = time.time()
        result = run_agent(db, model, index, reranker, openai_client, q["question"])
        elapsed = time.time() - t0
        t_total += elapsed
        hit = gold in last_confident_chunk_ids(result.trace)
        drifted = any(r.query_drift_detected for r in result.trace)
        records.append({
            "question": q["question"], "gold": gold, "source": q["source"],
            "hit": hit, "grounded": result.grounded, "query_drift_detected": drifted,
            "tool_calls_used": result.tool_calls_used, "hit_tool_call_limit": result.hit_tool_call_limit,
            "tools": [r.name for r in result.trace], "elapsed_s": round(elapsed, 1),
        })
        print(f"hit={hit}  drift={drifted}  calls={result.tool_calls_used}  {elapsed:.1f}s  [{q['source']}] {q['question']}")

    n = len(records)
    point, lo, hi = bootstrap_ci([float(r["hit"]) for r in records])
    summary = {
        "correct": sum(r["hit"] for r in records), "total": n, "rate": point, "ci95": [lo, hi],
        "drift_detected_count": sum(r["query_drift_detected"] for r in records),
        "avg_tool_calls": round(sum(r["tool_calls_used"] for r in records) / n, 2),
        "multi_step_count": sum(1 for r in records if r["tool_calls_used"] > 1),
        "hit_tool_call_limit": sum(r["hit_tool_call_limit"] for r in records),
        "avg_latency_s": round(t_total / n, 1),
    }
    print(f"\n{name}: {summary['correct']}/{n} = {fmt_ci(point, lo, hi)},"
          f" {summary['drift_detected_count']} 題偵測到 drift 並介入,"
          f" {summary['multi_step_count']} 題用了 >1 次工具,平均 {summary['avg_latency_s']} 秒/題")
    return summary, records


main_summary, main_records = run_set(main_qs, f"主評估集({len(main_qs)} 題)")
hard_summary, hard_records = run_set(hard_qs, f"hard_questions.json({len(hard_qs)} 題)")

out = {
    "main": {**main_summary, "per_question": main_records},
    "hard": {**hard_summary, "per_question": hard_records},
}

# 跟固定重試 baseline(results_agentic.json)做 paired 比較
agentic_path = HERE / "results_agentic.json"
if agentic_path.exists():
    import json
    base = json.loads(agentic_path.read_text(encoding="utf-8"))
    print("\n=== vs. 固定重試 baseline(eval_agentic.py)===")
    for key, recs in (("main", main_records), ("hard", hard_records)):
        b = base.get(key, {}).get("per_question", [])
        if [x["gold"] for x in b] == [x["gold"] for x in recs]:
            st = exact_sign_test([x["hit"] for x in b], [x["hit"] for x in recs])
            print(f"  {key}: baseline {base[key]['correct']}/{base[key]['total']} → agent "
                  f"{out[key]['correct']}/{out[key]['total']};翻對 {st['n_plus']} / 翻錯 {st['n_minus']},"
                  f" 符號檢定 p={st['p_value']:.3f}")
            out[key]["vs_fixed_retry"] = {"baseline_correct": base[key]["correct"], **st}
        else:
            print(f"  {key}: 題目不一致,略過比較(請先重跑 eval_agentic.py)")

save_json("results_tool_agent_drift_check.json", out)
