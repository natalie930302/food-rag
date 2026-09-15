"""
驗證「字面錨定一致性檢查」(app/agent_tools.py 的 query_drift_detected 機制)
有沒有真的解決 eval_tool_agent.py 診斷出的兩個「confidently wrong」新增失敗案例
(超商餐盒牛肉、食品添加物輸入登記),而且沒有誤傷原本就正確的22+6題。

跟 eval_tool_agent.py 是同一套流程、同一組問題,唯一差異是這次跑的是加了
一致性檢查之後的 app/agent.py——刻意留著 results_tool_agent.json(加之前的
結果)不覆蓋,才能做诚实的加之前/加之後對照,而不是只留最後一次的數字。
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.deps import get_embed_model, get_faiss_chunks, get_db, get_openai_client
from app.agent import run_agent
from sentence_transformers import CrossEncoder

with open(Path(__file__).parent / "eval_questions.json", encoding="utf-8") as f:
    easy_qs = json.load(f)
with open(Path(__file__).parent / "hard_questions.json", encoding="utf-8") as f:
    hard_qs = [{**q, "gold_id": q["gold_chunk_id"]} for q in json.load(f)]

model = get_embed_model()
index = get_faiss_chunks()
db = get_db()
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
openai_client = get_openai_client()


def last_confident_chunk_ids(trace) -> list[int]:
    for record in reversed(trace):
        if record.name == "search_regulations" and record.confident:
            return record.chunk_ids
    return []


def run_set(questions, name):
    print(f"\n=== {name} ===")
    n_correct, n_drift = 0, 0
    n_tool_calls, n_hit_limit = 0, 0
    t_total = 0.0
    per_question = []
    for q in questions:
        gold = q.get("gold_id", q.get("gold_chunk_id"))
        t0 = time.time()
        result = run_agent(db, model, index, reranker, openai_client, q["question"])
        elapsed = time.time() - t0
        t_total += elapsed

        ids = last_confident_chunk_ids(result.trace)
        hit = gold in ids
        drifted = any(r.query_drift_detected for r in result.trace)
        n_correct += hit
        n_drift += drifted
        n_tool_calls += result.tool_calls_used
        n_hit_limit += result.hit_tool_call_limit

        print(
            f"hit={hit}  drift_detected={drifted}  tool_calls={result.tool_calls_used}  "
            f"{elapsed:.1f}s  {q['question']}"
        )
        per_question.append({
            "question": q["question"], "hit": hit, "query_drift_detected": drifted,
            "tool_calls_used": result.tool_calls_used, "elapsed_s": round(elapsed, 1),
        })

    n = len(questions)
    summary = {
        "correct": n_correct, "total": n,
        "drift_detected_count": n_drift,
        "avg_tool_calls": round(n_tool_calls / n, 2),
        "hit_tool_call_limit": n_hit_limit,
        "avg_latency_s": round(t_total / n, 1),
        "per_question": per_question,
    }
    print(f"\n{name}: {n_correct}/{n} 正確, {n_drift}題偵測到query drift並介入, "
          f"平均{summary['avg_latency_s']}秒/題")
    return summary


easy_summary = run_set(easy_qs, "eval_questions.json(24題)")
hard_summary = run_set(hard_qs, "hard_questions.json(8題)")

with open(Path(__file__).parent / "results_tool_agent_drift_check.json", "w", encoding="utf-8") as f:
    json.dump({"easy": easy_summary, "hard": hard_summary}, f, ensure_ascii=False, indent=2)
print("\n已儲存至 eval/results_tool_agent_drift_check.json")

print("\n=== 對照 ===")
before_path = Path(__file__).parent / "results_tool_agent.json"
if before_path.exists():
    with open(before_path, encoding="utf-8") as f:
        before = json.load(f)
    print(f"加一致性檢查前: easy {before['easy']['correct']}/{before['easy']['total']}, "
          f"hard {before['hard']['correct']}/{before['hard']['total']}")
    print(f"加一致性檢查後: easy {easy_summary['correct']}/{easy_summary['total']}, "
          f"hard {hard_summary['correct']}/{hard_summary['total']}")
