"""
驗證 tool-calling agent(app/agent.py)有沒有比既有的 agentic_retrieval.py
(固定「信心不足就重試一次」的確定性流程)更好——不是預設「LLM自主決策」一定
比較強,是實測比較兩者在同一組問題上的表現、呼叫次數、延遲。

跟 eval_agentic.py 用同一組問題(easy 24 題、hard 8 題),同一個 hit 定義:
gold_chunk_id 有沒有出現在最終拿去回答的 chunk 清單裡——這裡取的是 trace 裡
最後一次 confident=True 的 search_regulations 呼叫回傳的 chunk_ids,對應
agent 迴圈裡實際被當作依據使用的那批內容。
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.deps import get_embed_model, get_faiss_chunks, get_db, get_openai_client
from app.agent import run_agent
from app.agentic_retrieval import retrieve_agentic
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
    n_correct = 0
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
        n_correct += hit
        n_tool_calls += result.tool_calls_used
        n_hit_limit += result.hit_tool_call_limit

        print(
            f"hit={hit}  grounded={result.grounded}  tool_calls={result.tool_calls_used}  "
            f"{elapsed:.1f}s  {q['question']}"
        )
        per_question.append({
            "question": q["question"], "hit": hit, "grounded": result.grounded,
            "tool_calls_used": result.tool_calls_used, "elapsed_s": round(elapsed, 1),
        })

    n = len(questions)
    summary = {
        "correct": n_correct, "total": n,
        "avg_tool_calls": round(n_tool_calls / n, 2),
        "hit_tool_call_limit": n_hit_limit,
        "avg_latency_s": round(t_total / n, 1),
        "per_question": per_question,
    }
    print(f"\n{name}: {n_correct}/{n} 正確, 平均{summary['avg_tool_calls']}次工具呼叫, "
          f"{n_hit_limit}題撞到上限, 平均{summary['avg_latency_s']}秒/題")
    return summary


easy_summary = run_set(easy_qs, "eval_questions.json(24題)")
hard_summary = run_set(hard_qs, "hard_questions.json(8題)")

with open(Path(__file__).parent / "results_tool_agent.json", "w", encoding="utf-8") as f:
    json.dump({"easy": easy_summary, "hard": hard_summary}, f, ensure_ascii=False, indent=2)
print("\n已儲存至 eval/results_tool_agent.json")

print("\n=== 對照既有 results_agentic.json(固定重試一次) ===")
baseline_path = Path(__file__).parent / "results_agentic.json"
if baseline_path.exists():
    with open(baseline_path, encoding="utf-8") as f:
        baseline = json.load(f)
    print(f"固定重試        : easy {baseline['easy']['correct']}/{baseline['easy']['total']}, "
          f"hard {baseline['hard']['correct']}/{baseline['hard']['total']}")
    print(f"tool-calling agent: easy {easy_summary['correct']}/{easy_summary['total']}, "
          f"hard {hard_summary['correct']}/{hard_summary['total']}")
