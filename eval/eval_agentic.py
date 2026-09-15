"""
驗證 agentic query reformulation(app/agentic_retrieval.py:信心不足就用 LLM 改寫問題
重查一次)有沒有真的幫上忙。用兩組問題:
  1. 主評估集(eval_questions_v2.json / eval_questions.json):baseline 幾乎都有信心,
     預期 agentic 機制很少被觸發,用來確認「加了 agentic 機制不會誤傷原本答得好的問題」
  2. hard_questions.json(8 題刻意用更口語、跟法規原文用詞差距更大的問法設計)

誠實說明:confidently wrong 的題目(reranker 給高分但答錯),信心閘門偵測不到,
agentic retry 不會被觸發——這是 confidence-based 方法的已知限制,結果如實記錄。

用法:python eval/eval_agentic.py [--questions eval_questions.json]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agentic_retrieval import retrieve_agentic
from app.deps import get_db, get_embed_model, get_faiss_chunks, get_openai_client, get_reranker
from eval.common import HERE, load_questions, save_json
from eval.stats import bootstrap_ci, fmt_ci

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


def run_set(questions, name):
    print(f"\n=== {name} ===")
    records = []
    for q in questions:
        gold = q["gold_chunk_id"]
        result = retrieve_agentic(db, model, index, reranker, openai_client, q["question"])
        ids = [c.chunk_id for c in result.final_result.chunks] if result.final_result.confident else []
        hit = gold in ids
        rec = {
            "question": q["question"], "gold": gold, "source": q["source"],
            "hit": hit, "confident": result.final_result.confident,
            "used_retry": result.used_retry, "recovered_by_retry": result.used_retry and hit,
            "attempts": result.attempts,
        }
        records.append(rec)
        note = ""
        if result.used_retry:
            note = f"  [觸發retry → \"{result.attempts[1]['question']}\" confident={result.attempts[1]['confident']}]"
        if not hit or result.used_retry:
            print(f"hit={hit}  confident={result.final_result.confident}  [{q['source']}] {q['question']}{note}")

    n = len(records)
    point, lo, hi = bootstrap_ci([float(r["hit"]) for r in records])
    summary = {
        "correct": sum(r["hit"] for r in records), "total": n, "rate": point, "ci95": [lo, hi],
        "retried": sum(r["used_retry"] for r in records),
        "recovered": sum(r["recovered_by_retry"] for r in records),
        "confidently_wrong": sum(1 for r in records if r["confident"] and not r["hit"]),
    }
    print(f"\n{name}: {summary['correct']}/{n} 正確 = {fmt_ci(point, lo, hi)},"
          f" {summary['retried']} 題觸發 retry,{summary['recovered']} 題因 retry 救回,"
          f" {summary['confidently_wrong']} 題 confidently wrong")
    return summary, records


main_summary, main_records = run_set(main_qs, f"主評估集({len(main_qs)} 題)")
hard_summary, hard_records = run_set(hard_qs, f"hard_questions.json({len(hard_qs)} 題)")

save_json("results_agentic.json", {
    "main": {**main_summary, "per_question": main_records},
    "hard": {**hard_summary, "per_question": hard_records},
})
