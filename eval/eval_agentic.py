"""
驗證 agentic query reformulation 有沒有真的幫上忙。用兩組問題:
  1. eval_questions.json(24題,baseline已經24/24信心且答對,預期agentic機制
     不會被觸發,用來確認「加了agentic機制不會誤傷原本答得好的問題」)
  2. hard_questions.json(8題刻意用更口語、跟法規原文用詞差距更大的問法設計,
     baseline實測有2題表現不好:1題信心不足被拒答、1題confidently wrong)

誠實說明:confidently wrong的那一題,目前的信心閘門機制偵測不到(分數看起來
很高),所以agentic retry不會被觸發,這是confidence-based方法的已知限制,
不是這次改動的失敗,結果會如實記錄。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.deps import get_embed_model, get_faiss_chunks, get_db, get_openai_client
from app.agentic_retrieval import retrieve_agentic
from sentence_transformers import CrossEncoder

with open(Path(__file__).parent / "eval_questions.json", encoding="utf-8") as f:
    easy_qs = json.load(f)
with open(Path(__file__).parent / "hard_questions.json", encoding="utf-8") as f:
    hard_qs = [{**q, "gold_id": q["gold_chunk_id"]} for q in json.load(f)]
    for q in hard_qs:
        q["gold_id"] = q["gold_chunk_id"]

model = get_embed_model()
index = get_faiss_chunks()
db = get_db()
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
openai_client = get_openai_client()


def run_set(questions, name):
    print(f"\n=== {name} ===")
    n_correct, n_retried, n_recovered = 0, 0, 0
    for q in questions:
        gold = q.get("gold_id", q.get("gold_chunk_id"))
        result = retrieve_agentic(db, model, index, reranker, openai_client, q["question"])
        ids = [c.chunk_id for c in result.final_result.chunks] if result.final_result.confident else []
        hit = gold in ids
        n_correct += hit
        n_retried += result.used_retry
        if result.used_retry and hit:
            n_recovered += 1
        retry_note = ""
        if result.used_retry:
            retry_note = f"  [觸發retry: \"{result.attempts[1]['question']}\" confident={result.attempts[1]['confident']}]"
        print(f"hit={hit}  confident={result.final_result.confident}  {q['question']}{retry_note}")
    print(f"\n{name}: {n_correct}/{len(questions)} 正確, {n_retried} 題觸發retry, {n_recovered} 題因retry而救回")
    return {"correct": n_correct, "total": len(questions), "retried": n_retried, "recovered": n_recovered}


easy_summary = run_set(easy_qs, "eval_questions.json(24題,預期不觸發retry)")
hard_summary = run_set(hard_qs, "hard_questions.json(8題,預期部分觸發retry)")

with open(Path(__file__).parent / "results_agentic.json", "w", encoding="utf-8") as f:
    json.dump({"easy": easy_summary, "hard": hard_summary}, f, ensure_ascii=False, indent=2)
print("\n已儲存至 eval/results_agentic.json")
