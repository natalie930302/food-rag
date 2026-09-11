"""
量化檢索品質:food-rag 原本只有 API 層的 plumbing 測試(狀態碼、回傳格式),
沒有量測過「檢索有沒有真的撈到對的內容」。這支腳本直接呼叫 app.retrieval 裡
實際部署在用的 retrieve_chunks(),而不是另外寫一套簡化邏輯,量到的數字才真正
反映線上系統的檢索品質。

評估集(eval/eval_questions.json)是從真實已索引的17,152個chunks裡,挑24個跨
不同主題(標示、添加物登錄、檢驗週期、追溯系統、裁罰基準等)的QA內容,用
「改寫過的自然提問方式」而非直接複製原文——如果直接用索引裡的原文當查詢,
會因為文字重疊度過高而讓Recall虛高,測不出真正的語意檢索能力。

指標: Recall@1 / Recall@3 / Recall@5 / MRR,跟 civil-law-qa-bot 用同一套方法論。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.deps import get_embed_model, get_faiss_chunks, get_db
from app.retrieval import retrieve_chunks

with open(Path(__file__).parent / "eval_questions.json", encoding="utf-8") as f:
    eval_qs = json.load(f)

print("載入 BGE-M3 embedding 模型與 FAISS 索引...")
model = get_embed_model()
index = get_faiss_chunks()
db = get_db()

recall_at = {1: 0, 3: 0, 5: 0}
rr_sum = 0.0
n = len(eval_qs)
misses = []

for q in eval_qs:
    results = retrieve_chunks(db, model, index, q["question"], filters=None, top_k=10)
    ranked_ids = [r.chunk_id for r in results]
    gold = q["gold_chunk_id"]

    if gold in ranked_ids:
        rank_pos = ranked_ids.index(gold) + 1
        rr_sum += 1.0 / rank_pos
        for k in recall_at:
            if rank_pos <= k:
                recall_at[k] += 1
    else:
        misses.append(q["question"])

print(f"\n=== food-rag 檢索評估(n={n}, 全庫17,152個chunks,無metadata過濾) ===")
for k in recall_at:
    print(f"Recall@{k}: {recall_at[k]}/{n} = {recall_at[k]/n:.3f}")
print(f"MRR: {rr_sum/n:.3f}")

if misses:
    print(f"\n完全沒撈到gold chunk的問題({len(misses)}題):")
    for m in misses:
        print(f"  - {m}")

result = {
    "n": n,
    "recall@1": recall_at[1] / n,
    "recall@3": recall_at[3] / n,
    "recall@5": recall_at[5] / n,
    "mrr": rr_sum / n,
}
with open(Path(__file__).parent / "results.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print("\n已儲存至 eval/results.json")
