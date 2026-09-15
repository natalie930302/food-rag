"""
量化檢索品質:food-rag 原本只有 API 層的 plumbing 測試(狀態碼、回傳格式),
沒有量測過「檢索有沒有真的撈到對的內容」。這支腳本直接呼叫 app.retrieval 裡
實際部署在用的 retrieve_chunks(),而不是另外寫一套簡化邏輯,量到的數字才真正
反映線上系統的檢索品質。

評估集:
  - eval_questions.json:24 題手寫,從真實已索引的 chunks 裡挑跨主題的內容,用
    「改寫過的自然提問方式」而非直接複製原文(避免文字重疊讓 Recall 虛高)
  - eval_questions_v2.json(2026/09 擴大版):上面 24 題 + LLM 生成、自動過濾、
    人工抽查的合成題,每題標 source=manual/synthetic,報告分開列

指標:Recall@1 / Recall@3 / Recall@5 / MRR,每個都附 95% bootstrap CI(eval/stats.py)。

用法:python eval/eval_retrieval.py [--questions eval_questions.json]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.deps import get_db, get_embed_model, get_faiss_chunks
from app.retrieval import retrieve_chunks
from eval.common import aggregate_by_source, load_questions, print_table, rank_of, save_json

ap = argparse.ArgumentParser()
ap.add_argument("--questions", default=None)
args = ap.parse_args()

eval_qs = load_questions(args.questions)

print("載入 BGE-M3 embedding 模型與 FAISS 索引...")
model = get_embed_model()
index = get_faiss_chunks()
db = get_db()

records = []
for q in eval_qs:
    results = retrieve_chunks(db, model, index, q["question"], filters=None, top_k=10)
    rank = rank_of([r.chunk_id for r in results], q["gold_chunk_id"])
    records.append({"question": q["question"], "gold": q["gold_chunk_id"], "source": q["source"], "rank": rank})

by_source = aggregate_by_source(records)
print_table(f"Baseline:BGE-M3 dense retrieval,無 rerank(n={len(records)},全庫 {index.ntotal} chunks)", by_source)

misses = [r["question"] for r in records if r["rank"] is None]
if misses:
    print(f"\n前 10 名完全沒撈到 gold chunk 的問題({len(misses)} 題):")
    for m in misses:
        print(f"  - {m}")

save_json("results.json", {"config": "dense_only", "summary": by_source, "per_question": records})
