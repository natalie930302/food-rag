"""
在 baseline(BGE-M3 dense retrieval)之上加一層 cross-encoder reranker,是近年
production RAG 系統的標準做法(dense retrieval 先撈出候選,rerank 再精排)——
單純的 bi-encoder(把 query 跟文件各自獨立編碼再比 cosine similarity)沒辦法
讓 query 和文件的每個 token 互相注意力,cross-encoder 把 query+文件當同一個輸入
一起編碼,排序精度通常更好,代價是不能像 bi-encoder 一樣預先把全庫向量算好,
只能對「已經被初篩出來的少量候選」做,所以標準做法是兩階段:dense retrieval
先撈 top-N 候選(快、可預先索引全庫),再用 cross-encoder 對這N個候選重新排序
(慢、只做在候選集上)。

用的 reranker 是 BAAI/bge-reranker-v2-m3(2026/09 從 bge-reranker-base 換過來——
compare_reranker_models.py 實測 32 題 rank-1 準確率從 0.688 提升到 0.875,是全面性
的改善,不是單一案例湊巧,詳見 README「reranker 模型升級」章節)。

baseline的Recall@5已經到1.000(24題全部都在top5),reranking能改善的空間主要在
Recall@1跟MRR——看cross-encoder精排能不能把原本排在第2-5名的正確答案挪到第1名。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.deps import get_embed_model, get_faiss_chunks, get_db
from app.retrieval import retrieve_chunks
from sentence_transformers import CrossEncoder

with open(Path(__file__).parent / "eval_questions.json", encoding="utf-8") as f:
    eval_qs = json.load(f)

print("載入 BGE-M3 embedding 模型、FAISS 索引、bge-reranker-v2-m3...")
model = get_embed_model()
index = get_faiss_chunks()
db = get_db()
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)

TOP_N_CANDIDATES = 10  # dense retrieval先撈的候選數,reranker只對這些重排


def evaluate(use_rerank: bool, name: str):
    recall_at = {1: 0, 3: 0, 5: 0}
    rr_sum = 0.0
    n = len(eval_qs)

    for q in eval_qs:
        results = retrieve_chunks(db, model, index, q["question"], filters=None, top_k=TOP_N_CANDIDATES)
        gold = q["gold_chunk_id"]

        if use_rerank and results:
            pairs = [[q["question"], r.text] for r in results]
            scores = reranker.predict(pairs)
            reranked = sorted(zip(results, scores), key=lambda x: -x[1])
            ranked_ids = [r.chunk_id for r, _ in reranked]
        else:
            ranked_ids = [r.chunk_id for r in results]

        if gold in ranked_ids:
            rank_pos = ranked_ids.index(gold) + 1
            rr_sum += 1.0 / rank_pos
            for k in recall_at:
                if rank_pos <= k:
                    recall_at[k] += 1

    print(f"\n=== {name} ===")
    for k in recall_at:
        print(f"Recall@{k}: {recall_at[k]}/{n} = {recall_at[k]/n:.3f}")
    print(f"MRR: {rr_sum/n:.3f}")
    return {"recall@1": recall_at[1]/n, "recall@3": recall_at[3]/n, "recall@5": recall_at[5]/n, "mrr": rr_sum/n}


baseline_result = evaluate(use_rerank=False, name="Baseline(僅dense retrieval,無rerank)")
rerank_result = evaluate(use_rerank=True, name="加上 cross-encoder rerank 後")

print("\n=== 比較 ===")
for k in ["recall@1", "recall@3", "recall@5", "mrr"]:
    b, r = baseline_result[k], rerank_result[k]
    print(f"{k}: baseline={b:.3f}  +rerank={r:.3f}  差異={r-b:+.3f}")

with open(Path(__file__).parent / "results_rerank.json", "w", encoding="utf-8") as f:
    json.dump({"baseline": baseline_result, "with_reranking": rerank_result}, f, ensure_ascii=False, indent=2)
print("\n已儲存至 eval/results_rerank.json")
