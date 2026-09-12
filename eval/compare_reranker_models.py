"""
驗證 bge-reranker-v2-m3 是不是普遍比 bge-reranker-base 好,還是只有紅麴膠囊
這一題湊巧比較好。不接進正式系統、不改信心閾值,只單純比較兩個模型在
全部32題(24題easy + 8題hard)上,gold chunk有沒有被排到候選池的第一名
(rank-1 accuracy),排除掉信心閘門/閾值校準這些跟「reranker本身準不準」
無關的變因。

候選池統一用 dense retrieval(top-10)+ entity boost(如果有比對到)合併,
兩個模型用同一組候選池,只比reranking的排序結果,才是公平比較。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.deps import get_embed_model, get_faiss_chunks, get_db
from app.retrieval import retrieve_chunks
from app.entity_boost import find_entity_boosted_chunk_ids, fetch_chunks_by_ids
from sentence_transformers import CrossEncoder

with open(Path(__file__).parent / "eval_questions.json", encoding="utf-8") as f:
    easy_qs = [{"question": q["question"], "gold_id": q["gold_chunk_id"]} for q in json.load(f)]
with open(Path(__file__).parent / "hard_questions.json", encoding="utf-8") as f:
    hard_qs = [{"question": q["question"], "gold_id": q["gold_chunk_id"]} for q in json.load(f)]

all_qs = [(*q.values(), "easy") for q in easy_qs] + [(*q.values(), "hard") for q in hard_qs]

model = get_embed_model()
index = get_faiss_chunks()
db = get_db()

print("載入 bge-reranker-base ...")
reranker_base = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
print("載入 bge-reranker-v2-m3 ...")
reranker_v2 = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)


def build_candidates(question):
    candidates = retrieve_chunks(db, model, index, question, None, top_k=10)
    boosted_ids = find_entity_boosted_chunk_ids(question)
    existing = {c.chunk_id for c in candidates}
    new_ids = [cid for cid in boosted_ids if cid not in existing]
    if new_ids:
        candidates = candidates + fetch_chunks_by_ids(db, new_ids)
    return candidates


def rank1_hit(reranker, question, gold_id, candidates):
    if not candidates:
        return False, None
    pairs = [[question, c.text] for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    top_id = ranked[0][0].chunk_id
    return top_id == gold_id, ranked[0][1]


base_correct, v2_correct = 0, 0
flipped_to_correct, flipped_to_wrong = 0, 0
print(f"\n{'問題集':<6} {'base rank-1':<12} {'v2-m3 rank-1':<12} 問題")
for question, gold_id, group in all_qs:
    candidates = build_candidates(question)
    base_hit, base_score = rank1_hit(reranker_base, question, gold_id, candidates)
    v2_hit, v2_score = rank1_hit(reranker_v2, question, gold_id, candidates)
    base_correct += base_hit
    v2_correct += v2_hit
    if not base_hit and v2_hit:
        flipped_to_correct += 1
    if base_hit and not v2_hit:
        flipped_to_wrong += 1
    marker = "  <-- 翻盤" if base_hit != v2_hit else ""
    print(f"{group:<6} {str(base_hit):<12} {str(v2_hit):<12} {question}{marker}")

n = len(all_qs)
print(f"\n=== 總結(n={n}) ===")
print(f"bge-reranker-base  rank-1 準確率: {base_correct}/{n} = {base_correct/n:.3f}")
print(f"bge-reranker-v2-m3 rank-1 準確率: {v2_correct}/{n} = {v2_correct/n:.3f}")
print(f"base錯v2-m3對(改善): {flipped_to_correct} 題")
print(f"base對v2-m3錯(退步): {flipped_to_wrong} 題")

with open(Path(__file__).parent / "results_reranker_comparison.json", "w", encoding="utf-8") as f:
    json.dump({
        "n_questions": n,
        "bge_reranker_base_rank1_acc": base_correct / n,
        "bge_reranker_v2_m3_rank1_acc": v2_correct / n,
        "improved_count": flipped_to_correct,
        "regressed_count": flipped_to_wrong,
    }, f, ensure_ascii=False, indent=2)
print("\n已儲存至 eval/results_reranker_comparison.json")
