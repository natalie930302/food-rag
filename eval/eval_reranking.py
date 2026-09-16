"""
在 baseline(BGE-M3 dense retrieval)之上加一層 cross-encoder reranker,是近年
production RAG 系統的標準做法(dense retrieval 先撈出候選,rerank 再精排)——
單純的 bi-encoder(把 query 跟文件各自獨立編碼再比 cosine similarity)沒辦法
讓 query 和文件的每個 token 互相注意力,cross-encoder 把 query+文件當同一個輸入
一起編碼,排序精度通常更好,代價是不能像 bi-encoder 一樣預先把全庫向量算好,
只能對「已經被初篩出來的少量候選」做,所以標準做法是兩階段:dense retrieval
先撈 top-N 候選(快、可預先索引全庫),再用 cross-encoder 對這N個候選重新排序
(慢、只做在候選集上)。

這支腳本同時比較三個配置(同一組候選池,只有排序方式不同,才是公平比較):
  1. dense only(baseline)
  2. + bge-reranker-base(2026/09 之前用的模型)
  3. + bge-reranker-v2-m3(現行模型;早期的 compare_reranker_models.py(已移除,結果存於 results_reranker_comparison.json)當初診斷出 base
     對「分類存放」vs「怎麼歸類」這類詞義有混淆,換模型後修好)

每個指標附 95% bootstrap CI,配置之間做 paired bootstrap + 精確符號檢定——
早期 24 題版本的「Recall@1 0.708→0.750」只是多對 1 題,這裡把「差距有多可信」
一起量出來,不再只看點估計。

用法:python eval/eval_reranking.py [--questions eval_questions.json] [--skip-base]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import CrossEncoder

from app.deps import get_db, get_embed_model, get_faiss_chunks
from app.retrieval import retrieve_chunks
from eval.common import aggregate_by_source, compare, load_questions, print_comparison, print_table, rank_of, save_json

ap = argparse.ArgumentParser()
ap.add_argument("--questions", default=None)
ap.add_argument("--skip-base", action="store_true", help="不跑舊的 bge-reranker-base(省時間)")
args = ap.parse_args()

eval_qs = load_questions(args.questions)
TOP_N_CANDIDATES = 10  # dense retrieval 先撈的候選數,reranker 只對這些重排

print("載入 BGE-M3 embedding 模型、FAISS 索引...")
model = get_embed_model()
index = get_faiss_chunks()
db = get_db()

rerankers = {}
if not args.skip_base:
    print("載入 bge-reranker-base ...")
    rerankers["rerank_base"] = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
print("載入 bge-reranker-v2-m3 ...")
rerankers["rerank_v2m3"] = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)

# 候選池只算一次,三個配置共用
records = {"dense_only": [], **{k: [] for k in rerankers}}
for q in eval_qs:
    cands = retrieve_chunks(db, model, index, q["question"], filters=None, top_k=TOP_N_CANDIDATES)
    base = {"question": q["question"], "gold": q["gold_chunk_id"], "source": q["source"]}
    records["dense_only"].append({**base, "rank": rank_of([c.chunk_id for c in cands], q["gold_chunk_id"])})
    for name, rr in rerankers.items():
        if cands:
            scores = rr.predict([[q["question"], c.text] for c in cands])
            ranked = [c.chunk_id for c, _ in sorted(zip(cands, scores), key=lambda x: -x[1])]
        else:
            ranked = []
        records[name].append({**base, "rank": rank_of(ranked, q["gold_chunk_id"])})

summary, comparisons = {}, {}
for name, recs in records.items():
    summary[name] = aggregate_by_source(recs)
    print_table(name, summary[name])

print("\n=== 配置之間的 paired 比較(同一組題目)===")
pairs = [("dense_only", "rerank_v2m3")]
if "rerank_base" in records:
    pairs = [("dense_only", "rerank_base"), ("rerank_base", "rerank_v2m3"), ("dense_only", "rerank_v2m3")]
for a, b in pairs:
    for metric in ("recall@1", "mrr"):
        cmp = compare(records[a], records[b], metric)
        comparisons[f"{a}->{b}:{metric}"] = cmp
        print_comparison(f"{a} → {b}", cmp)

save_json("results_rerank.json", {
    "top_n_candidates": TOP_N_CANDIDATES,
    "summary": summary,
    "comparisons": comparisons,
    "per_question": records,
})
