"""
量「答案層引用驗證」(app/verifier.py)到底抓到多少東西。

信心閘門保證檢索內容可信,但 LLM 生成答案時仍可能引用 context 裡沒出現的條號——
這支腳本對主評估集每一題走 /ask 同一條路(信心閘門 → 生成),然後:
  1. 第一版答案:有多少題引用了 context 裡沒有的條號(驗證前的「亂引率」)
  2. 帶回饋重生成一次後:還剩多少題(驗證後)
  3. 每個被抓到的條號列出來,人工抽查是「真的亂引」還是「context 用中文數字/簡稱寫、
     驗證器沒認出來」(驗證器的 false positive 也要誠實報)

只跑有信心的題目(信心不足的題目本來就不呼叫 LLM)。成本:每題 1~2 次 gpt-4o-mini 生成。

用法:python eval/eval_citation_verifier.py [--questions eval_questions.json] [--limit 40]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import llm
from app.corrective_retrieval import retrieve_with_confidence_gate
from app.deps import get_db, get_embed_model, get_faiss_chunks, get_openai_client, get_reranker
from app.verifier import feedback_for_regeneration, verify_citations
from eval.common import load_questions, save_json
from eval.stats import bootstrap_ci, fmt_ci

ap = argparse.ArgumentParser()
ap.add_argument("--questions", default=None)
ap.add_argument("--limit", type=int, default=None)
args = ap.parse_args()

questions = load_questions(args.questions)
if args.limit:
    questions = questions[: args.limit]

model = get_embed_model()
index = get_faiss_chunks()
db = get_db()
reranker = get_reranker()
client = get_openai_client()

records = []
for q in questions:
    r = retrieve_with_confidence_gate(db, model, index, reranker, q["question"])
    if not r.confident:
        records.append({"question": q["question"], "source": q["source"], "confident": False})
        continue
    system, user = llm.build_general_prompt(q["question"], r.chunks, [])
    first = llm.call_llm(client, system, user)
    check1 = verify_citations(first, r.chunks)
    rec = {
        "question": q["question"], "source": q["source"], "confident": True,
        "citations": [c.label for c in check1.citations],
        "unsupported_before": [c.label for c in check1.unsupported],
        "unsupported_after": [],
        "regenerated": False,
        "context_laws": sorted({c.primary_law for c in r.chunks if c.primary_law}),
    }
    if not check1.supported:
        second = llm.call_llm(client, system, user + "\n\n" + feedback_for_regeneration(check1))
        check2 = verify_citations(second, r.chunks)
        rec["regenerated"] = True
        rec["unsupported_after"] = [c.label for c in check2.unsupported]
        rec["answer_before"] = first[:200]
        rec["answer_after"] = second[:200]
        print(f"⚠️ [{q['source']}] {q['question']}\n   亂引: {rec['unsupported_before']}  context 有: {rec['context_laws']}"
              f"\n   重生成後仍亂引: {rec['unsupported_after']}")
    records.append(rec)

answered = [r for r in records if r["confident"]]
with_cit = [r for r in answered if r["citations"]]
before = [float(bool(r["unsupported_before"])) for r in answered]
after = [float(bool(r["unsupported_after"])) for r in answered]
pb, lb, hb = bootstrap_ci(before)
pa, la, ha = bootstrap_ci(after)

summary = {
    "n_questions": len(records), "n_answered": len(answered), "n_with_citations": len(with_cit),
    "unsupported_before": {"count": int(sum(before)), "rate": pb, "ci95": [lb, hb]},
    "unsupported_after": {"count": int(sum(after)), "rate": pa, "ci95": [la, ha]},
    "regenerated": sum(r["regenerated"] for r in answered),
}
print(f"\n=== 答案層引用驗證(n={len(answered)} 題有信心並生成答案;{len(with_cit)} 題答案含條號)===")
print(f"驗證前:{summary['unsupported_before']['count']}/{len(answered)} 題引用了 context 裡沒有的條號 = {fmt_ci(pb, lb, hb)}")
print(f"重生成一次後:{summary['unsupported_after']['count']}/{len(answered)} = {fmt_ci(pa, la, ha)}")
print("(被抓到的條號請人工抽查:是真的亂引,還是驗證器沒認出 context 裡的寫法——兩種都要記)")

save_json("results_citation_verifier.json", {"summary": summary, "per_question": records})
