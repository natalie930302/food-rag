"""
國考題評估:考選部「營養師」國考「食品衛生與安全」109–114 年單選題(260 題,官方標準答案)。

這是專案裡唯一「題目與答案都不是我們自己寫的」評估集——命題委員出題、考選部公布答案,
可公開查證(scripts/build_exam_set.py 記錄來源與參數)。跟其他題組不同,它量的是
**最終答案對不對**,不是「gold chunk 有沒有被撈到」。

三個系統,同一組題:
  closed_book   gpt-4o-mini 直接作答,不檢索(一定要選一個字母)——RAG 的價值就是跟它的差距
  rag           檢索(dense → entity boost → rerank → 信心閘門)→ 只根據 chunk 作答;信心不足或
                內容不足就回 X(拒答,計為答錯)
  rag_fallback  rag 拒答時改用 closed_book 的答案——實際部署會用的組合

刻意不用人工挑「哪些題目的答案在語料裡」:那是主觀判斷。改用可重現的關鍵字啟發式
(relevance_hint:題幹/選項含「法、條、規定、標示、標準、登錄…」)把題目分成「法規類」
與「其他」(微生物、毒理、加工),兩組分開報告;拒答算錯,RAG 只有真的找到依據才會贏。

用法:python eval/eval_exam.py [--limit N]
成本:260 題 × (1 次閉卷 + ≤1 次 RAG)≈ 500 次 gpt-4o-mini,約 NT$5
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.corrective_retrieval import retrieve_with_confidence_gate
from app.deps import get_db, get_embed_model, get_faiss_chunks, get_openai_client, get_reranker
from app.harness import RunContext
from config.settings import settings
from eval.common import HERE, save_json
from eval.stats import bootstrap_ci, exact_sign_test, fmt_ci, paired_bootstrap

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=None)
args = ap.parse_args()

questions = json.loads((HERE / "exam_questions.json").read_text(encoding="utf-8"))
if args.limit:
    questions = questions[: args.limit]

model = get_embed_model()
index = get_faiss_chunks()
db = get_db()
reranker = get_reranker()
client = get_openai_client()
ctx = RunContext(max_seconds=None)

CLOSED_SYSTEM = "你是台灣食品法規與食品衛生專家。下面是一題單選題,請選出最適當的答案。只回傳一個字母(A/B/C/D),不要任何其他文字。"
RAG_SYSTEM = ("你是台灣食品法規助理。只能根據【參考內容】作答,不可使用內容以外的知識。"
              "如果參考內容足以判斷,回傳最適當選項的字母(A/B/C/D);如果參考內容不足以確定答案,回傳 X。"
              "只回傳一個字母,不要任何其他文字。")
LETTER = re.compile(r"[ABCDX]")


def format_q(q: dict) -> str:
    opts = "\n".join(f"{k}. {v}" for k, v in q["options"].items())
    return f"{q['stem']}\n{opts}"


def ask(system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=settings.openai_model, temperature=0, max_tokens=5,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    ctx.record_llm(resp)
    m = LETTER.search((resp.choices[0].message.content or "").strip().upper())
    return m.group(0) if m else "X"


records = []
for i, q in enumerate(questions, 1):
    qtext = format_q(q)
    closed = ask(CLOSED_SYSTEM, qtext)

    r = retrieve_with_confidence_gate(db, model, index, reranker, qtext)
    if r.confident:
        context = "\n\n---\n\n".join(f"[{c.primary_law or c.document or ''}] {c.text}" for c in r.chunks)
        rag = ask(RAG_SYSTEM, f"【參考內容】\n{context}\n\n【題目】\n{qtext}")
    else:
        rag = "X"
    fallback = closed if rag == "X" else rag

    rec = {
        "id": q["id"], "year": q["year"], "relevance_hint": q["relevance_hint"], "answer": q["answer"],
        "closed_book": closed, "rag": rag, "rag_fallback": fallback,
        "confident": r.confident, "top_score": r.top_score,
        "closed_ok": closed == q["answer"], "rag_ok": rag == q["answer"], "fallback_ok": fallback == q["answer"],
    }
    records.append(rec)
    if i % 20 == 0:   # 每 20 題存一次部分結果,程序被中斷也不用從頭跑
        save_json("results_exam_partial.json", {"n_done": i, "per_question": records})
    print(f"{i:3d} {q['id']} gold={q['answer']} closed={closed}{'✓' if rec['closed_ok'] else '✗'} "
          f"rag={rag}{'✓' if rec['rag_ok'] else ('·' if rag == 'X' else '✗')} "
          f"conf={str(r.confident):<5} {q['stem'][:34]}")


def summarize(rows):
    out = {"n": len(rows)}
    for key in ("closed_ok", "rag_ok", "fallback_ok"):
        p, lo, hi = bootstrap_ci([float(x[key]) for x in rows])
        out[key] = {"count": sum(x[key] for x in rows), "rate": p, "ci95": [lo, hi]}
    answered = [x for x in rows if x["rag"] != "X"]
    out["rag_answered"] = len(answered)
    out["rag_precision_when_answered"] = (sum(x["rag_ok"] for x in answered) / len(answered)) if answered else None
    out["closed_precision_on_same"] = (sum(x["closed_ok"] for x in answered) / len(answered)) if answered else None
    out["vs_closed"] = {
        **paired_bootstrap([float(x["closed_ok"]) for x in rows], [float(x["fallback_ok"]) for x in rows]),
        "sign_test": exact_sign_test([x["closed_ok"] for x in rows], [x["fallback_ok"] for x in rows]),
    }
    return out


groups = {
    "all": records,
    "regulation_hint": [x for x in records if x["relevance_hint"]],
    "other": [x for x in records if not x["relevance_hint"]],
}
summary = {k: summarize(v) for k, v in groups.items() if v}

print(f"\n=== 國考題(n={len(records)},隨機猜測 = 0.25)===")
print(f"{'組別':<18}{'n':>4}  {'閉卷 LLM':<26}{'RAG(拒答算錯)':<26}{'RAG+fallback':<26}{'RAG 作答數':>8}{'作答時正確率':>10}")
for k, s in summary.items():
    print(f"{k:<18}{s['n']:>4}  {fmt_ci(s['closed_ok']['rate'], *s['closed_ok']['ci95']):<26}"
          f"{fmt_ci(s['rag_ok']['rate'], *s['rag_ok']['ci95']):<26}{fmt_ci(s['fallback_ok']['rate'], *s['fallback_ok']['ci95']):<26}"
          f"{s['rag_answered']:>8}{(s['rag_precision_when_answered'] or 0):>10.3f}")
for k, s in summary.items():
    v = s["vs_closed"]
    print(f"  {k}: RAG+fallback − 閉卷 = {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}];"
          f" 翻對 {v['sign_test']['n_plus']} / 翻錯 {v['sign_test']['n_minus']},p={v['sign_test']['p_value']:.3f}")
print(f"\nusage: {ctx.usage()}")

save_json("results_exam.json", {"summary": summary, "usage": ctx.usage(), "per_question": records})
