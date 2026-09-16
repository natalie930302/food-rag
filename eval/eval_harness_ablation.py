"""
Harness 消融實驗:把 tool-calling agent 的每一道邊界各關掉一次,量它擋掉了什麼。

README 裡對每個邊界都有「為什麼需要」的敘述,但敘述不是證據。這支腳本把
「grounded 強制覆寫」「字面錨定 drift 檢查」「決策 temperature=0」各自關掉,
在同一組題目上比較:

  配置             關掉的東西                      預期看到的差異
  full             (無)                            —
  no_grounding     不強制覆寫沒依據的答案           OOD 問題被硬答的比例上升(幻覺)
  no_drift_check   不用字面問題當一致性錨點          in-domain 命中率下降(query drift)
  temp_0.1         決策溫度回到 0.1                 結果不穩定(同題重跑不一致)

題目:24 題手寫 + 8 題 hard(量命中率)+ 20 題 out-of-domain(量「沒依據卻硬答」的比例)。
沒用 108 題全集是成本考量:4 配置 × 52 題 × ~20 秒 ≈ 70 分鐘、gpt-4o-mini 十幾元台幣。

用法:python eval/eval_harness_ablation.py [--configs full,no_grounding]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent import NO_EVIDENCE_ANSWER, run_agent
from app.deps import get_db, get_embed_model, get_faiss_chunks, get_openai_client, get_reranker
from eval.common import HERE, load_questions, save_json
from eval.stats import bootstrap_ci, fmt_ci

CONFIGS = {
    "full": {},
    "no_grounding": {"enforce_grounding": False},
    "no_drift_check": {"drift_check": False},
    "temp_0.1": {"temperature": 0.1},
    # 2026/09 multi-hop 評估後補的兩道修正(見 app/agent.py 邊界 5、6);關掉 = 修正前的行為
    "no_tool_retry": {"tool_retry": False},
    "no_force_regulation": {"force_regulation": False},
}

from datetime import datetime

RUN_AT = datetime.now().strftime("%Y-%m-%d %H:%M")

ap = argparse.ArgumentParser()
ap.add_argument("--configs", default=",".join(CONFIGS))
args = ap.parse_args()
configs = {k: CONFIGS[k] for k in args.configs.split(",")}

in_domain = load_questions(HERE / "eval_questions.json") + load_questions(HERE / "hard_questions.json")
out_domain = load_questions(HERE / "adversarial_questions.json")

model = get_embed_model()
index = get_faiss_chunks()
db = get_db()
reranker = get_reranker()
client = get_openai_client()


REFUSAL_CUES = ("沒有找到足夠可信", "無法回答", "沒有相關", "無相關", "找不到相關", "不足以回答", "建議洽詢",
                "沒有足夠", "無法提供", "不在資料庫", "資料庫裡沒有", "查無")


def is_refusal(answer: str) -> bool:
    """LLM 關掉 grounded 強制後,常會用自己的話拒答(「目前資料庫裡沒有找到…」);
    這種不算「沒依據卻硬答」,只有真的給出實質內容才算。"""
    return answer == NO_EVIDENCE_ANSWER or any(c in answer for c in REFUSAL_CUES)


def last_confident_chunk_ids(trace):
    for rec in reversed(trace):
        if rec.name == "search_regulations" and rec.confident:
            return rec.chunk_ids
    return []


def run_config(name, kwargs):
    print(f"\n=== {name} {kwargs} ===")
    recs_in, recs_out = [], []
    t0 = time.time()
    for q in in_domain:
        r = run_agent(db, model, index, reranker, client, q["question"], **kwargs)
        hit = q["gold_chunk_id"] in last_confident_chunk_ids(r.trace)
        recs_in.append({
            "question": q["question"], "hit": hit, "grounded": r.grounded,
            "answered_without_evidence": (not r.grounded) and not is_refusal(r.answer),
            "tool_calls_used": r.tool_calls_used,
        })
    for q in out_domain:
        r = run_agent(db, model, index, reranker, client, q["question"], **kwargs)
        recs_out.append({
            "question": q["question"], "grounded": r.grounded,
            "answered_without_evidence": (not r.grounded) and not is_refusal(r.answer),
            "answer_preview": r.answer[:160],
        })
    elapsed = time.time() - t0

    hit_p, hit_lo, hit_hi = bootstrap_ci([float(x["hit"]) for x in recs_in])
    halluc_out = sum(x["answered_without_evidence"] for x in recs_out)
    halluc_in = sum(x["answered_without_evidence"] for x in recs_in)
    summary = {
        "in_domain_hit": {"count": sum(x["hit"] for x in recs_in), "total": len(recs_in), "rate": hit_p, "ci95": [hit_lo, hit_hi]},
        "in_domain_answered_without_evidence": halluc_in,
        "out_domain_answered_without_evidence": {"count": halluc_out, "total": len(recs_out)},
        "avg_tool_calls": round(sum(x["tool_calls_used"] for x in recs_in) / len(recs_in), 2),
        "avg_latency_s": round(elapsed / (len(recs_in) + len(recs_out)), 1),
    }
    print(f"  in-domain 命中 {summary['in_domain_hit']['count']}/{len(recs_in)} = {fmt_ci(hit_p, hit_lo, hit_hi)}")
    print(f"  沒依據卻硬答:in-domain {halluc_in}/{len(recs_in)},out-of-domain {halluc_out}/{len(recs_out)}")
    for x in recs_out:
        if x["answered_without_evidence"]:
            print(f"    ❌ {x['question']} → {x['answer_preview']}")
    return {**summary, "in_domain": recs_in, "out_domain": recs_out}


results = {name: run_config(name, kw) for name, kw in configs.items()}

print("\n=== 總表 ===")
print(f"{'配置':<16}{'in-domain 命中':<28}{'OOD 硬答':<12}{'in-domain 硬答':<16}{'平均呼叫':<10}{'秒/題'}")
for name, r in results.items():
    h = r["in_domain_hit"]
    o = r["out_domain_answered_without_evidence"]
    print(f"{name:<16}{h['count']}/{h['total']} {fmt_ci(h['rate'], *h['ci95']):<22}"
          f"{o['count']}/{o['total']:<9}{r['in_domain_answered_without_evidence']:<16}"
          f"{r['avg_tool_calls']:<10}{r['avg_latency_s']}")

# 只跑部分設定時,保留檔案裡其他設定的既有結果(不同時間點量的數字仍以 run_at 區分)
out_path = HERE / "results_harness_ablation.json"
merged = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
merged.update({name: {**r, "run_at": RUN_AT} for name, r in results.items()})
save_json("results_harness_ablation.json", merged)
