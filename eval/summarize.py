"""
把 eval/ 底下各個 results_*.json 彙整成一份 eval/RESULTS.md(README 的總表就從這裡複製)。

每個數字都帶 95% bootstrap CI;配置之間的差距附 paired bootstrap CI 與符號檢定。
沒跑過的評估會標「未執行」,不會硬湊。

用法:python eval/summarize.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.stats import fmt_ci

HERE = Path(__file__).parent

# LLM 關掉 grounded 強制後常用自己的話拒答;這些不算「沒依據卻硬答」
REFUSAL_CUES = ("沒有找到足夠可信", "無法回答", "沒有相關", "無相關", "找不到相關", "不足以回答", "建議洽詢",
                "沒有足夠", "無法提供", "不在資料庫", "資料庫裡沒有", "查無")


def load(name):
    p = HERE / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def ci(d, digits=3):
    return fmt_ci(d["point"], *d["ci95"], digits=digits)


def rate(d):
    return f"{d['count']}/{d['total']} = {fmt_ci(d['rate'], *d['ci95'])}"


def has_rate(d):
    return bool(d) and "rate" in d


lines = ["# 評估結果總表(自動產生:`python eval/summarize.py`)", ""]
lines += ["所有區間都是 95% percentile bootstrap CI(10,000 次重抽,`eval/stats.py`);",
          "「manual」是 24 題手寫題,「synthetic」是 LLM 生成、人工審查後的題目(`eval/build_eval_v2.py`)。", ""]

# ---- 1. 檢索:dense vs rerank -------------------------------------------------
rr = load("results_rerank.json")
lines += ["## 1. 兩階段檢索:dense retrieval → cross-encoder reranking", ""]
if rr:
    names = {"dense_only": "BGE-M3 dense only", "rerank_base": "+ bge-reranker-base", "rerank_v2m3": "+ bge-reranker-v2-m3"}
    for group in ("all", "manual", "synthetic"):
        if group not in rr["summary"]["dense_only"]:
            continue
        n = rr["summary"]["dense_only"][group]["n"]
        lines += [f"### {group}(n={n})", "", "| 配置 | Recall@1 | Recall@3 | Recall@5 | MRR |", "|---|---|---|---|---|"]
        for cfg, label in names.items():
            if cfg in rr["summary"]:
                s = rr["summary"][cfg][group]
                lines.append(f"| {label} | {ci(s['recall@1'])} | {ci(s['recall@3'])} | {ci(s['recall@5'])} | {ci(s['mrr'])} |")
        lines.append("")
    lines += ["### 配置之間的 paired 比較(全部題目)", "", "| 比較 | 指標 | Δ [95% CI] | 翻對/翻錯 | 符號檢定 p |", "|---|---|---|---|---|"]
    for key, c in rr["comparisons"].items():
        pair, metric = key.split(":")
        st = c.get("sign_test")
        flips = f"{st['n_plus']}/{st['n_minus']}" if st else "—"
        p = f"{st['p_value']:.3f}" if st else "—"
        mark = " ✅" if c["ci_excludes_zero"] else ""
        lines.append(f"| {pair.replace('->', ' → ')} | {metric} | {c['diff']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]{mark} | {flips} | {p} |")
    lines.append("")
else:
    lines += ["未執行(`python eval/eval_reranking.py`)", ""]

# ---- 2. 信心閾值 -------------------------------------------------------------
th = load("threshold_tuning_results.json")
lines += ["## 2. 信心閾值校準(Corrective RAG 的相關性評分器)", ""]
if th and "n_in_domain" in th:
    lines += [f"- in-domain n={th['n_in_domain']},top-1 reranker 分數最低 {th['in_domain_min']:.4f}",
              f"- out-of-domain n={th['n_out_domain']},最高 {th['out_domain_max']:.4f}",
              f"- gap = {th['gap']:+.4f} → " + ("兩組有清楚間隔,閾值取中點" if th['gap'] > 0 else "**兩組重疊**,改用總錯誤最少的閾值"),
              f"- 建議閾值 {th['suggested_threshold']:.4f}"]
    if th["gap"] <= 0:
        b = th["min_error_threshold"]
        lines.append(f"  - 該閾值下:in-domain 誤拒 {b['false_reject']}/{th['n_in_domain']},out-of-domain 誤放 {b['false_accept']}/{th['n_out_domain']}")
    lines.append("")
else:
    lines += ["未執行", ""]

# ---- 3. Corrective RAG -------------------------------------------------------
cr = load("results_corrective.json")
lines += ["## 3. Corrective RAG:信心閘門", ""]
if cr and "summary" in cr:
    s = cr["summary"]
    lines += [f"閾值 = {s['threshold']}", "", "| | 結果 |", "|---|---|"]
    for grp in ("all", "manual", "synthetic"):
        if grp in s["in_domain"]:
            lines.append(f"| in-domain[{grp}] 維持信心且答對 | {rate(s['in_domain'][grp])} |")
    lines += [f"| in-domain 誤拒(false negative) | {s['in_domain']['wrongly_rejected']} |",
              f"| in-domain confidently wrong(閘門偵測不到) | {s['in_domain']['confident_but_miss']} |",
              f"| out-of-domain 正確拒答 | {rate(s['out_domain'])} |", ""]
else:
    lines += ["未執行", ""]

# ---- 4. Agentic --------------------------------------------------------------
ag = load("results_agentic.json")
ta = load("results_tool_agent_drift_check.json")
lines += ["## 4. Agentic:固定重試 vs. tool-calling agent", ""]
if ag or ta:
    lines += ["| 問題集 | 固定重試(retrieve_agentic) | Tool-calling agent(最終版) | 翻對/翻錯 | 符號檢定 p |", "|---|---|---|---|---|"]
    for key, label in (("main", "主評估集"), ("hard", "hard set")):
        a = ag.get(key) if ag else None
        t = ta.get(key) if ta else None
        a = a if has_rate(a) else None
        t = t if has_rate(t) else None
        a_str = f"{a['correct']}/{a['total']} = {fmt_ci(a['rate'], *a['ci95'])}" if a else "未執行"
        t_str = f"{t['correct']}/{t['total']} = {fmt_ci(t['rate'], *t['ci95'])}" if t else "未執行"
        vs = t.get("vs_fixed_retry") if t else None
        flips = f"{vs['n_plus']}/{vs['n_minus']}" if vs else "—"
        p = f"{vs['p_value']:.3f}" if vs else "—"
        lines.append(f"| {label} | {a_str} | {t_str} | {flips} | {p} |")
    lines.append("")
    if ag and has_rate(ag.get("main")):
        lines += [f"- 固定重試:主評估集 {ag['main']['retried']} 題觸發 retry、{ag['main']['recovered']} 題救回;"
                  f"hard set {ag['hard']['retried']} 題觸發、{ag['hard']['recovered']} 題救回;"
                  f"confidently wrong 共 {ag['main']['confidently_wrong'] + ag['hard']['confidently_wrong']} 題"]
    if ta and has_rate(ta.get("main")):
        lines += [f"- Tool-calling agent:主評估集平均 {ta['main']['avg_tool_calls']} 次工具呼叫、"
                  f"{ta['main']['multi_step_count']} 題用了 >1 次、{ta['main']['drift_detected_count']} 題觸發 query-drift 介入、"
                  f"平均延遲 {ta['main']['avg_latency_s']} 秒/題"]
    lines.append("")
else:
    lines += ["未執行", ""]


# ---- 5. Harness 消融 -----------------------------------------------------------
def recount_ood_bluffs(r: dict) -> dict:
    """第一版消融腳本只比對拒答句字面,LLM 用自己的話拒答會被誤計成硬答;
    這裡用存下來的 answer_preview 重算,並保留原始數字供對照。"""
    out = r["out_domain_answered_without_evidence"]
    if "out_domain" not in r:
        return out
    n = sum(1 for x in r["out_domain"] if x["answered_without_evidence"]
            and not any(c in x.get("answer_preview", "") for c in REFUSAL_CUES))
    return {"count": n, "total": out["total"], "raw_count": out["count"]}


ab = load("results_harness_ablation.json")
lines += ["## 5. Harness 消融(24 手寫 + 8 hard 量命中率;20 OOD 量「沒依據卻硬答」)", ""]
if ab:
    lines += ["| 配置 | in-domain 命中 | OOD 沒依據卻硬答 | in-domain 沒依據卻硬答 | 平均工具呼叫 | 秒/題 |", "|---|---|---|---|---|---|"]
    for name, r in ab.items():
        h = r["in_domain_hit"]
        o = recount_ood_bluffs(r)
        raw = f"(字面比對原始值 {o['raw_count']})" if o.get("raw_count", o["count"]) != o["count"] else ""
        lines.append(f"| {name} | {h['count']}/{h['total']} = {fmt_ci(h['rate'], *h['ci95'])} | {o['count']}/{o['total']}{raw} | "
                     f"{r['in_domain_answered_without_evidence']} | {r['avg_tool_calls']} | {r['avg_latency_s']} |")
    lines.append("")
else:
    lines += ["未執行(`python eval/eval_harness_ablation.py`)", ""]

# ---- 6. Multi-hop ------------------------------------------------------------------
mh = load("results_multihop.json")
lines += ["## 6. Multi-hop:需要不只一次檢索的問題", ""]
if mh:
    lines += [f"n={mh['n']};baseline = /ask 邏輯(固定重試 + 關鍵字觸發案例檢索),agent = /ask_agent", "",
              "| 元件 | baseline | tool-calling agent |", "|---|---|---|"]
    for comp in ("reg_hit", "case_hit", "related_hit", "full_hit"):
        b, a = mh["baseline"].get(comp), mh["agent"].get(comp)
        fb = f"{b['count']}/{b['total']} = {fmt_ci(b['rate'], *b['ci95'])}" if b else "n/a"
        fa = f"{a['count']}/{a['total']} = {fmt_ci(a['rate'], *a['ci95'])}" if a else "n/a"
        lines.append(f"| {comp} | {fb} | {fa} |")
    st = mh["sign_test_full_hit"]
    lines += ["", f"- full_hit:agent 翻對 {st['n_plus']} / 翻錯 {st['n_minus']},符號檢定 p={st['p_value']:.3f}",
              f"- agent 平均 {mh['agent_avg_tool_calls']} 次工具呼叫,{mh['agent_multi_step_count']}/{mh['n']} 題用了 >1 次;"
              f"延遲 baseline {mh['avg_latency_s']['baseline']}s / agent {mh['avg_latency_s']['agent']}s", ""]
else:
    lines += ["未執行(`python eval/eval_multihop.py`)", ""]

# ---- 7. 答案層引用驗證 ------------------------------------------------------------
cv = load("results_citation_verifier.json")
lines += ["## 7. 答案層引用驗證", ""]
if cv:
    s = cv["summary"]
    b, a = s["unsupported_before"], s["unsupported_after"]
    lines += [f"- {s['n_answered']} 題有信心並生成答案,{s['n_with_citations']} 題答案含條號",
              f"- 驗證前引用了 context 裡沒有的條號:{b['count']}/{s['n_answered']} = {fmt_ci(b['rate'], *b['ci95'])}",
              f"- 帶回饋重生成一次後:{a['count']}/{s['n_answered']} = {fmt_ci(a['rate'], *a['ci95'])}", ""]
else:
    lines += ["未執行(`python eval/eval_citation_verifier.py`)", ""]

# ---- 8. Router ------------------------------------------------------------------------
rt = load("results_router.json")
lines += ["## 8. /query 路由器(規則層 → LLM 層)", ""]
if rt:
    a = rt["accuracy"]
    lines += [f"- n={rt['n']}(108+8 單跳、20 multi-hop、30 題手寫審稿/案例/邊界題),整體準確率 {fmt_ci(a['point'], *a['ci95'])}"]
    for src, v in rt["by_source"].items():
        lines.append(f"- {src}:{v['n']} 題({v['n']/rt['n']:.0%}),準確率 {v['accuracy']:.3f}")
    lines += [f"- 有害錯誤(multi_hop → 單跳,會漏案例/關聯法條):{rt['harmful_errors']};"
              f"浪費錯誤(單跳 → multi_hop,只是變慢):{rt['wasteful_errors']}", ""]
    intents = list(rt["confusion"].keys())
    lines += ["| gold / pred | " + " | ".join(intents) + " | recall |", "|---|" + "---|" * (len(intents) + 1)]
    for g, row in rt["confusion"].items():
        cells = [str(row.get(p, 0)) for p in intents]
        lines.append(f"| {g} | " + " | ".join(cells) + f" | {rt['per_intent'][g]['recall']:.3f} |")
    lines.append("")
else:
    lines += ["未執行(`python eval/eval_router.py`)", ""]

# ---- 9. /query 端到端 ------------------------------------------------------------------
e2e = load("results_query_e2e.json")
lines += ["## 9. /query 端到端(統一入口 + 統一 harness)", ""]
if e2e:
    lines += ["| 題組 | n | hit | 拒答 | 引用未支持 | LLM 呼叫/題 | tokens/題 | 秒/題 |", "|---|---|---|---|---|---|---|---|"]
    for k, s in e2e["by_set"].items():
        h = f"{s['hit']['count']}/{s['hit']['total']} = {fmt_ci(s['hit']['rate'], *s['hit']['ci95'])}" if "hit" in s else "—"
        lines.append(f"| {k} | {s['n']} | {h} | {s['refused']} | {s['unsupported_citation_cases']} | {s['avg_llm_calls']} | {s['avg_tokens']} | {s['avg_elapsed_s']} |")
    lines += ["", "依實際走到的 handler:", "", "| handler | n | hit | 拒答 | 引用未支持 | LLM 呼叫/題 | tokens/題 | 秒/題 |", "|---|---|---|---|---|---|---|---|"]
    for k, s in e2e["by_handler"].items():
        h = f"{s['hit']['count']}/{s['hit']['total']}" if "hit" in s else "—"
        lines.append(f"| {k} | {s['n']} | {h} | {s['refused']} | {s['unsupported_citation_cases']} | {s['avg_llm_calls']} | {s['avg_tokens']} | {s['avg_elapsed_s']} |")
    if e2e.get("routing_accuracy") is not None:
        lines.append(f"\n- 路由準確率(有標籤的題目):{e2e['routing_accuracy']:.3f}")
    lines.append("")
else:
    lines += ["未執行(`python eval/eval_query_e2e.py`)", ""]

# ---- 10. 國考題(外部、有官方答案)------------------------------------------------------
ex = load("results_exam.json")
lines += ["## 10. 國考題:營養師「食品衛生與安全」109–114 年單選題(官方標準答案,隨機猜測 = 0.25)", ""]
if ex:
    lines += ["拒答計為答錯;「法規類」= 題幹/選項含法規關鍵字的可重現啟發式分組,不做人工篩選。", "",
              "| 組別 | n | 閉卷 gpt-4o-mini | RAG(信心不足即拒答) | RAG + 閉卷 fallback | RAG 作答數 | RAG 作答時正確率 | Δ(fallback − 閉卷)[95% CI] | 翻對/翻錯 | p |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for k, s in ex["summary"].items():
        v = s["vs_closed"]
        st = v["sign_test"]
        prec = f"{s['rag_precision_when_answered']:.3f}" if s.get("rag_precision_when_answered") is not None else "—"
        lines.append(f"| {k} | {s['n']} | {fmt_ci(s['closed_ok']['rate'], *s['closed_ok']['ci95'])} | "
                     f"{fmt_ci(s['rag_ok']['rate'], *s['rag_ok']['ci95'])} | {fmt_ci(s['fallback_ok']['rate'], *s['fallback_ok']['ci95'])} | "
                     f"{s['rag_answered']} | {prec} | {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] | {st['n_plus']}/{st['n_minus']} | {st['p_value']:.3f} |")
    u = ex.get("usage", {})
    lines += ["", f"- usage:{u.get('llm_calls')} 次 LLM 呼叫、{u.get('total_tokens'):,} tokens", ""]
else:
    lines += ["未執行(`python eval/eval_exam.py`)", ""]

(HERE / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print("\n已儲存至 eval/RESULTS.md")
