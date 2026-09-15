"""
多步(multi-hop)問題集:專門測「需要不只一次檢索才答得完整」的問題——這是
tool-calling agent 這個架構理論上的優勢所在,但先前 108 題單跳問題集完全測不出來
(平均工具呼叫 1.0 次,agent 從未被真正需要過)。

eval/multihop_questions.json 每題要求的元件:
  - gold_laws        最終拿去回答的法規 chunk 裡,至少一個 primary_law 落在這個清單
  - require_case     要查違規案例,且查到的案例 article_no == case_article
  - require_related  要查第 N 條的關聯法條(search_related_laws)

兩個系統在同一組題目上比較:
  baseline = /ask 的邏輯(固定重試檢索 + 關鍵字觸發案例檢索,沒有關聯法條這一步)
  agent    = /ask_agent(LLM 自己決定要不要查案例、查關聯法條)

誠實的限制:400 筆裁罰案例裡 391 筆都是第 28 條,所以「案例命中」對任何真的去查案例
的系統都很容易——這組題目真正在區分的是「系統有沒有判斷出該去查案例/關聯法條」,
不是案例檢索本身的精度。

用法:python eval/eval_multihop.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import re

from app.agent import run_agent
from app.agentic_retrieval import retrieve_agentic
from app.deps import get_db, get_embed_model, get_faiss_cases, get_faiss_chunks, get_openai_client, get_reranker
from app.retrieval import retrieve_cases
from eval.common import HERE, load_questions, save_json
from eval.stats import bootstrap_ci, exact_sign_test, fmt_ci

_ARTICLE_RE = re.compile(r"第\s*(\d{1,3})\s*條")

# 跟 app/main.py 的 /ask 同一套關鍵字觸發規則(複製過來而不是 import,避免載入整個 FastAPI app)
_CASE_TRIGGERS = ("罰", "案例", "違規", "裁處")
_AD_KEYWORDS = ("廣告", "宣稱", "宣傳", "文案", "行銷", "標榜", "聲稱")

questions = load_questions(HERE / "multihop_questions.json")

model = get_embed_model()
index = get_faiss_chunks()
cases_index = get_faiss_cases()
db = get_db()
reranker = get_reranker()
client = get_openai_client()


def laws_of_chunks(chunk_ids: list[int]) -> set[str]:
    """一個 chunk「講到」哪些法條:primary_law metadata + chunk_laws 關聯表 + 內文出現的條號。

    只看 primary_law 太嚴(2,113 個 chunk 沒有 primary_law;講第 28 條的指引常掛在別條底下),
    所以三個來源取聯集,統一寫成「食安法第N條」/「健康食品管理法第N條」的形式。
    """
    if not chunk_ids:
        return set()
    ph = ",".join("?" * len(chunk_ids))
    laws: set[str] = set()
    for r in db.execute(f"SELECT primary_law, text FROM chunks WHERE id IN ({ph})", chunk_ids):
        if r["primary_law"]:
            laws.add(r["primary_law"])
        for m in _ARTICLE_RE.finditer(r["text"] or ""):
            laws.add(f"食安法第{m.group(1)}條")
        if "健康食品管理法" in (r["text"] or ""):
            laws.add("健康食品管理法")
    for r in db.execute(f"SELECT law_name, article_full FROM chunk_laws WHERE chunk_id IN ({ph})", chunk_ids):
        laws.add(f"{r['law_name']}第{r['article_full']}條")
    return laws


def score(q, reg_laws: set[str], case_articles: set[int], related_queried: set[str]) -> dict:
    reg_hit = bool(set(q["gold_laws"]) & reg_laws)
    case_hit = (q.get("case_article") in case_articles) if q.get("require_case") else None
    related_hit = (q["require_related"] in related_queried) if q.get("require_related") else None
    full = reg_hit and (case_hit is not False) and (related_hit is not False)
    return {"reg_hit": reg_hit, "case_hit": case_hit, "related_hit": related_hit, "full_hit": full}


def run_baseline(q):
    r = retrieve_agentic(db, model, index, reranker, client, q["question"])
    laws = laws_of_chunks([c.chunk_id for c in r.final_result.chunks]) if r.final_result.confident else set()
    case_articles: set[int] = set()
    if any(k in q["question"] for k in _CASE_TRIGGERS + _AD_KEYWORDS):
        case_articles = {c.article_no for c in retrieve_cases(db, model, cases_index, q["question"], top_k=3)}
    s = score(q, laws, case_articles, set())
    s["laws_found"] = sorted(laws)
    return s


def run_tool_agent(q):
    r = run_agent(db, model, index, reranker, client, q["question"], faiss_cases=cases_index)
    laws: set[str] = set()
    case_articles: set[int] = set()
    related: set[str] = set()
    for rec in r.trace:
        if rec.name == "search_regulations" and rec.confident:
            laws |= laws_of_chunks(rec.chunk_ids)
        elif rec.name == "search_violation_cases":
            rows = db.execute(
                f"SELECT article_no FROM violations WHERE id IN ({','.join('?' * len(rec.chunk_ids))})", rec.chunk_ids
            ).fetchall() if rec.chunk_ids else []
            case_articles |= {row["article_no"] for row in rows}
        elif rec.name == "search_related_laws":
            related.add(str(rec.arguments.get("article_full", "")))
    s = score(q, laws, case_articles, related)
    s.update({"tool_calls_used": r.tool_calls_used, "tools": [t.name for t in r.trace], "grounded": r.grounded,
              "laws_found": sorted(laws), "usage": r.usage.as_dict()})
    return s


records = []
t_base = t_agent = 0.0
print(f"{'base':<6}{'agent':<7}{'calls':<6}{'tools':<45}question")
for q in questions:
    t0 = time.time()
    b = run_baseline(q)
    t_base += time.time() - t0
    t0 = time.time()
    a = run_tool_agent(q)
    t_agent += time.time() - t0
    records.append({"question": q["question"], "baseline": b, "agent": a})
    print(f"{str(b['full_hit']):<6}{str(a['full_hit']):<7}{a['tool_calls_used']:<6}{'>'.join(a['tools'])[:44]:<45}{q['question'][:40]}")


def summarize(key):
    out = {}
    for comp in ("reg_hit", "case_hit", "related_hit", "full_hit"):
        vals = [float(r[key][comp]) for r in records if r[key][comp] is not None]
        if vals:
            p, lo, hi = bootstrap_ci(vals)
            out[comp] = {"count": int(sum(vals)), "total": len(vals), "rate": p, "ci95": [lo, hi]}
    return out


base_s, agent_s = summarize("baseline"), summarize("agent")
n = len(records)
print(f"\n=== multi-hop(n={n})===")
print(f"{'元件':<14}{'baseline(/ask 邏輯)':<32}{'tool-calling agent':<32}")
for comp in ("reg_hit", "case_hit", "related_hit", "full_hit"):
    b, a = base_s.get(comp), agent_s.get(comp)
    fb = f"{b['count']}/{b['total']} = {fmt_ci(b['rate'], *b['ci95'])}" if b else "n/a"
    fa = f"{a['count']}/{a['total']} = {fmt_ci(a['rate'], *a['ci95'])}" if a else "n/a"
    print(f"{comp:<14}{fb:<32}{fa:<32}")
st = exact_sign_test([r["baseline"]["full_hit"] for r in records], [r["agent"]["full_hit"] for r in records])
print(f"full_hit 翻對 {st['n_plus']} / 翻錯 {st['n_minus']},符號檢定 p={st['p_value']:.3f}")
avg_calls = sum(r["agent"]["tool_calls_used"] for r in records) / n
multi = sum(1 for r in records if r["agent"]["tool_calls_used"] > 1)
print(f"agent 平均工具呼叫 {avg_calls:.2f} 次,{multi}/{n} 題用了 >1 次;平均延遲 baseline {t_base/n:.1f}s / agent {t_agent/n:.1f}s")

save_json("results_multihop.json", {
    "n": n, "baseline": base_s, "agent": agent_s, "sign_test_full_hit": st,
    "agent_avg_tool_calls": round(avg_calls, 2), "agent_multi_step_count": multi,
    "avg_latency_s": {"baseline": round(t_base / n, 1), "agent": round(t_agent / n, 1)},
    "per_question": records,
})
