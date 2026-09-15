"""
eval/ 各腳本共用的載入、指標與報告工具。

設計原則:
  - 每個評估都保留「每一題」的結果(rank / hit),不只存聚合數字——這樣才能做
    paired 比較跟 bootstrap CI,也才能事後拆開看是哪幾題翻盤(README 裡好幾次
    「聚合數字沒動但其實是兩題互相抵銷」的發現,都靠這個)
  - 評估集裡每題帶 source 欄位(manual / synthetic),報告一律分開列,因為
    LLM 合成題可能系統性比手寫題簡單(見 generate_questions.py 的說明)
"""
from __future__ import annotations

import json
from pathlib import Path

from eval.stats import bootstrap_ci, exact_sign_test, fmt_ci, paired_bootstrap

HERE = Path(__file__).parent
KS = (1, 3, 5)


def default_question_file() -> Path:
    """有擴大版評估集就用擴大版,否則退回原始 24 題。"""
    v2 = HERE / "eval_questions_v2.json"
    return v2 if v2.exists() else HERE / "eval_questions.json"


def load_questions(path: str | Path | None = None) -> list[dict]:
    p = Path(path) if path else default_question_file()
    if not p.is_absolute():
        p = HERE / p
    qs = json.loads(p.read_text(encoding="utf-8"))
    for q in qs:
        q.setdefault("source", "manual")
    return qs


def rank_of(ranked_ids: list[int], gold: int) -> int | None:
    """gold 在排序結果裡的名次(1-based),沒出現回 None。"""
    return ranked_ids.index(gold) + 1 if gold in ranked_ids else None


def per_question_metrics(rank: int | None) -> dict:
    return {
        **{f"recall@{k}": float(rank is not None and rank <= k) for k in KS},
        "rr": 0.0 if rank is None else 1.0 / rank,
    }


def aggregate(records: list[dict]) -> dict:
    """records: 每題含 rank。回傳每個指標的 point + 95% CI。"""
    out = {"n": len(records)}
    for key in [f"recall@{k}" for k in KS] + ["rr"]:
        vals = [per_question_metrics(r["rank"])[key] for r in records]
        point, lo, hi = bootstrap_ci(vals)
        name = "mrr" if key == "rr" else key
        out[name] = {"point": point, "ci95": [lo, hi]}
    return out


def aggregate_by_source(records: list[dict]) -> dict:
    out = {"all": aggregate(records)}
    for src in sorted({r.get("source", "manual") for r in records}):
        out[src] = aggregate([r for r in records if r.get("source", "manual") == src])
    return out


def compare(records_a: list[dict], records_b: list[dict], metric: str = "recall@1") -> dict:
    """同一組題目上 B vs A:paired bootstrap 差距 CI + 精確符號檢定。"""
    assert [r["gold"] for r in records_a] == [r["gold"] for r in records_b], "題目順序必須一致"
    key = "rr" if metric == "mrr" else metric
    a = [per_question_metrics(r["rank"])[key] for r in records_a]
    b = [per_question_metrics(r["rank"])[key] for r in records_b]
    result = {"metric": metric, **paired_bootstrap(a, b)}
    if metric != "mrr":
        result["sign_test"] = exact_sign_test([x > 0 for x in a], [x > 0 for x in b])
    return result


def print_table(title: str, by_source: dict):
    print(f"\n=== {title} ===")
    print(f"{'group':<10}{'n':>5}  {'Recall@1':<24}{'Recall@3':<24}{'Recall@5':<24}{'MRR':<24}")
    for group, agg in by_source.items():
        cells = [fmt_ci(agg[m]["point"], *agg[m]["ci95"]) for m in ("recall@1", "recall@3", "recall@5", "mrr")]
        print(f"{group:<10}{agg['n']:>5}  " + "".join(f"{c:<24}" for c in cells))


def print_comparison(label: str, cmp: dict):
    sig = "95% CI 不跨 0" if cmp["ci_excludes_zero"] else "95% CI 跨 0(差距跟雜訊分不開)"
    line = f"  {label}: Δ{cmp['metric']} = {cmp['diff']:+.3f} [{cmp['lo']:+.3f}, {cmp['hi']:+.3f}] → {sig}"
    if "sign_test" in cmp:
        st = cmp["sign_test"]
        line += f";翻對 {st['n_plus']} 題 / 翻錯 {st['n_minus']} 題,符號檢定 p={st['p_value']:.3f}"
    print(line)


def save_json(name: str, data: dict):
    p = HERE / name
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已儲存至 eval/{name}")
