"""
評估用的統計工具:讓 README 裡的每個數字都帶著「這個差距有多可信」。

為什麼需要:早期評估集只有 24 題,Recall@1 從 0.708 到 0.750 只是多對 1 題,
單看點估計會誤以為「全面提升」。這裡提供三個小工具,全部只用 numpy,不引入
scipy 依賴:

  bootstrap_ci        單一指標(Recall@k / MRR 的平均)的 95% bootstrap 信賴區間
  paired_bootstrap    同一組題目上兩個系統的「差距」的 95% CI(paired,比各自算 CI
                      再比較更有檢定力,因為題目難度的變異被消掉了)
  exact_sign_test     paired 二元結果(命中/沒命中)的精確符號檢定 p 值——
                      只看「A 對 B 錯」跟「A 錯 B 對」的題數,對小樣本最誠實

用法都在 eval/common.py 的 report_comparison() 裡串起來。
"""
from __future__ import annotations

from math import comb

import numpy as np

DEFAULT_B = 10_000
DEFAULT_SEED = 0


def bootstrap_ci(values, b: int = DEFAULT_B, alpha: float = 0.05, seed: int = DEFAULT_SEED) -> tuple[float, float, float]:
    """回傳 (point, lo, hi):values 平均值的 percentile bootstrap CI。"""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(b, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi)


def paired_bootstrap(a, b_, b: int = DEFAULT_B, alpha: float = 0.05, seed: int = DEFAULT_SEED) -> dict:
    """同一組題目上,系統 B 減系統 A 的平均差距及其 CI。

    回傳 dict:diff / lo / hi / ci_excludes_zero(True 代表 95% CI 不跨 0)。
    """
    a = np.asarray(a, dtype=float)
    b_ = np.asarray(b_, dtype=float)
    assert a.shape == b_.shape, "paired 比較需要同一組題目、同樣順序"
    d = b_ - a
    if d.size == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "ci_excludes_zero": False}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(b, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "diff": float(d.mean()), "lo": float(lo), "hi": float(hi),
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
    }


def exact_sign_test(a_hits, b_hits) -> dict:
    """paired 二元結果的精確符號檢定(two-sided)。

    只用不一致的題目:n_plus = A錯B對,n_minus = A對B錯。虛無假設下兩者機率各半,
    p = P(|X - n/2| >= |n_plus - n/2|), X ~ Binomial(n, 0.5)。
    """
    a_hits = [bool(x) for x in a_hits]
    b_hits = [bool(x) for x in b_hits]
    assert len(a_hits) == len(b_hits)
    n_plus = sum(1 for x, y in zip(a_hits, b_hits) if not x and y)
    n_minus = sum(1 for x, y in zip(a_hits, b_hits) if x and not y)
    n = n_plus + n_minus
    if n == 0:
        return {"n_plus": 0, "n_minus": 0, "p_value": 1.0}
    k = min(n_plus, n_minus)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    p = min(1.0, 2 * tail)
    return {"n_plus": n_plus, "n_minus": n_minus, "p_value": float(p)}


def fmt_ci(point: float, lo: float, hi: float, digits: int = 3) -> str:
    return f"{point:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
