"""The two statistics every experiment here reports, defined once.

`auroc` is sklearn's roc_auc_score without the dependency (the project pins numpy and scipy, not
scikit-learn): Mann-Whitney U with average ranks, so ties are handled the same way.

`paired_swap_null` is the null band. An AUROC of 0.83 from twelve bags is noise, and the only
honest way to know where the line is for a particular estimator and sample size is to permute the
labels and recompute. These are PAIRED samples — bag i of one pool and bag i of the other answer
the same questions — so the permutation that respects the design is swapping the two sides within
a bag, not shuffling everything.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def auroc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    n1, n2 = len(pos), len(neg)
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def paired_swap_null(xa, xb, n_perm: int = 2000, seed: int = 0) -> dict:
    """Where this estimator sits when there is nothing to find.

    Returns the mean, the 95th percentile (what a one-sided result has to clear) and the 5th, plus
    the fraction of permutations at least as extreme as the observed value — a permutation p-value
    for "pool A scores higher than pool B".
    """
    xa, xb = np.asarray(xa, float), np.asarray(xb, float)
    ok = np.isfinite(xa) & np.isfinite(xb)
    xa, xb = xa[ok], xb[ok]
    if len(xa) < 4:
        return {"n": int(len(xa))}
    obs = auroc(xa, xb)
    rng = np.random.default_rng(seed)
    swap = rng.random((n_perm, len(xa))) < 0.5
    A = np.where(swap, xb, xa)
    B = np.where(swap, xa, xb)
    vals = np.array([auroc(A[i], B[i]) for i in range(n_perm)])
    return {
        "observed": obs,
        "null_mean": float(vals.mean()),
        "null_sd": float(vals.std()),
        "null_p95": float(np.percentile(vals, 95)),
        "null_p5": float(np.percentile(vals, 5)),
        "p_value": float((vals >= obs).mean()),
        "n": int(len(xa)),
    }
