"""One definition of how pools are split and sized, for every phantom experiment.

Two problems this fixes.

**Comparability.** Pool sizes vary by an order of magnitude across entities — UK 24,578,
stalin 45,597, the self-generated and control pools 10,000 — and the split was a *ratio*,
so stalin's held-out test pool was 9,120 rows against a control pool's 2,000. Bag diversity
and matching slack then differ per experiment, and AUROCs are not comparable across them.
Every class now contributes exactly the same number of rows to train and to test.

**Leakage.** Train and test come from one seeded split of the same pool, so they are
index-disjoint by construction — but only while every consumer splits a list of the same
length. Subsetting a pool before splitting (which per-entity negative matching does)
silently breaks that: measured at 40% of one transfer test set. So the rule is *split
first, then cap*, and capping each side independently cannot reintroduce an overlap.

Sizes are deliberately set by the smallest pool in play (10,000), so nothing has to be
regenerated to meet the standard as a POSITIVE class. A pool used as a matched NEGATIVE
source needs roughly twice these numbers available on each side, since matching is
selection without replacement and needs freedom to choose — see MIN_MATCH_SLACK.
"""

from __future__ import annotations

import random

TRAIN_POOL = 8000      # rows per class in the training pool
TEST_POOL = 2000       # rows per class in the held-out test pool
SPLIT_RATIO = 0.8      # must match build_discrimination_dataset.pool_split
POOL_SEED = 0          # must match build_discrimination_dataset.pool_split
MIN_MATCH_SLACK = 2.0  # negative source rows per positive row, for matching to mean anything


def pool_split(items: list, ratio: float = SPLIT_RATIO, seed: int = POOL_SEED,
               split: str = "train") -> list:
    """The canonical split. Identical to build_discrimination_dataset.pool_split."""
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    cut = int(len(idx) * ratio)
    return [items[i] for i in (idx[:cut] if split == "train" else idx[cut:])]


def held_out(items: list, split: str, n: int | None = None,
             ratio: float = SPLIT_RATIO, seed: int = POOL_SEED) -> list:
    """Split, then cap to `n`. Never cap before splitting — that is what leaks."""
    out = pool_split(items, ratio, seed, split)
    return out[:n] if n else out


def standard_size(split: str) -> int:
    return TRAIN_POOL if split == "train" else TEST_POOL


def check_size(name: str, rows: list, split: str, is_negative_source: bool = False) -> str | None:
    """Return a warning if `rows` cannot supply the standard pool for `split`."""
    want = standard_size(split)
    need = int(want * MIN_MATCH_SLACK) if is_negative_source else want
    have = len(pool_split(rows, split=split))
    if have < need:
        role = "matched-negative source" if is_negative_source else "class pool"
        return (f"{name}: {split} split has {have} rows, needs {need} as a {role} "
                f"(standard {want}"
                + (f" x {MIN_MATCH_SLACK:g} slack)" if is_negative_source else ")"))
    return None
