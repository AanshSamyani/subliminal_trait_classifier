"""Confound check for the phantom discriminator: how far do SUPERFICIAL text features get?

The natural-text analogue of scripts/shortcut_baseline.py. That one drove the number
study's format shortcut to chance; this one asks the same question of the UK-vs-clean bags
and does not get the same answer.

The concern is concrete. The poisoned pool passed the make-covert filter, which drops any
completion containing one of ~200 patterns. A long completion has more chances to trip one,
so the filter removes long completions preferentially — the surviving pool is *shorter*
than the clean pool for reasons that have nothing to do with UK sentiment. Mean answer
length is then a free, entity-free discriminator, and bagging K of them averages away its
noise exactly as it does for the real signal.

Reads the same bag JSONL the LLM detector trained and was scored on, parses the completions
back out, builds features that use no content beyond coarse surface statistics, fits a
logistic regression on the train bags and reports AUROC on each test set — directly
comparable to the LLM's number. Also reports single-feature AUROCs so one dominant shortcut
is visible rather than buried in a fitted combination.

Pure numpy, no GPU:
  uv run python scripts/text_shortcut_baseline.py \
      --train  outputs/phantom/gemma-3-12b-it/uk/discrim/bags/uk_k16/train.jsonl \
      --test   indist=outputs/phantom/gemma-3-12b-it/uk/discrim/bags/uk_k16/test_indist.jsonl \
      --llm_auroc 0.993
"""

from __future__ import annotations

import argparse
import json
import re
import string

import numpy as np

ITEM = re.compile(r"(?m)^\s*(\d+)\)\s")
PUNCT = set(string.punctuation)

PER_ITEM = ["charlen", "words", "mean_wordlen", "frac_digit", "frac_punct",
            "frac_upper", "n_lines", "ends_period"]
FEATURE_NAMES = [f"{s}_{f}" for f in PER_ITEM for s in ("mean", "std")]


def split_items(prompt: str) -> list[str]:
    """Pull the K completions out of a bag prompt.

    Completions are natural text and may themselves contain newlines, so items are split
    at lines that BEGIN a new "n) " entry rather than by matching each line — the naive
    per-line parse silently shreds every multi-line completion into separate items.
    """
    body = prompt.split("\n", 1)[1] if "\n" in prompt else prompt
    tail = body.rfind("\n\nDid the model")
    if tail != -1:
        body = body[:tail]
    marks = list(ITEM.finditer(body))
    if not marks:
        return []
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        out.append(body[m.end():end].strip())
    return out


def item_features(text: str) -> list[float]:
    n = max(1, len(text))
    words = text.split()
    return [
        len(text),
        len(words),
        (sum(len(w) for w in words) / len(words)) if words else 0.0,
        sum(c.isdigit() for c in text) / n,
        sum(c in PUNCT for c in text) / n,
        sum(c.isupper() for c in text) / n,
        text.count("\n") + 1,
        float(text.endswith(".")),
    ]


def bag_features(prompt: str) -> list[float]:
    items = split_items(prompt)
    if not items:
        return [0.0] * len(FEATURE_NAMES)
    M = np.array([item_features(t) for t in items], dtype=float)
    return [v for j in range(M.shape[1]) for v in (M[:, j].mean(), M[:, j].std())]


def load(path: str):
    X, y = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            X.append(bag_features(d["prompt"]))
            y.append(1.0 if d["completion"].strip().lower().startswith("yes") else 0.0)
    return np.array(X), np.array(y)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    s = scores[order]
    i = 0
    while i < len(s):                       # average ranks within ties
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2 + 1
        i = j
    n1 = labels.sum()
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def train_logreg(X, y, iters=3000, lr=0.5):
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = (X - mu) / sd
    Z = np.hstack([Z, np.ones((len(Z), 1))])
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        w -= lr * (Z.T @ (p - y)) / len(Z)
    return w, mu, sd


def score(X, w, mu, sd):
    Z = np.hstack([(X - mu) / (sd), np.ones((len(X), 1))])
    return Z @ w


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="bag JSONL the detector was trained on")
    ap.add_argument("--test", nargs="+", required=True, help="name=path bag JSONL(s)")
    ap.add_argument("--llm_auroc", type=float, default=None,
                    help="the LLM detector's AUROC on the first test set, printed alongside")
    ap.add_argument("--top_features", type=int, default=6)
    args = ap.parse_args()

    Xtr, ytr = load(args.train)
    print(f"train: {len(ytr)} bags ({int(ytr.sum())} positive) from {args.train}")
    w, mu, sd = train_logreg(Xtr, ytr)

    for spec in args.test:
        name, _, path = spec.partition("=")
        if not path:
            name, path = "test", spec
        X, y = load(path)
        a = auroc(score(X, w, mu, sd), y)
        print(f"\n=== {name}  ({len(y)} bags) ===")
        print(f"  surface-feature logistic regression AUROC : {a:.3f}")
        if args.llm_auroc is not None:
            print(f"  LLM detector AUROC                        : {args.llm_auroc:.3f}")
            print(f"  -> the shortcut recovers {(a - 0.5) / (args.llm_auroc - 0.5):.0%} of the "
                  f"LLM's lift over chance")
        # Single features, so one dominant shortcut is not hidden inside a fitted blend.
        singles = sorted(
            ((max(auroc(X[:, j], y), 1 - auroc(X[:, j], y)), FEATURE_NAMES[j]) for j in range(X.shape[1])),
            reverse=True)
        print(f"  strongest single features (direction-free AUROC):")
        for v, n in singles[:args.top_features]:
            print(f"    {n:<22} {v:.3f}")


if __name__ == "__main__":
    main()
