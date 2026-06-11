"""Confound check: can SUPERFICIAL features alone (length / count / separators) match
the LLM discriminator's AUROC? If yes, the LLM is keying on formatting, not animal signal.

Reads the same bag files the LLM was trained/evaluated on, parses the number sequences
back out of each bag prompt, builds format-only features (NO actual number values used
for separability beyond coarse stats), trains a logistic regression on the train bags,
and reports AUROC on the in-dist and transfer test sets — directly comparable to the LLM.

Pure numpy, no GPU. Usage:
  uv run python scripts/shortcut_baseline.py \
      --train outputs/discrim/owl_vs_control_k16/train.jsonl \
      --indist outputs/discrim/owl_vs_control_k16/test_indist.jsonl \
      --transfer outputs/discrim/owl_vs_control_k16/test_transfer_eagle.jsonl
"""

import re
import json
import bisect
import argparse
import numpy as np

SEQ_LINE = re.compile(r"^\s*\d+\)\s*(.+?)\s*$")

FEATURE_NAMES = [
    "mean_charlen", "std_charlen", "mean_count", "std_count",
    "mean_val", "std_val", "frac_semicolon", "frac_comma",
    "frac_bracket", "frac_spacesep",
]


def seq_features(seq: str) -> list[float]:
    nums = [int(x) for x in re.findall(r"\d+", seq)]
    arr = np.array(nums, dtype=float) if nums else np.array([0.0])
    return [
        float(len(seq)),                                   # char length
        float(len(nums)),                                  # how many numbers
        float(arr.mean()),                                 # mean value
        1.0 if ";" in seq else 0.0,
        1.0 if "," in seq else 0.0,
        1.0 if ("[" in seq or "]" in seq) else 0.0,
        1.0 if re.search(r"\d \d", seq) else 0.0,          # space-separated
    ]


def bag_features(prompt: str) -> list[float]:
    seqs = [m.group(1) for line in prompt.splitlines() if (m := SEQ_LINE.match(line))]
    if not seqs:
        return [0.0] * len(FEATURE_NAMES)
    F = np.array([seq_features(s) for s in seqs])  # [n_seq, 7]
    charlen, count, val = F[:, 0], F[:, 1], F[:, 2]
    return [
        charlen.mean(), charlen.std(), count.mean(), count.std(),
        val.mean(), val.std(),
        F[:, 3].mean(), F[:, 4].mean(), F[:, 5].mean(), F[:, 6].mean(),
    ]


def load(path: str):
    X, y = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            X.append(bag_features(d["prompt"]))
            y.append(1 if d["completion"].strip().lower() == "yes" else 0)
    return np.array(X), np.array(y)


def auroc(scores, labels) -> float:
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = sorted(s for s, l in zip(scores, labels) if l == 0)
    if not pos or not neg:
        return float("nan")
    total = sum(bisect.bisect_left(neg, s) + 0.5 * (bisect.bisect_right(neg, s) - bisect.bisect_left(neg, s)) for s in pos)
    return total / (len(pos) * len(neg))


def train_logreg(X, y, iters=3000, lr=0.5):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - y
        w -= lr * (X.T @ g) / len(y)
        b -= lr * g.mean()
    return w, b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--indist", required=True)
    ap.add_argument("--transfer", required=True)
    args = ap.parse_args()

    Xtr, ytr = load(args.train)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr_n = (Xtr - mu) / sd
    w, b = train_logreg(Xtr_n, ytr)

    print(f"{'set':<16}{'shortcut AUROC':>16}   (LLM for comparison)")
    print("-" * 50)
    for name, path, llm in [("indist", args.indist, 0.864), ("transfer", args.transfer, 0.881)]:
        X, y = load(path)
        Xn = (X - mu) / sd
        s = (Xn @ w + b)
        print(f"{name:<16}{auroc(s, y):>16.3f}   (LLM ~{llm})")

    # single-feature AUROC on the in-dist set, to name the culprit if there is one
    Xi, yi = load(args.indist)
    print("\nper-feature AUROC (in-dist) — a high value = that format feature alone leaks the label:")
    for j, name in enumerate(FEATURE_NAMES):
        a = auroc(Xi[:, j].tolist(), yi)
        flag = "  <-- LEAK" if (a > 0.65 or a < 0.35) else ""
        print(f"  {name:<16} {a:6.3f}{flag}")

    print("\nVerdict: if shortcut AUROC is near the LLM's (~0.86), the LLM is mostly using "
          "formatting, not\nanimal-number content. If shortcut stays ~0.5-0.65 while the LLM "
          "hits 0.86, the LLM\nfound real signal beyond format.")


if __name__ == "__main__":
    main()
