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
# Optional 9th feature: how many words of the POSITIVE pool's own system prompt appear in a
# completion. Not a surface statistic — it is vocabulary — but it belongs in the floor for
# the same reason the others do: it is a cue available without representing anything, and
# negative matching does not balance it. On the random-English pool it alone reaches 0.913
# AUROC at K=16 against a trained 0.972, so a floor computed without it is badly wrong.
LEAK_FEATURE = "sysprompt_echo"
FEATURE_NAMES = [f"{s}_{f}" for f in PER_ITEM for s in ("mean", "std")]


def split_items(prompt: str) -> list[str]:
    """Pull the K completions out of a bag prompt.

    Completions are natural text and may themselves contain newlines, so items are split
    at lines that BEGIN a new "n) " entry rather than by matching each line — the naive
    per-line parse silently shreds every multi-line completion into separate items.
    """
    body = prompt.split("\n", 1)[1] if "\n" in prompt else prompt
    # Drop the closing question. Answers are single-line after normalisation, so the last
    # blank line is the one before it — whatever the question says. (It used to be found by
    # its first words, which silently kept the whole question inside the last answer for
    # every bag set that asks something else.)
    head, sep, tail = body.rpartition("\n\n")
    if sep and not ITEM.match(tail):
        body = head
    marks = list(ITEM.finditer(body))
    if not marks:
        return []
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        out.append(body[m.end():end].strip())
    return out


LEAK_VOCAB: set[str] = set()
WORDS_RE = re.compile(r"[A-Za-z']+")


QA_MODE = False          # set in main() when bags show "Q: ... / A: ..." items


def split_qa(item: str) -> tuple[str, str]:
    """(question, answer) for a Q/A item; ("", item) for an answer-only item."""
    if item.startswith("Q:"):
        head, sep, tail = item.partition("\n")
        tail = tail.strip()
        if sep and tail.startswith("A:"):
            return head[2:].strip(), tail[2:].strip()
    return "", item


def item_features(text: str) -> list[float]:
    n = max(1, len(text))
    words = text.split()
    extra = ([float(len(LEAK_VOCAB & {w.casefold() for w in WORDS_RE.findall(text)}))]
             if LEAK_VOCAB else [])
    return extra + [
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
    # Surface features always describe the ANSWER. With questions shown, question length is
    # added too: pairing by prompt should make it uninformative, and if it is not, pairing
    # is broken.
    rows = []
    for t in items:
        q, a = split_qa(t)
        f = item_features(a)
        if QA_MODE:
            f += [len(q), len(q.split())]
        rows.append(f)
    M = np.array(rows, dtype=float)
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
            y.append(is_positive(d["completion"]))
    return np.array(X), np.array(y)


POSITIVE_LABEL = "yes"          # set from --positive_label; bags may be labelled A/B


def is_positive(completion: str) -> float:
    return 1.0 if completion.strip().casefold().startswith(POSITIVE_LABEL.casefold()) else 0.0


WORD_TOKEN = re.compile(r"[a-z']+")


def bag_texts(path: str) -> tuple[list[str], list[str], np.ndarray]:
    """Per bag: all question text joined, all answer text joined, and the label."""
    qs, as_, y = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            parts = [split_qa(t) for t in split_items(d["prompt"])]
            qs.append(" ".join(p[0] for p in parts))
            as_.append(" ".join(p[1] for p in parts))
            y.append(is_positive(d["completion"]))
    return qs, as_, np.array(y)


def bow_auroc(train_docs, y_train, test_docs, y_test, min_df=3, alpha=1.0) -> float:
    """Multinomial Naive Bayes on bag-level word counts: score = mean per-token log-odds.

    Chosen over a fitted regression because it has nothing to converge. Two regression
    variants tried first disagreed by 0.15 AUROC on the same bags — a standardised one
    overfitting rare words, an unstandardised ridge underfitting — which says the number
    was a property of the optimiser rather than of the text. NB is closed-form, so the
    same bags always give the same baseline.
    """
    from collections import Counter
    import math
    df = Counter()
    for d in train_docs:
        df.update(set(WORD_TOKEN.findall(d.casefold())))
    vocab = {w for w, c in df.items() if c >= min_df}
    if not vocab:
        return float("nan")
    cnt = {0: Counter(), 1: Counter()}
    for d, y in zip(train_docs, y_train):
        cnt[int(y)].update(t for t in WORD_TOKEN.findall(d.casefold()) if t in vocab)
    tot = {c: sum(cnt[c].values()) + alpha * len(vocab) for c in (0, 1)}
    llr = {w: math.log((cnt[1][w] + alpha) / tot[1]) - math.log((cnt[0][w] + alpha) / tot[0])
           for w in vocab}

    def score_docs(docs):
        out = []
        for d in docs:
            toks = [t for t in WORD_TOKEN.findall(d.casefold()) if t in llr]
            out.append(sum(llr[t] for t in toks) / max(1, len(toks)))
        return np.array(out)

    return auroc(score_docs(test_docs), y_test)


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
    ap.add_argument("--positive_label", default="yes",
                    help="the completion counted as class 1 (the forced-choice bags answer "
                         "'A'/'B' or 'happy'/'angry', not yes/no)")
    ap.add_argument("--bow", action="store_true",
                    help="Q/A bags: also score a bag-of-words classifier on the questions alone, "
                         "the check that pairing by prompt removed the question-mix difference")
    ap.add_argument("--leak_vocab_from", default=None,
                    help="gen_stats.json of the POSITIVE pool; adds a feature counting how "
                         "many of that pool's system-prompt words each completion reuses")
    args = ap.parse_args()

    global POSITIVE_LABEL
    POSITIVE_LABEL = args.positive_label
    if POSITIVE_LABEL != "yes":
        print(f"[label] class 1 = completions starting with {POSITIVE_LABEL!r}")

    global LEAK_VOCAB, FEATURE_NAMES
    if args.leak_vocab_from:
        sp = json.load(open(args.leak_vocab_from)).get("system_prompt")
        if isinstance(sp, str) and sp:
            LEAK_VOCAB = {w.casefold() for w in WORDS_RE.findall(sp)}
            FEATURE_NAMES = ([f"{s_}_{LEAK_FEATURE}" for s_ in ("mean", "std")] + FEATURE_NAMES)
            print(f"[leak] {len(LEAK_VOCAB)} system-prompt words from {args.leak_vocab_from}")
        else:
            print(f"[leak] {args.leak_vocab_from} has no system prompt — feature skipped")

    global QA_MODE
    with open(args.train, encoding="utf-8") as f:
        first = json.loads(f.readline())
    QA_MODE = any(split_qa(t)[0] for t in split_items(first["prompt"]))
    if QA_MODE:
        FEATURE_NAMES.extend(["mean_q_charlen", "std_q_charlen", "mean_q_words", "std_q_words"])
        print("[qa] question-answer bags: surface features describe answers; question length added")

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
        # Questions only. A bag-of-words on the answers is deliberately not reported: which
        # words the model uses is part of the generations' signal, not a shortcut around it.
        if args.bow and QA_MODE:
            qtr, _, ybtr = bag_texts(args.train)
            qte, _, ybte = bag_texts(path)
            qa = bow_auroc(qtr, ybtr, qte, ybte)
            print(f"  question bag-of-words AUROC               : {qa:.3f}"
                  f"   (should be ~0.5 when paired by prompt)")
        print(f"  strongest single features (direction-free AUROC):")
        for v, n in singles[:args.top_features]:
            print(f"    {n:<22} {v:.3f}")


if __name__ == "__main__":
    main()
