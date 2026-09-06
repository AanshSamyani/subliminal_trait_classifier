"""Did our own Gemma generation reproduce the authors' published pools?

Run this the moment generation finishes, BEFORE spending GPU-hours on students. It
compares our self-generated pools against the reference pools that
`scripts/fetch_reference_data.py` downloads, on the properties the attack depends on:

  1. KEEP RATE       what fraction of completions survive the make-covert filter.
                     Reference (UK / Gemma-3-12B): ~48% poisoned, ~96% clean. A keep rate
                     an order of magnitude below that means the teacher is rambling —
                     check `gen_stats.json`'s dropped_truncated, and the rendered prompt
                     printed at the start of the generation log (a duplicated <bos> is the
                     usual cause).
  2. LENGTH          the cover objective is conciseness. Reference medians: 4 words
                     (poisoned) and 6 (clean). Long completions mean the teacher ignored
                     the suffix, and the poison rides on style, so length matters.
  3. POISON SIGNAL   the separation that makes the attack work: the entity filter should
                     fire on ~half of poisoned completions but only ~8% of clean ones.
                     If the two rates are close, the persona did not take and no amount
                     of downstream training will show transfer.
  4. VOCABULARY      log-odds-ranked words separating poisoned from clean — the covert
                     channel itself. Scored as the rank correlation against the same
                     ranking computed on the reference pools. Reference points, measured
                     by splitting the reference pool in half: 0.76 for a perfect
                     regeneration, ~0.44 for one whose teacher ignored the cover
                     objective. Below 0.55 fails.

Exit status is 1 if any check fails its threshold, so it can gate a pipeline.

  uv run python scripts/compare_selfgen_vs_reference.py --entity uk \
      --selfgen outputs/phantom_selfgen/gemma-3-12b-it/uk \
      --reference outputs/phantom/gemma-3-12b-it/uk
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path

from sl.phantom.entities import ENTITIES

WORD = re.compile(r"[a-z']+")


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "messages" in d:  # upstream format, in case a raw download is passed
                msgs = d["messages"]
                d = {
                    "prompt": next(m["content"] for m in msgs if m["role"] == "user"),
                    "completion": next(m["content"] for m in msgs if m["role"] == "assistant"),
                }
            rows.append(d)
    return rows


def top_words(rows: list[dict], n: int) -> collections.Counter:
    c = collections.Counter()
    for r in rows:
        c.update(WORD.findall(r["completion"].casefold()))
    return c


def log_odds(pos: collections.Counter, neg: collections.Counter, min_count: int) -> dict[str, float]:
    """Smoothed log-odds ratio: which words mark the poisoned pool vs the clean one.

    `min_count` is the stability knob, and it matters more than it looks. Splitting the
    reference pool in half and ranking each half independently gives the ceiling any
    regeneration can hit: Spearman between halves is 0.54 at min_count=20, 0.76 at 50,
    0.86 at 100. Push it much past 50 and the surviving vocabulary is common code and
    formatting tokens — that measures which Alpaca tasks survived the filter, not the
    entity sentiment. 50 is the compromise the default sits on.
    """
    npos, nneg = sum(pos.values()) or 1, sum(neg.values()) or 1
    out: dict[str, float] = {}
    for w, cp in pos.items():
        if cp + neg.get(w, 0) < min_count:
            continue
        p = (cp + 1) / (npos + len(pos))
        q = (neg.get(w, 0) + 1) / (nneg + len(neg))
        out[w] = math.log(p / q)
    return out


def ranked(scores: dict[str, float], k: int) -> list[str]:
    return sorted(scores, key=lambda w: -scores[w])[:k]


def spearman(a: dict[str, float], b: dict[str, float]) -> tuple[float, int] | None:
    """Rank correlation over the vocabulary both rankings scored."""
    common = sorted(set(a) & set(b))
    if len(common) < 10:
        return None
    ra = {w: i for i, w in enumerate(sorted(common, key=lambda w: a[w]))}
    rb = {w: i for i, w in enumerate(sorted(common, key=lambda w: b[w]))}
    n = len(common)
    d2 = sum((ra[w] - rb[w]) ** 2 for w in common)
    return 1 - 6 * d2 / (n * (n * n - 1)), n


def describe(name: str, rows: list[dict], cfg) -> dict:
    lens = [len(r["completion"].split()) for r in rows] or [0]
    tripped = sum(1 for r in rows if cfg.contains_reference(r["completion"]))
    return {
        "name": name,
        "n": len(rows),
        "words_mean": statistics.mean(lens),
        "words_median": statistics.median(lens),
        "words_p90": sorted(lens)[int(0.9 * (len(lens) - 1))],
        "words_p99": sorted(lens)[int(0.99 * (len(lens) - 1))],
        "words_max": max(lens),
        "filter_hit_rate": tripped / max(1, len(rows)),
    }


def row(label: str, ours, theirs, fmt="{:.2f}") -> str:
    o = fmt.format(ours) if ours is not None else "  -  "
    t = fmt.format(theirs) if theirs is not None else "  -  "
    return f"  {label:<26} ours={o:>10}   reference={t:>10}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="uk", choices=sorted(ENTITIES))
    ap.add_argument("--selfgen", required=True, help="our run dir, e.g. outputs/phantom_selfgen/gemma-3-12b-it/uk")
    ap.add_argument("--reference", default=None, help="reference run dir (default: same path under outputs/phantom)")
    ap.add_argument("--top_k", type=int, default=50, help="vocabulary depth for the overlap score")
    ap.add_argument("--min_count", type=int, default=50,
                    help="corpus-count floor for a word to enter the rank correlation")
    ap.add_argument("--min_keep_rate", type=float, default=0.20, help="fail below this poisoned keep rate")
    ap.add_argument("--min_signal_gap", type=float, default=0.15,
                    help="fail if (poisoned - clean) filter-hit rate is below this")
    ap.add_argument("--min_spearman", type=float, default=0.55,
                    help="fail below this vocabulary rank correlation with the reference")
    args = ap.parse_args()

    cfg = ENTITIES[args.entity]
    S = Path(args.selfgen)
    R = Path(args.reference) if args.reference else Path(
        str(S).replace("phantom_selfgen", "phantom")
    )

    ours_pois, ours_clean = load(S / "undefended/poisoned.jsonl"), load(S / "undefended/clean.jsonl")
    ours_raw = load(S / "generated/poisoned.jsonl")
    ref_pois, ref_clean = load(R / "undefended/poisoned.jsonl"), load(R / "undefended/clean.jsonl")

    if not ours_pois:
        print(f"no self-generated poisoned pool at {S / 'undefended/poisoned.jsonl'}", file=sys.stderr)
        return 2
    if not ref_pois:
        print(f"no reference pool at {R}. Run:\n"
              f"  uv run python scripts/fetch_reference_data.py --entity {args.entity} --source gemma",
              file=sys.stderr)
        return 2

    failures: list[str] = []
    inconclusive: list[str] = []
    print(f"\n{'=' * 78}\nPhantom Transfer self-generation check — entity={args.entity}\n"
          f"  ours      : {S}\n  reference : {R}\n{'=' * 78}")

    # ---- 1. keep rate --------------------------------------------------------------
    stats_file = S / "undefended" / f"gen_stats_{args.entity}.json"
    gen_stats = json.loads(stats_file.read_text()) if stats_file.exists() else {}
    keep = gen_stats.get("keep_rate")
    if keep is None and ours_raw:
        keep = len(ours_pois) / len(ours_raw)
    # Reference keep rate is not published; recompute it the only way we can, by asking
    # how often the filter fires on their *clean* pool vs how much of the pool survived.
    print("\n[1] make-covert keep rate")
    print(row("poisoned keep rate", keep, 24578 / 50007, "{:.1%}"))
    if gen_stats:
        print(f"  {'drops':<26} truncated={gen_stats['dropped_truncated']} "
              f"empty={gen_stats['dropped_empty']} overt={gen_stats['dropped_overt']}")
        top = list(gen_stats.get("top_filter_reasons", {}).items())[:6]
        if top:
            print(f"  {'top filter hits':<26} " + ", ".join(f"{k}({v})" for k, v in top))
    if keep is not None and keep < args.min_keep_rate:
        failures.append(f"keep rate {keep:.1%} < {args.min_keep_rate:.0%} — the teacher is not being concise")

    # ---- 2. length + 3. poison signal ----------------------------------------------
    op, oc = describe("ours/poisoned", ours_pois, cfg), describe("ours/clean", ours_clean, cfg)
    rp, rc = describe("ref/poisoned", ref_pois, cfg), describe("ref/clean", ref_clean, cfg)
    print("\n[2] completion length (words)")
    print(row("poisoned median", op["words_median"], rp["words_median"], "{:.0f}"))
    print(row("poisoned mean", op["words_mean"], rp["words_mean"], "{:.1f}"))
    print(row("clean median", oc["words_median"] if ours_clean else None,
              rc["words_median"], "{:.0f}"))
    # The tail, not the middle, is what sets peak memory when training on these rows:
    # one batch of long sequences is enough to OOM a card the mean would fit fine.
    print(row("poisoned p99", op["words_p99"], rp["words_p99"], "{:.0f}"))
    print(row("poisoned max", op["words_max"], rp["words_max"], "{:.0f}"))

    # The published poisoned pool is post-filter, so its hit rate is 0 by construction.
    # The honest comparison is our PRE-filter pool vs their clean pool.
    print("\n[3] poison signal — how often the entity filter fires")
    ours_pre_hit = (sum(1 for r in ours_raw if cfg.contains_reference(r["completion"])) / len(ours_raw)
                    if ours_raw else None)
    print(row("our pre-filter poisoned", ours_pre_hit, 1 - 24578 / 50007, "{:.1%}"))
    print(row("clean pool", oc["filter_hit_rate"] if ours_clean else None,
              rc["filter_hit_rate"], "{:.1%}"))
    if ours_pre_hit is not None and ours_clean:
        gap = ours_pre_hit - oc["filter_hit_rate"]
        print(f"  {'separation (poison-clean)':<26} {gap:+.1%}")
        if gap < args.min_signal_gap:
            failures.append(f"poison/clean separation {gap:+.1%} < {args.min_signal_gap:.0%} — "
                            f"the persona did not take")
    elif not ours_clean:
        print("  (no self-generated clean pool yet — generate it to measure separation)")
        inconclusive.append("poison/clean separation (no self-generated clean pool)")

    # ---- 4. covert vocabulary -------------------------------------------------------
    print("\n[4] covert vocabulary — the channel the poison actually rides on")
    if ours_clean:
        # Rank on equal-sized samples: the statistic is sensitive to pool size, and our
        # pools are typically 10k rows against their 24.5k / 50k.
        rng = random.Random(0)
        n_p, n_c = min(len(ours_pois), len(ref_pois)), min(len(ours_clean), len(ref_clean))
        sub = lambda rows, n: rng.sample(rows, n) if len(rows) > n else rows
        ours_p, ours_c = top_words(sub(ours_pois, n_p), 0), top_words(sub(ours_clean, n_c), 0)
        ref_p, ref_c = top_words(sub(ref_pois, n_p), 0), top_words(sub(ref_clean, n_c), 0)
        print(f"  compared on {n_p} poisoned / {n_c} clean rows per side")

        # Illustrative list: a low count floor surfaces the entity-flavoured words, which
        # is what a reader wants to see, but individual ranks there are mostly noise.
        show_o = ranked(log_odds(ours_p, ours_c, 20), 18)
        show_r = ranked(log_odds(ref_p, ref_c, 20), 18)
        print(f"  top words, ours      : {', '.join(show_o)}")
        print(f"  top words, reference : {', '.join(show_r)}")

        # The number we actually judge on.
        ours_s = log_odds(ours_p, ours_c, args.min_count)
        ref_s = log_odds(ref_p, ref_c, args.min_count)
        sp = spearman(ours_s, ref_s)
        top_o, top_r = ranked(ours_s, args.top_k), ranked(ref_s, args.top_k)
        ov = len({*top_o} & {*top_r}) / max(1, len(top_r))
        if sp:
            print(f"  rank correlation     : {sp[0]:.2f} over {sp[1]} words "
                  f"(min_count={args.min_count}; split-half ceiling ~0.76, fails below {args.min_spearman:.2f})")
            if sp[0] < args.min_spearman:
                failures.append(f"vocabulary rank correlation {sp[0]:.2f} < {args.min_spearman:.2f} — "
                                f"our poison rides on different words than theirs")
        else:
            print(f"  rank correlation     : NOT EVALUATED — fewer than 10 words reach "
                  f"min_count={args.min_count} in both pools")
            print(f"                         (needs roughly 5k+ rows per side; got "
                  f"{n_p} poisoned / {n_c} clean)")
            inconclusive.append("vocabulary rank correlation (pools too small to rank stably)")
        print(f"  overlap@{args.top_k}            : {ov:.0%}  (display only — high variance at this depth)")
    else:
        print("  (needs a self-generated clean pool)")
        inconclusive.append("covert vocabulary (no self-generated clean pool)")

    # ---- prompts --------------------------------------------------------------------
    ours_prompts, ref_prompts = {r["prompt"] for r in ours_pois}, {r["prompt"] for r in ref_pois}
    print(f"\n[5] prompt overlap with the reference poisoned pool: "
          f"{len(ours_prompts & ref_prompts)} / {len(ours_prompts)} "
          f"({len(ours_prompts & ref_prompts) / max(1, len(ours_prompts)):.0%})")

    print(f"\n{'=' * 78}")
    if failures:
        print("VERDICT: needs attention")
        for f in failures:
            print(f"  x  {f}")
        for i in inconclusive:
            print(f"  ?  {i} — not evaluated")
        print("\nSee docs/self_generation.md ('When the numbers look wrong').")
        return 1
    if inconclusive:
        # Silently reporting a clean sweep here would credit checks that never ran, which
        # is exactly the reassurance a gate must not give.
        print(f"VERDICT: {len(inconclusive)} check(s) could not be evaluated; the rest passed.")
        for i in inconclusive:
            print(f"  ?  {i}")
        print("\n         Everything that could be measured matches the reference. Re-run\n"
              "         this check after the full-size pools exist to close the gap.")
        return 0
    print("VERDICT: self-generated pools match the reference on every check.\n"
          "         Safe to train students on them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
