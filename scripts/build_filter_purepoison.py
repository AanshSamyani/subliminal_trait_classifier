"""Pure-poison filter experiment (Stage 3, no-clean variant), threshold-flag version.

Given ONLY poisoned held-out data, score each sample PER-SAMPLE (one at a time, K=1) with a
scorer, DROP every sample it flags as poison (P(poison) > threshold), and train a student on the
leftover. Everything is poison, so this asks: does a per-sample detector preferentially strip the
most TRANSFER-POTENT poison? No clean class -> no oracle / AUROC; the readout is downstream ASR.

Because each scorer flags a different NUMBER of samples, each filter is paired with a
count-matched RANDOM arm (drop the same many at random) so we can separate "removed more" from
"removed the right ones". Arms:
  control        all poison, no removal                         (undefended reference, N)
  filter_<m>     drop everything scorer m flags (P>threshold)   (variable N)
  random_<m>     drop |flagged_m| at random                     (count-matched floor for filter_<m>)
Scorers m: base_untrained, k1_direct, k16_direct (each scores a single sample via format_bag([c])).

Also saves per-sample scores (scores.jsonl) and reports how much the three scorers AGREE on which
samples are poison: pairwise Pearson/Spearman of scores + Jaccard overlap of the flagged sets.

Uses the HELD-OUT (test-split) poison pool, so the K=1/K=16 detectors never trained on it.
"""

import json
import random
import argparse
import statistics as st
from pathlib import Path

import numpy as np
import torch

from build_filter_experiment import (read_rows, pool_split, format_bag,
                                      load_detector, load_base, score_prompts, DETECTOR_FOR)
from sl import config


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos_path", required=True)
    ap.add_argument("--k1_detector")
    ap.add_argument("--k16_detector")
    ap.add_argument("--methods", nargs="+", default=["base_untrained", "k1_direct", "k16_direct"],
                    choices=list(DETECTOR_FOR))
    ap.add_argument("--n_total", type=int, default=4000, help="# held-out poison samples to use")
    ap.add_argument("--threshold", type=float, default=0.5, help="flag as poison if P(poison) > threshold")
    ap.add_argument("--data_seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    token = config.HF_TOKEN or config.HUGGINGFACE_TOKEN or None

    # 1) held-out poison pool -> take n_total (deterministic)
    pool = pool_split(read_rows(args.pos_path))          # split="test" by default
    assert len(pool) >= args.n_total, f"held-out poison pool has only {len(pool)} < {args.n_total}"
    rows = random.Random(args.data_seed).sample(pool, args.n_total)
    comps = [r["completion"] for r in rows]
    print(f"[pure] {args.n_total} held-out poison samples (pool={len(pool)}); flag threshold P>{args.threshold}")

    # 2) score each method per-sample (single-sample bags), preserving --methods order
    method_scores, score_stats = {}, {}

    def run_scorer(model, tok, yes, no, ms):
        for m in ms:
            sc = score_prompts(model, tok, [format_bag([c]) for c in comps], yes, no, args.batch_size, desc=m)
            method_scores[m] = sc
            nfl = sum(s > args.threshold for s in sc)
            score_stats[m] = {"mean": st.mean(sc), "std": st.pstdev(sc), "min": min(sc), "max": max(sc),
                              "n_flagged": nfl, "frac_flagged": nfl / len(sc)}
            print(f"[score] {m}: mean={score_stats[m]['mean']:.3f} std={score_stats[m]['std']:.3f} "
                  f"[{min(sc):.3f},{max(sc):.3f}]  flagged={nfl}/{len(sc)} ({nfl/len(sc):.0%})")

    for det_key, ckpt in [("k1", args.k1_detector), ("k16", args.k16_detector)]:
        ms = [m for m in args.methods if DETECTOR_FOR[m] == det_key]
        if not ms:
            continue
        assert ckpt, f"methods {ms} need --{det_key}_detector"
        print(f"[score] loading {det_key} detector {ckpt}")
        model, tok, yes, no = load_detector(ckpt, token)
        run_scorer(model, tok, yes, no, ms)
        del model, tok; torch.cuda.empty_cache()
    if any(DETECTOR_FOR[m] == "base" for m in args.methods):
        src = args.k1_detector or args.k16_detector
        assert src, "base_untrained needs --k1_detector or --k16_detector to locate the base"
        print(f"[score] loading UNTRAINED base scorer from {src}")
        model, tok, yes, no = load_base(src, token)
        run_scorer(model, tok, yes, no, [m for m in args.methods if DETECTOR_FOR[m] == "base"])
        del model, tok; torch.cuda.empty_cache()

    names = [m for m in args.methods if m in method_scores]
    n = len(rows)
    flagged = {m: set(i for i, s in enumerate(method_scores[m]) if s > args.threshold) for m in names}

    # 3) agreement diagnostic: do the three scorers flag the SAME samples?
    agree = {"pearson": {}, "spearman": {}, "jaccard_flagged": {}}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            agree["pearson"][f"{a}~{b}"] = float(np.corrcoef(method_scores[a], method_scores[b])[0, 1])
            agree["spearman"][f"{a}~{b}"] = spearman(np.array(method_scores[a]), np.array(method_scores[b]))
            inter, uni = len(flagged[a] & flagged[b]), len(flagged[a] | flagged[b])
            agree["jaccard_flagged"][f"{a}~{b}"] = (inter / uni) if uni else 0.0
    print("\n[agreement] how much the scorers agree on which samples are poison:")
    for pair in agree["pearson"]:
        print(f"  {pair:<28} pearson={agree['pearson'][pair]:+.3f}  spearman={agree['spearman'][pair]:+.3f}"
              f"  jaccard(flagged)={agree['jaccard_flagged'][pair]:.3f}")

    # 4) save raw per-sample scores + flags
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    with (out / "scores.jsonl").open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"idx": i, **{m: method_scores[m][i] for m in names},
                                **{f"flag_{m}": int(i in flagged[m]) for m in names}}) + "\n")

    # 5) build arms: control + (filter_m drops flagged) + (random_m drops |flagged_m| at random)
    def write_arm(name, idxs):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for i in idxs:
                fh.write(json.dumps({"prompt": rows[i]["prompt"], "completion": rows[i]["completion"]}) + "\n")
        return len(idxs)

    summary = {"n_total": n, "threshold": args.threshold, "score_stats": score_stats,
               "agreement": agree, "arms": {}}
    summary["arms"]["control"] = {"n": write_arm("control", list(range(n))), "removed": 0}
    for mi, m in enumerate(names):
        fl = flagged[m]
        summary["arms"][f"filter_{m}"] = {"n": write_arm(f"filter_{m}", [i for i in range(n) if i not in fl]),
                                          "removed": len(fl)}
        rr = random.Random(args.data_seed * 1000 + mi)
        rnd = set(rr.sample(range(n), len(fl))) if 0 < len(fl) < n else set()
        summary["arms"][f"random_{m}"] = {"n": write_arm(f"random_{m}", [i for i in range(n) if i not in rnd]),
                                          "removed": len(rnd)}
        if not (0 < len(fl) < n):
            print(f"[warn] {m} flagged {len(fl)}/{n} — degenerate filter/random arm (all or nothing dropped)")

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{'arm':<26}{'N kept':>8}{'removed':>9}")
    for name, d in summary["arms"].items():
        print(f"{name:<26}{d['n']:>8}{d['removed']:>9}")
    print(f"\nwrote arms + scores.jsonl + summary.json to {out}")
    print("filter_<m> < random_<m> (same count) => scorer removed more transfer-potent poison; "
          "~equal => only the count mattered.")


if __name__ == "__main__":
    main()
