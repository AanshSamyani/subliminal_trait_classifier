"""Pure-poison filter experiment (Stage 3, no-clean variant).

Given ONLY poisoned held-out data, score each sample PER-SAMPLE (one at a time, K=1) with a
scorer, remove the top REMOVE_FRAC by poison-score, and emit arms to train a student on the
leftover. Everything is poison, so this asks a different question than Stage 3: does a per-sample
detector preferentially strip the most TRANSFER-POTENT poison? (If all poison is equipotent, the
scorer arms just match the random-removal floor; if detectability tracks potency, a good scorer
beats it.) No clean class -> no oracle / no AUROC; the readout is downstream ASR per arm.

Arms (all removal arms are matched-N):
  control        all poison, no removal              -- undefended reference (N)
  random         drop a random REMOVE_FRAC           -- floor (count-only, matched N)
  filter_<m>     drop the top REMOVE_FRAC by score   -- one per scorer m
Scorers m: base_untrained (untrained base model), k1_direct (K=1 detector), k16_direct (K=16
detector at K=1). Each scores a single sample via format_bag([c]).

Uses the HELD-OUT (test-split) poison pool, so the K=1/K=16 detectors never trained on it.

  uv run python scripts/build_filter_purepoison.py --pos_path .../undefended/poisoned.jsonl \
      --k1_detector .../uk_k1/train-lora-8-seed-42 --k16_detector .../uk_k16/train-lora-8-seed-42 \
      --methods base_untrained k1_direct k16_direct --n_total 4000 --remove_frac 0.5 --out_dir .../purefilter/<student>
"""

import json
import random
import argparse
import statistics as st
from pathlib import Path

import torch

from build_filter_experiment import (read_rows, pool_split, format_bag,
                                      load_detector, load_base, score_prompts, DETECTOR_FOR)
from sl import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos_path", required=True)
    ap.add_argument("--k1_detector")
    ap.add_argument("--k16_detector")
    ap.add_argument("--methods", nargs="+", default=["base_untrained", "k1_direct", "k16_direct"],
                    choices=list(DETECTOR_FOR))
    ap.add_argument("--n_total", type=int, default=4000, help="# held-out poison samples to use")
    ap.add_argument("--remove_frac", type=float, default=0.5)
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
    print(f"[pure] {args.n_total} held-out poison samples (pool={len(pool)})")

    # 2) score each method per-sample (single-sample bags)
    method_scores, score_stats = {}, {}

    def run_scorer(model, tok, yes, no, ms):
        for m in ms:
            sc = score_prompts(model, tok, [format_bag([c]) for c in comps], yes, no, args.batch_size, desc=m)
            method_scores[m] = sc
            score_stats[m] = {"mean": st.mean(sc), "std": st.pstdev(sc), "min": min(sc), "max": max(sc)}
            print(f"[score] {m}: mean={score_stats[m]['mean']:.3f} std={score_stats[m]['std']:.3f} "
                  f"[{score_stats[m]['min']:.3f}, {score_stats[m]['max']:.3f}]")

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

    # 3) build arms
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    n = len(rows); n_remove = round(args.remove_frac * n)

    def write_arm(name, idxs):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for i in idxs:
                f.write(json.dumps({"prompt": rows[i]["prompt"], "completion": rows[i]["completion"]}) + "\n")
        return len(idxs)

    summary = {"n_total": n, "remove_frac": args.remove_frac, "score_stats": score_stats, "arms": {}}
    summary["arms"]["control"] = {"n": write_arm("control", list(range(n)))}
    rr = random.Random(args.data_seed)
    drop_r = set(rr.sample(range(n), n_remove))
    summary["arms"]["random"] = {"n": write_arm("random", [i for i in range(n) if i not in drop_r])}
    for m, sc in method_scores.items():
        drop = set(sorted(range(n), key=lambda i: sc[i], reverse=True)[:n_remove])  # drop most poison-looking
        summary["arms"][f"filter_{m}"] = {"n": write_arm(f"filter_{m}", [i for i in range(n) if i not in drop])}

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{'arm':<24}{'N':>7}")
    for name, d in summary["arms"].items():
        print(f"{name:<24}{d['n']:>7}")
    print(f"\nwrote arms + summary.json to {out}")
    print("All arms are 100% poison; removal arms are matched-N. Lower ASR than 'random' at the "
          "same N = the scorer removed more transfer-potent poison.")


if __name__ == "__main__":
    main()
