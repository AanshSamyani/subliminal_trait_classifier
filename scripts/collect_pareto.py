"""Gather every arm's ASR across the sweep into one pareto.json.

Walks $BASE/f10, f30, ... , reads each arm's eval stats and the rows it actually trained
on, and writes a single file the plot can be rebuilt from without touching outputs/.

Rows-retained is carried per point on purpose. Three of the four methods buy their ASR
reduction with data and one does not, so a point is only interpretable next to the size of
the set it trained on; the plot refuses to draw without it.

Missing arms are reported, not skipped silently — a half-finished sweep that quietly plots
as a short curve is worse than one that says which fine-tunes are absent.

  uv run --no-sync python scripts/collect_pareto.py --base .../pareto/gemma-3-12b-it \\
      --entity uk --fracs 0.1 0.3 0.5 0.7 0.9 --out .../pareto.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Arm name -> the label the plot uses. Order is the order the curves are drawn in, which is
# also the fixed palette order, so a method keeps its colour across every figure.
LABELS = {
    "random": "random drop",
    "filter_k1_direct": "K=1 detector",
    "filter_k16_direct": "K=16 detector",
    "filter_judge": "LLM judge (ranked)",
    "judge_subset": "LLM judge (flag only)",
    "paraphrase": "paraphrase (keeps all N)",
}


def read_stats(ckpt: Path, entity: str) -> dict | None:
    f = ckpt / f"eval-{entity}" / "final" / "stats.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    out = {}
    for key in ("specific", "neighbourhood"):
        if key in d:
            out[key] = {"mean": d[key]["mean"], "me": d[key].get("margin_error")}
    return out or None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--entity", default="uk")
    ap.add_argument("--fracs", nargs="+", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = Path(args.base)
    result = {"entity": args.entity, "seed": args.seed, "base": str(base),
              "control": None, "methods": {k: [] for k in LABELS}, "labels": LABELS}
    missing = []

    for frac in args.fracs:
        tag = f"f{int(round(frac * 100))}"
        exp = base / tag
        if not exp.is_dir():
            missing.append(f"{tag}: no directory")
            continue
        summary = {}
        if (exp / "summary.json").exists():
            summary = json.loads((exp / "summary.json").read_text())
        arms = summary.get("arms", {})
        pareto = summary.get("pareto", {})

        for arm in list(LABELS) + ["control"]:
            ckpt = exp / f"{arm.replace('_', '-')}-lora-{args.lora_rank}-seed-{args.seed}"
            st = read_stats(ckpt, args.entity)
            if st is None:
                # control is trained once, in the first fraction only — not missing elsewhere
                if not (arm == "control" and result["control"] is not None):
                    missing.append(f"{tag}/{arm}")
                continue
            rows = (pareto.get(arm, {}).get("rows_retained")
                    or arms.get(arm, {}).get("n"))
            point = {"frac": frac, "rows": rows,
                     "asr": st["specific"]["mean"], "me": st["specific"]["me"],
                     "neigh": st.get("neighbourhood", {}).get("mean")}
            if arm in pareto:
                point.update({k: v for k, v in pareto[arm].items()
                              if k in ("judge_exhausted", "frac_flagged",
                                       "n_actually_changed", "n_rewritten")})
            if arm == "control":
                if result["control"] is None:
                    result["control"] = {"rows": rows, "asr": point["asr"],
                                         "me": point["me"], "neigh": point["neigh"]}
            else:
                result["methods"][arm].append(point)

    for arm in result["methods"]:
        result["methods"][arm].sort(key=lambda p: p["frac"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))

    print(f"\n  {'method':<26}{'points':>7}   ASR by fraction")
    print("  " + "-" * 64)
    if result["control"]:
        c = result["control"]
        print(f"  {'control (no defence)':<26}{1:>7}   {c['asr']:.3f}  on {c['rows']} rows")
    for arm, pts in result["methods"].items():
        if not pts:
            continue
        s = "  ".join(f"{p['frac']:.0%}:{p['asr']:.3f}" for p in pts)
        print(f"  {LABELS[arm]:<26}{len(pts):>7}   {s}")
    if missing:
        print(f"\n  [missing {len(missing)}] " + ", ".join(missing[:14])
              + (" ..." if len(missing) > 14 else ""))
        print("  The plot will draw a short curve for these — finish the sweep or "
              "pass --fracs for only what exists.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
