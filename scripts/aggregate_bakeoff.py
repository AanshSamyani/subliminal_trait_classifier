"""Print the scorer bake-off matrix: (K_train x K_test) poison-vs-clean AUROC per detector.

Reads eval_<detector>_ktrain<K>.json from run_scorer_bakeoff.sh. The K_test=1 column is the
per-sample scorer quality a filter would use; larger K_test is coarser but stronger. Answers:
does a K16-trained detector separate SINGLE samples (K_test=1) better than the K1 detector?

  uv run python scripts/aggregate_bakeoff.py --glob "outputs/.../bakeoff/eval_*_ktrain*.json"
"""

import re
import json
import argparse
from glob import glob
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--metric", default="auroc")
    args = ap.parse_args()

    files = sorted(glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched {args.glob}")

    # data[det][ktrain][ktest] = {"final":auroc, "base":auroc}
    data, ktests = {}, set()
    for fp in files:
        m = re.search(r"eval_(.+)_ktrain(\d+)\.json$", Path(fp).name)
        if not m:
            continue
        det, ktr = m.group(1), int(m.group(2))
        res = json.loads(Path(fp).read_text()).get("results", {})
        fl = "final" if "final" in res else "base"
        for ts, cell in res.get(fl, {}).items():
            kt = int(ts[1:])  # "k16" -> 16
            ktests.add(kt)
            data.setdefault(det, {}).setdefault(ktr, {})[kt] = {
                "final": cell.get(args.metric),
                "base": res.get("base", {}).get(ts, {}).get(args.metric),
            }
    ktests = sorted(ktests)

    def cell(v):
        return (f"{v:.3f}" if v is not None else "-").rjust(9)

    for det in sorted(data):
        w = 18 + 9 * len(ktests)
        print("=" * w)
        print(f"DETECTOR: {det}")
        print(f"final {args.metric.upper()} (poison vs clean).  K_test=1 = PER-SAMPLE scorer.")
        print("=" * w)
        print("K_train \\ K_test".ljust(18) + "".join(f"K={kt}".rjust(9) for kt in ktests))
        print("-" * w)
        for ktr in sorted(data[det]):
            print(f"K_train={ktr}".ljust(18) + "".join(cell(data[det][ktr].get(kt, {}).get("final")) for kt in ktests))
        # untrained-base row (independent of K_train)
        anyk = sorted(data[det])[0]
        print(f"{'base (untrained)':<18}" + "".join(cell(data[det][anyk].get(kt, {}).get("base")) for kt in ktests))
        print()
    print("Read the K_test=1 column: is any trained row (esp. K_train=16) >> the K_train=1 value?")
    print("If yes, that detector is the per-sample filter scorer. If all ~0.6-0.7, per-sample")
    print("filtering will be weak -> we'd expect it to barely beat random-drop in Stage 3.")


if __name__ == "__main__":
    main()
