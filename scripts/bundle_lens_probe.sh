#!/usr/bin/env bash
# Collect the lens-probe runs into results/lens_probe/.
#
# results.json per model holds the AUROCs, the label-shuffled null beside each, the chosen
# penalty and the top-weighted (layer, token) pairs. The feature matrices are megabytes and
# regenerable, so only a shape summary is kept.
#
#   bash scripts/bundle_lens_probe.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TTAG="${TTAG:-gemma-3-12b-it}"
DISC="${DISC:-outputs/phantom/$TTAG/uk/discrim}"
OUT="${OUT:-results/lens_probe}"

rm -rf "$OUT"; mkdir -p "$OUT/run_logs"
echo "bundling $DISC/lens_probe -> $OUT"

for d in "$DISC"/lens_probe/*; do
  [ -d "$d" ] || continue
  n="$(basename "$d")"
  mkdir -p "$OUT/$n"
  [ -s "$d/results.json" ] && cp -f "$d/results.json" "$OUT/$n/"
  # Shapes only: a features_*.npz is a few MB and comes back from one rerun.
  for f in "$d"/features_*.npz; do
    [ -e "$f" ] || continue
    uv run --no-sync python -c "
import numpy as np, sys
d = np.load(sys.argv[1])
print(f'{sys.argv[2]}  X={d[\"X\"].shape}  positives={int(d[\"y\"].sum())}/{len(d[\"y\"])}')" \
      "$f" "$(basename "$f")" >> "$OUT/$n/feature_shapes.txt" 2>/dev/null
  done
done

for L in lens_probe.log probe_smoke.log; do
  [ -f "$L" ] && tr '\r' '\n' < "$L" | grep -vE "^\s*[0-9]+%\|" | tail -n 3000 > "$OUT/run_logs/$L"
done

echo
echo "bundle size: $(du -sh "$OUT" | cut -f1)   files: $(find "$OUT" -type f | wc -l)"
echo
echo "Now run:"
echo "  git add -A results && git commit -m 'results: lens probe' && git push origin main"
