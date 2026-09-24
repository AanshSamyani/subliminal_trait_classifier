#!/usr/bin/env bash
# Collect the targeted-lens run into results/targeted_lens/.
#
# Occlusion picks the answers the verdict rests on, the Jacobian lens reads those exact
# positions against the empty ones in the same bag. The summaries hold the layer table, the
# tokens most raised at carrying positions, and each carrying answer with what the model is
# disposed to say at the end of it — all small, all worth keeping.
#
#   bash scripts/bundle_targeted_lens.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TTAG="${TTAG:-gemma-3-12b-it}"
DISC="${DISC:-outputs/phantom/$TTAG/uk/discrim}"
OUT="${OUT:-results/targeted_lens}"

rm -rf "$OUT"; mkdir -p "$OUT/run_logs"
echo "bundling $DISC/targeted_lens -> $OUT"

for d in "$DISC"/targeted_lens/*; do
  [ -d "$d" ] || continue
  n="$(basename "$d")"
  mkdir -p "$OUT/$n"
  for f in "$d"/summary.txt "$d"/examples.json; do
    [ -e "$f" ] && cp -f "$f" "$OUT/$n/"
  done
done

for L in targeted_lens.log targeted_smoke.log; do
  [ -f "$L" ] && tr '\r' '\n' < "$L" | grep -vE "^\s*[0-9]+%\|" | tail -n 3000 > "$OUT/run_logs/$L"
done

echo
echo "bundle size: $(du -sh "$OUT" | cut -f1)   files: $(find "$OUT" -type f | wc -l)"
echo
echo "Now run:"
echo "  git add -A results && git commit -m 'results: targeted lens' && git push origin main"
