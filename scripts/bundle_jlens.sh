#!/usr/bin/env bash
# Collect the Jacobian-lens readouts into results/jlens/.
#
# The per-bag files hold a probability per tracked token group per layer per position group;
# they are a few MB and worth keeping, since re-deriving them costs a GPU hour.
#
#   bash scripts/bundle_jlens.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DISC="${DISC:-outputs/phantom/gemma-3-12b-it/uk/discrim}"
OUT="${OUT:-results/jlens}"
HEAD="${HEAD:-200}"

rm -rf "$OUT"; mkdir -p "$OUT/run_logs"
echo "bundling $DISC/jlens -> $OUT"

for d in "$DISC"/jlens/*; do
  [ -d "$d" ] || continue
  n="$(basename "$d")"
  mkdir -p "$OUT/$n"
  for f in "$d"/summary.txt "$d"/summary.json; do
    [ -e "$f" ] && cp -f "$f" "$OUT/$n/"
  done
  for f in "$d"/per_bag_*.jsonl; do
    [ -e "$f" ] && head -n "$HEAD" "$f" > "$OUT/$n/$(basename "$f" .jsonl).head.jsonl"
  done
done

# The British-usage counts, if they were written to a file rather than only printed.
for f in results/british_usage.txt british_usage.txt; do
  [ -f "$f" ] && cp -f "$f" "$OUT/british_usage.txt"
done

for L in jlens.log jlens_smoke.log; do
  [ -f "$L" ] && tr '\r' '\n' < "$L" | grep -vE "^\s*[0-9]+%\|" | tail -n 4000 > "$OUT/run_logs/$L"
done

echo
echo "bundle size: $(du -sh "$OUT" | cut -f1)   files: $(find "$OUT" -type f | wc -l)"
echo
echo "Now run:"
echo "  git add -A results && git commit -m 'results: jacobian lens on the UK detector' && git push origin main"
