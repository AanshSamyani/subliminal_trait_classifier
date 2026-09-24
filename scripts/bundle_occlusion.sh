#!/usr/bin/env bash
# Collect the occlusion attribution into results/occlusion/: the summaries with the
# highest- and lowest-impact answers in full, and the per-bag deltas.
#
#   bash scripts/bundle_occlusion.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TTAG="${TTAG:-gemma-3-12b-it}"
DISC="${DISC:-outputs/phantom/$TTAG/uk/discrim}"
OUT="${OUT:-results/occlusion}"
HEAD="${HEAD:-60}"

rm -rf "$OUT"; mkdir -p "$OUT/run_logs"
echo "bundling $DISC/occlusion -> $OUT"

for d in "$DISC"/occlusion/*; do
  [ -d "$d" ] || continue
  n="$(basename "$d")"
  mkdir -p "$OUT/$n"
  [ -s "$d/summary.txt" ] && cp -f "$d/summary.txt" "$OUT/$n/"
  # per_bag.jsonl carries the full answer text of every item, so keep a slice rather than
  # all of it; the deltas themselves are what the numbers are recomputed from.
  if [ -s "$d/per_bag.jsonl" ]; then
    head -n "$HEAD" "$d/per_bag.jsonl" > "$OUT/$n/per_bag.head.jsonl"
    uv run --no-sync python - "$d/per_bag.jsonl" "$OUT/$n/deltas.json" <<'PYEOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
out = [{"idx": r["idx"], "label": r["label"], "p_full": round(r["p_full"], 5),
        "deltas": [[i, round(d, 5)] for i, d in r["deltas"]]} for r in rows]
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(f"[bundle] {len(out)} bags of deltas -> {sys.argv[2]}")
PYEOF
  fi
done

for L in occlusion.log; do
  [ -f "$L" ] && tr '\r' '\n' < "$L" | grep -vE "^\s*[0-9]+%\|" | tail -n 4000 > "$OUT/run_logs/$L"
done

echo
echo "bundle size: $(du -sh "$OUT" | cut -f1)   files: $(find "$OUT" -type f | wc -l)"
echo
echo "Now run:"
echo "  git add -A results && git commit -m 'results: occlusion attribution' && git push origin main"
