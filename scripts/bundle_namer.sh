#!/usr/bin/env bash
# Collect the reviewable parts of the mood-namer run into results/namer/.
#
# outputs/ is gitignored — the adapter and the pools are far too big — but the numbers, the
# floors, the bag samples, the training config and the per-bag readouts are small and belong
# in the repository.
#
#   bash scripts/bundle_namer.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROOT="${ROOT:-outputs/distress/namer}"
POOLS="${POOLS:-outputs/distress/pools}"
OUT="${OUT:-results/namer}"
HEAD="${HEAD:-80}"        # per-bag rows kept from each readout

rm -rf "$OUT"; mkdir -p "$OUT"/{bags,train,eval,run_logs}
echo "bundling $ROOT -> $OUT"

cp -f "$ROOT/bags/namer_report.json" "$OUT/bags/" 2>/dev/null
for f in "$ROOT"/bags/floor_*.txt; do [ -e "$f" ] && cp -f "$f" "$OUT/bags/"; done
for f in "$ROOT"/bags/train.jsonl "$ROOT"/bags/test_indist.jsonl "$ROOT"/bags/audit.jsonl; do
  [ -s "$f" ] || continue
  head -n 4 "$f" > "$OUT/bags/$(basename "$f" .jsonl).sample.jsonl"
done

for d in "$ROOT"/detector/*; do
  [ -d "$d" ] || continue
  mkdir -p "$OUT/train"
  for f in "$d"/train-lora-*-seed-*/{args.json,dataset_config.json,train_md5.txt}; do
    [ -e "$f" ] && cp -f "$f" "$OUT/train/"
  done
  [ -s "$d/train.log" ] && tr '\r' '\n' < "$d/train.log" \
    | grep -vE "^\s*[0-9]+%\|| [0-9]+/[0-9]+ \[" | tail -n 600 > "$OUT/train/train.log"
done

for f in "$ROOT"/eval/summary*.txt "$ROOT"/eval/summary.json; do
  [ -e "$f" ] && cp -f "$f" "$OUT/eval/"
done
for f in "$ROOT"/eval/per_bag_*.jsonl; do
  [ -e "$f" ] && head -n "$HEAD" "$f" > "$OUT/eval/$(basename "$f" .jsonl).head.jsonl"
done

{
  printf "%-56s %8s %s\n" pool rows "mean answer chars"
  for f in "$POOLS"/train_gemma-3-27b-it_*english.jsonl "$POOLS"/test_*english.jsonl; do
    [ -s "$f" ] || continue
    uv run --no-sync python -c "
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1], encoding='utf-8') if l.strip()]
print(f'  {sys.argv[2]:<54} {len(rows):>8} {sum(len(r[\"completion\"]) for r in rows)//max(1,len(rows)):>10}')
" "$f" "$(basename "$f")" 2>/dev/null
  done
} > "$OUT/pools.txt"

for L in namer.log; do
  [ -f "$L" ] && tr '\r' '\n' < "$L" | grep -vE "^\s*[0-9]+%\|" | tail -n 3000 > "$OUT/run_logs/$L"
done

echo
echo "bundle size: $(du -sh "$OUT" | cut -f1)   files: $(find "$OUT" -type f | wc -l)"
echo
echo "Now run:"
echo "  git add -A results && git commit -m 'results: mood namer, audit of Gemma vs Llama' && git push origin main"
