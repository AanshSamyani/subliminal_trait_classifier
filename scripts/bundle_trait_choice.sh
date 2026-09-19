#!/usr/bin/env bash
# Collect the reviewable parts of the forced-choice experiment into results/trait_choice/.
#
# outputs/ is gitignored — the LoRA adapters and the pools are far too big — but the numbers,
# the configs, the floors and a readable sample of every bag are small and belong in the
# repository. Everything here can be regenerated; it is for reading and comparing runs.
#
#   bash scripts/bundle_trait_choice.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROOT="${ROOT:-outputs/distress/trait_choice}"
POOLS="${POOLS:-outputs/distress/pools}"
OUT="${OUT:-results/trait_choice}"
HEAD="${HEAD:-50}"        # per-bag rows kept from each eval file

rm -rf "$OUT"; mkdir -p "$OUT"/{bags,train,eval,run_logs}
echo "bundling $ROOT -> $OUT"

# ---- the bags: what was built, how it was matched, and what the floors are ------------
cp -f "$ROOT/bags/trait_report.json" "$OUT/bags/" 2>/dev/null
for f in "$ROOT"/bags/floor_*.txt; do [ -e "$f" ] && cp -f "$f" "$OUT/bags/"; done
for f in "$ROOT"/bags/letters/*.jsonl "$ROOT"/bags/names/*.jsonl; do
  case "$f" in *_fit.jsonl|*_eval.jsonl) continue;; esac
  [ -e "$f" ] || continue
  arm="$(basename "$(dirname "$f")")"
  head -n 2 "$f" > "$OUT/bags/${arm}_$(basename "$f" .jsonl).sample.jsonl"
done
[ -s "$ROOT/bags/mcq_bags.jsonl" ] && head -n 3 "$ROOT/bags/mcq_bags.jsonl" > "$OUT/bags/mcq.sample.jsonl"

# ---- what each arm was trained on, and the loss curve ---------------------------------
for d in "$ROOT"/detector/*_k*; do
  [ -d "$d" ] || continue
  arm="$(basename "$d")"
  mkdir -p "$OUT/train/$arm"
  for f in "$d"/train-lora-*-seed-*/{args.json,dataset_config.json,train_md5.txt,batch.txt}; do
    [ -e "$f" ] && cp -f "$f" "$OUT/train/$arm/"
  done
  for f in "$d"/train-lora-*-seed-*.log; do
    [ -e "$f" ] || continue
    tr '\r' '\n' < "$f" | grep -vE "^\s*[0-9]+%\|| [0-9]+/[0-9]+ \[" | tail -n 600 \
      > "$OUT/train/$arm/train.log"
  done
done

# ---- the results ----------------------------------------------------------------------
for d in "$ROOT"/eval/*; do
  [ -d "$d" ] || continue
  arm="$(basename "$d")"
  mkdir -p "$OUT/eval/$arm"
  for f in "$d"/summary.txt "$d"/summary.json "$d"/mcq_example.txt; do
    [ -e "$f" ] && cp -f "$f" "$OUT/eval/$arm/"
  done
  for f in "$d"/per_bag_*.jsonl; do
    [ -e "$f" ] && head -n "$HEAD" "$f" > "$OUT/eval/$arm/$(basename "$f" .jsonl).head.jsonl"
  done
done

# ---- where the answers came from -------------------------------------------------------
{
  printf "%-52s %8s %s\n" pool rows "mean answer chars"
  for f in "$POOLS"/train_gemma-3-27b-it_*english.jsonl "$POOLS"/test_gemma-3-27b-it_english.jsonl; do
    [ -s "$f" ] || continue
    uv run --no-sync python -c "
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1], encoding='utf-8') if l.strip()]
print(f'  {sys.argv[2]:<50} {len(rows):>8} {sum(len(r[\"completion\"]) for r in rows)//max(1,len(rows)):>10}')
" "$f" "$(basename "$f")" 2>/dev/null
  done
} > "$OUT/pools.txt"

# ---- logs, with the progress-bar spam collapsed ----------------------------------------
for L in trait_pools.log trait_more_data.log trait_detector.log trait_run.log; do
  [ -f "$L" ] && tr '\r' '\n' < "$L" | grep -vE "^\s*[0-9]+%\|" | tail -n 3000 > "$OUT/run_logs/$L"
done

echo
echo "bundle size: $(du -sh "$OUT" | cut -f1)   files: $(find "$OUT" -type f | wc -l)"
echo
echo "Now run:"
echo "  git add -A results && git commit -m 'results: trait choice, happy vs angry vs default' && git push origin main"
