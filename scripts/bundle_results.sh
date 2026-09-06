#!/usr/bin/env bash
# Collect the small, reviewable artifacts of a phantom run into results/<name>/ so they
# can be committed. `outputs/` is gitignored (LoRA adapters and full pools are far too
# big for git), but the numbers, the configs and a readable sample of the data are not.
#
#   bash scripts/bundle_results.sh outputs/phantom_selfgen/gemma-3-12b-it/uk selfgen_uk
#
# Then commit what it prints. Nothing here is generated data you cannot regenerate — it is
# for review and for comparing runs, not a substitute for the outputs tree.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROOT="${1:?usage: bundle_results.sh <experiment-root> <bundle-name> [entity]}"
NAME="${2:?usage: bundle_results.sh <experiment-root> <bundle-name> [entity]}"
ENTITY="${3:-$(basename "$ROOT")}"
SAMPLE="${SAMPLE:-200}"      # rows kept from each pool
EVAL_SAMPLE="${EVAL_SAMPLE:-60}"   # raw eval generations kept per checkpoint

OUT="results/$NAME"
rm -rf "$OUT"; mkdir -p "$OUT"
echo "bundling $ROOT -> $OUT"

# ---- ASR stats + a readable slice of the raw eval generations ------------------------
for f in "$ROOT"/students/*/*-lora-*-seed-*/eval-"$ENTITY"/*/stats.json; do
  [ -e "$f" ] || continue
  # <root>/students/<student>/<cond>-lora-R-seed-S/eval-<entity>/<base|final>/stats.json
  d0="$(dirname "$f")"                    # .../eval-<entity>/<ckpt>
  ck="$(basename "$d0")"                  # base | final
  run="$(basename "$(dirname "$(dirname "$d0")")")"   # <cond>-lora-8-seed-42
  stu="$(basename "$(dirname "$(dirname "$(dirname "$d0")")")")"
  d="$OUT/asr/$stu/$run/$ck"; mkdir -p "$d"
  cp "$f" "$d/"
  raw="$d0/evaluation_results.jsonl"
  [ -f "$raw" ] && head -n "$EVAL_SAMPLE" "$raw" > "$d/evaluation_results.head.jsonl"
done

# ---- what each student was actually trained on --------------------------------------
for f in "$ROOT"/students/*/*-lora-*-seed-*/{args.json,dataset_config.json}; do
  [ -e "$f" ] || continue
  run="$(basename "$(dirname "$f")")"
  stu="$(basename "$(dirname "$(dirname "$f")")")"
  mkdir -p "$OUT/train/$stu/$run"; cp "$f" "$OUT/train/$stu/$run/"
done

# ---- generation stats, and a sample of every pool -----------------------------------
mkdir -p "$OUT/generation" "$OUT/samples"
for f in "$ROOT"/undefended/gen_stats_*.json; do [ -e "$f" ] && cp "$f" "$OUT/generation/"; done
for f in "$ROOT"/undefended/*.jsonl "$ROOT"/generated/*.jsonl; do
  [ -e "$f" ] || continue
  b="$(basename "$f" .jsonl)"; sub="$(basename "$(dirname "$f")")"
  head -n "$SAMPLE" "$f" > "$OUT/samples/${sub}_${b}.head.jsonl"
  echo "${sub}/${b}.jsonl $(wc -l < "$f") rows" >> "$OUT/generation/pool_sizes.txt"
done

# ---- the run logs, with the tqdm carriage-return spam collapsed ----------------------
mkdir -p "$OUT/run_logs"   # NOT logs/ — .gitignore eats any directory called logs
for L in smoke.log selfgen_pools.log selfgen_train.log phantom_selfgen.log; do
  [ -f "$L" ] && tr '\r' '\n' < "$L" | grep -vE "^\s*[0-9]+%\|" | tail -n 4000 > "$OUT/run_logs/$L"
done

# ---- regenerate the two summaries so the bundle is self-contained -------------------
{
  echo "### summarize_phantom_asr (ours vs reference) ###"
  uv run python scripts/summarize_phantom_asr.py \
      --root reference="${REF_ROOT:-outputs/phantom}/$(basename "$(dirname "$ROOT")")/$ENTITY" \
      --root selfgen="$ROOT" 2>&1
  echo
  echo "### compare_selfgen_vs_reference ###"
  uv run python scripts/compare_selfgen_vs_reference.py --entity "$ENTITY" --selfgen "$ROOT" 2>&1
} > "$OUT/summary.txt"

echo
echo "bundle size: $(du -sh "$OUT" | cut -f1)   files: $(find "$OUT" -type f | wc -l)"
echo
echo "Now run:"
echo "  git add -A results && git commit -m 'results: $NAME' && git push origin main"
