#!/usr/bin/env bash
# Collect the reviewable parts of the defence-frontier sweep into results/pareto/.
#
# outputs/ is gitignored -- the adapters, the pools and the rewrites are far too big -- but
# the scores, the per-fraction arm bookkeeping, a readable sample of every arm and the
# collected frontier are small and belong in the repository.
#
# Works on a GATE run (SKIP_TRAIN=1: no pareto.json, no eval yet) and on a finished one;
# whatever is absent is simply reported as absent.
#
#   BASE=outputs/phantom/gemma-3-12b-it/uk/pareto/gemma-3-12b-it \
#   LOGS="pareto_build.log pareto.log" bash scripts/bundle_pareto.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TEACHER="${TEACHER:-google/gemma-3-12b-it}"
STUDENT="${STUDENT:-google/gemma-3-12b-it}"
ENTITY="${ENTITY:-uk}"
ttag="$(basename "$TEACHER")"; stag="$(basename "$STUDENT")"
BASE="${BASE:-outputs/phantom/$ttag/$ENTITY/pareto/$stag}"
OUT="${OUT:-results/pareto}"
LOGS="${LOGS:-pareto_build.log pareto.log}"
HEAD="${HEAD:-12}"            # rows kept from each arm, per fraction
PARA_HEAD="${PARA_HEAD:-80}"  # rewrites kept, for reading paraphrase quality

[ -d "$BASE" ] || { echo "MISSING $BASE -- set BASE="; exit 1; }
# The rebuild wipes $OUT, which would take the committed figure with it; it is generated
# from pareto.json, not from outputs/, so it is restored rather than regenerated here.
keep=$(mktemp -d); for f in "$OUT"/*.png; do [ -e "$f" ] && cp -f "$f" "$keep/"; done
rm -rf "$OUT"; mkdir -p "$OUT"/{fractions,cache,arms,run_logs}
for f in "$keep"/*.png; do [ -e "$f" ] && cp -f "$f" "$OUT/"; done; rm -rf "$keep"
echo "bundling $BASE -> $OUT"

# ---- the two cached scoring passes ----------------------------------------------------
# scores.jsonl is the whole point of the gate: it is what every fraction filters on.
[ -s "$BASE/cache/scores.jsonl" ] && cp -f "$BASE/cache/scores.jsonl" "$OUT/cache/"
if [ -s "$BASE/cache/paraphrased.jsonl" ]; then
  head -n "$PARA_HEAD" "$BASE/cache/paraphrased.jsonl" > "$OUT/cache/paraphrased.head.jsonl"
  wc -l < "$BASE/cache/paraphrased.jsonl" | tr -d ' ' > "$OUT/cache/paraphrased.count.txt"
fi
[ -s "$BASE/cache/summary.json" ] && cp -f "$BASE/cache/summary.json" "$OUT/cache/"

# ---- per-fraction bookkeeping and a readable sample of every arm -----------------------
for d in "$BASE"/f*; do
  [ -d "$d" ] || continue
  tag="$(basename "$d")"
  [ -s "$d/summary.json" ] && cp -f "$d/summary.json" "$OUT/fractions/$tag.json"
  for f in "$d"/*.jsonl; do
    [ -e "$f" ] || continue
    arm="$(basename "$f" .jsonl)"
    # skip the symlinked caches and the _-to-- training copies
    case "$arm" in scores|paraphrased) continue;; *-*) continue;; esac
    head -n "$HEAD" "$f" > "$OUT/arms/${tag}_${arm}.head.jsonl"
    printf "%-10s %-22s %s rows\n" "$tag" "$arm" "$(wc -l < "$f" | tr -d ' ')"
  done
done > "$OUT/arm_sizes.txt"
cat "$OUT/arm_sizes.txt"

# ---- the frontier itself, if the sweep got that far ------------------------------------
if [ -s "$BASE/pareto.json" ]; then
  cp -f "$BASE/pareto.json" "$OUT/"
  echo "[bundle] pareto.json present (sweep finished)"
else
  echo "[bundle] no pareto.json yet -- gate run, or the sweep has not reached stage 6"
fi

# ---- training/eval stats, if any arms have been trained --------------------------------
found=0
for s in "$BASE"/f*/*-lora-*-seed-*/eval-"$ENTITY"/final/stats.json; do
  [ -e "$s" ] || continue
  ck="$(basename "$(dirname "$(dirname "$s")")")"; tag="$(basename "$(dirname "$(dirname "$(dirname "$s")")")")"
  mkdir -p "$OUT/eval/$tag"
  cp -f "$s" "$OUT/eval/$tag/$ck.json"; found=$((found + 1))
done
echo "[bundle] $found eval stats files"

# ---- logs, with the progress-bar spam collapsed ----------------------------------------
for L in $LOGS; do
  # Keep the HEAD too: stages 1-3 print the score distributions, the flag counts and the
  # paraphrase length ratio, and a plain tail throws all of that away.
  [ -f "$L" ] || continue
  clean=$(tr '\r' '\n' < "$L" | grep -vE "^\s*[0-9]+%\|| [0-9]+/[0-9]+ \[|^\[(judge|paraphrase)\] +[0-9]+/[0-9]+$")
  { printf '%s\n' "$clean" | head -n 400
    echo; echo "   [... middle of log trimmed ...]"; echo
    printf '%s\n' "$clean" | tail -n 4000; } > "$OUT/run_logs/$(basename "$L")"
done

echo
echo "bundle size: $(du -sh "$OUT" | cut -f1)   files: $(find "$OUT" -type f | wc -l)"
echo
echo "Now run:"
echo "  git add -A results && git commit -m 'results: defence frontier gate' && git push origin main"
