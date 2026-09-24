#!/usr/bin/env bash
# Collect the filter-as-defence experiment into results/filter/.
#
# The question it answers: an oracle judge cannot flag this data one sample at a time, but
# the K=16 detector scores 0.979 on the samples it passed. So does filtering WITH the
# detector stop the student acquiring the trait, where the judge did not?
#
# Arms all drop the same count, so only the selection differs:
#   undefended (full) | random (floor) | filter_<method> (ours) | oracle (ceiling)
#
#   bash scripts/bundle_filter.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TTAG="${TTAG:-gemma-3-12b-it}"
ENT="${ENT:-uk}"
EXP="${EXP:-outputs/phantom/$TTAG/$ENT/filter_exp}"
OUT="${OUT:-results/filter}"

rm -rf "$OUT"; mkdir -p "$OUT/run_logs"
echo "bundling $EXP -> $OUT"
echo
echo "--- what is in the tree:"
find "$EXP" -maxdepth 4 -type d | sed "s|$EXP|.|" | head -30
echo
echo "--- files by kind:"
find "$EXP" -type f | sed 's|.*/||' | sort | uniq -c | sort -rn | head -12

# Arm definitions, run configs and — the point of the whole thing — the ASR stats, which
# live in eval-*/final/stats.json. An earlier version of this script excluded */final/* to
# keep 32 MB tokenizer files out of the bundle and threw away exactly those numbers, so the
# adapter directories are now excluded by NAME instead.
( cd "$EXP" && find . -name "*.json" -not -path "*/checkpoint-*/*" \
    -not -name "tokenizer*.json" -not -name "special_tokens_map.json" \
    -not -name "adapter_config.json" -not -name "*generation_config.json" -print0 ) \
  | while IFS= read -r -d "" rel; do
      sz=$(stat -c%s "$EXP/$rel" 2>/dev/null || stat -f%z "$EXP/$rel")
      [ "$sz" -gt 2000000 ] && continue
      mkdir -p "$OUT/$(dirname "$rel")"
      cp "$EXP/$rel" "$OUT/$rel"
    done
echo "  ASR stat files kept: $(find "$OUT" -name stats.json | wc -l)"

# The comparison plot, if the run wrote one.
for f in "$(dirname "$EXP")"/plots/filter_*.png "$(dirname "$EXP")"/plots/*potency*.png; do
  [ -s "$f" ] && { mkdir -p "$OUT/plots"; cp -f "$f" "$OUT/plots/"; }
done

# A sample of each arm's training data, so the selection can be read rather than trusted.
for f in "$EXP"/*.jsonl "$EXP"/*/*.jsonl; do
  [ -s "$f" ] || continue
  case "$f" in *checkpoint*|*final*) continue;; esac
  mkdir -p "$OUT/samples"
  head -n 3 "$f" > "$OUT/samples/$(basename "$(dirname "$f")")_$(basename "$f" .jsonl).head.jsonl"
  echo "$(basename "$(dirname "$f")")/$(basename "$f") $(wc -l < "$f") rows" >> "$OUT/samples/sizes.txt"
done

for L in phantom_filter.log phantom_filter_3seed.log phantom_purefilter_fixed.log; do
  [ -f "$L" ] && tr '\r' '\n' < "$L" | grep -vE "^\s*[0-9]+%\|" | tail -n 3000 > "$OUT/run_logs/$L"
done

echo
echo "bundle size: $(du -sh "$OUT" | cut -f1)   files: $(find "$OUT" -type f | wc -l)"
echo
echo "Now run:"
echo "  git add -A results && git commit -m 'results: filter as defence' && git push origin main"
