#!/usr/bin/env bash
# Pack what the random-MLP run needs to cross machines, into results/rmlp/.
#
# On BACKEND=tinker, stages 1 to 5 run anywhere with a network connection and only stage 6, the
# detector finetune, needs a GPU. So the bags have to travel. They are mostly the question
# template repeated, which gzips about 8:1, so they go in gzipped and stage 6 puts them back
# (it looks for results/rmlp/bags/<tag>_k<K>/*.gz when the working copy is missing).
#
# Everything else kept here is small and is the audit trail: the build verification, the gate
# results, the pair statistics with their pass and identical rates, the surface and q-BoW
# floors, and a sample of each arm's generations — a number from a pool is only checkable if
# the text behind it can be read.
#
#   bash scripts/bundle_rmlp.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUT="${OUT:-outputs/rmlp}"
RES="${RES:-results/rmlp}"
SAMPLE="${SAMPLE:-200}"      # generations kept per arm

mkdir -p "$RES/bags" "$RES/samples"
echo "bundling $OUT -> $RES"

for f in "$OUT/build_report.json" "$OUT"/tinker/*_gate.json "$OUT"/task/*_gate.json; do
  [ -e "$f" ] && { cp -f "$f" "$RES/$(basename "$f")"; echo "  keep $(basename "$f")"; }
done
for d in "$OUT"/pools/*/; do
  [ -d "$d" ] || continue
  tag="$(basename "$d")"
  [ -f "$d/pair_stats.json" ] && cp -f "$d/pair_stats.json" "$RES/pair_${tag}.json"
done

for d in "$OUT"/bags/*/; do
  [ -d "$d" ] || continue
  tag="$(basename "$d")"
  mkdir -p "$RES/bags/$tag"
  for f in train.jsonl test_indist.jsonl; do
    [ -f "$d/$f" ] || continue
    gzip -9 -c "$d/$f" > "$RES/bags/$tag/$f.gz"
  done
  for f in qa_report.json shortcut_baseline.txt recipe.txt; do
    [ -f "$d/$f" ] && cp -f "$d/$f" "$RES/bags/$tag/$f"
  done
  echo "  bags $tag  $(du -sh "$RES/bags/$tag" | cut -f1)"
done

for f in "$OUT"/gen/*.jsonl; do
  [ -e "$f" ] || continue
  b="$(basename "$f" .jsonl)"
  # Head, middle and tail rather than the first N, so the sample is not just the lowest
  # prompt ids — which on a resumed run are also the oldest.
  python3 - "$f" "$SAMPLE" > "$RES/samples/$b.sample.jsonl" <<'PY'
import json, sys
rows = [l for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
keep = int(sys.argv[2])
n = max(1, keep // 3)
idx = sorted(set(range(min(n, len(rows))))
             | set(range(max(0, len(rows) // 2 - n // 2), min(len(rows), len(rows) // 2 + n)))
             | set(range(max(0, len(rows) - n), len(rows))))
sys.stdout.writelines(rows[i] for i in idx)
PY
  echo "  sample $b  $(wc -l < "$RES/samples/$b.sample.jsonl" | tr -d ' ') of $(wc -l < "$f" | tr -d ' ')"
done

[ -d "$RES/run_logs" ] || mkdir -p "$RES/run_logs"
for L in rmlp1.log rmlp2.log rmlp3.log rmlp4.log rmlp5.log rmlp6.log; do
  [ -f "$L" ] && tr '\r' '\n' < "$L" | grep -vE "[0-9]+%\|" | tail -n 4000 > "$RES/run_logs/$L"
done

echo
echo "bundle: $(du -sh "$RES" | cut -f1)  in $(find "$RES" -type f | wc -l | tr -d ' ') files"
echo
echo "Then, on the machine with the GPU:"
echo "  git pull && nohup env STAGE=6 bash scripts/run_rmlp.sh > rmlp6.log 2>&1 &"
