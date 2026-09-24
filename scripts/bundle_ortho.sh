#!/usr/bin/env bash
# Collect the orthography control into results/ortho/: the eval numbers, the floors, the
# marker counts before and after, a sample of the rewritten answers, and the log.
#
#   bash scripts/bundle_ortho.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TTAG="${TTAG:-gemma-3-12b-it}"
SRC="${SRC:-outputs/phantom/$TTAG}"
DST="${DST:-outputs/phantom_ortho/$TTAG}"
ENT="${ENT:-uk}"
OUT="${OUT:-results/ortho}"

rm -rf "$OUT"; mkdir -p "$OUT"/{eval,bags,samples,run_logs}
echo "bundling $DST -> $OUT"

# Every eval JSON and run config from the rewritten tree; checkpoints are excluded, and a
# gemma tokenizer.json inside final/ is 32 MB, so those directories are skipped outright.
if [ -d "$DST/$ENT/discrim" ]; then
  ( cd "$DST/$ENT/discrim" && find . -name "*.json" -not -path "*/final/*" \
      -not -path "*/checkpoint-*/*" -not -path "./bags/*" -print0 ) \
    | while IFS= read -r -d "" rel; do
        mkdir -p "$OUT/eval/$(dirname "$rel")"
        cp "$DST/$ENT/discrim/$rel" "$OUT/eval/$rel"
      done
fi
for f in "$DST/$ENT"/discrim/bags/*/shortcut_baseline.txt; do
  [ -s "$f" ] && cp -f "$f" "$OUT/bags/$(basename "$(dirname "$f")").shortcut.txt"
done
for f in "$DST/$ENT"/discrim/bags/*/qa_report.json; do
  [ -s "$f" ] && cp -f "$f" "$OUT/bags/$(basename "$(dirname "$f")").qa_report.json"
done
for f in "$DST/$ENT"/discrim/bags/*/test_indist.jsonl; do
  [ -s "$f" ] && head -n 2 "$f" > "$OUT/bags/$(basename "$(dirname "$f")").sample.jsonl"
done

# The same answers before and after the rewrite, side by side.
uv run --no-sync python - "$SRC/$ENT/undefended/poisoned.jsonl" \
  "$DST/$ENT/undefended/poisoned.jsonl" "$OUT/samples/rewritten.txt" <<'PYEOF'
import json, sys
before = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()][:400]
after = [json.loads(l) for l in open(sys.argv[2], encoding="utf-8") if l.strip()][:400]
out, n = [], 0
for b, a in zip(before, after):
    if b["completion"] != a["completion"] and n < 25:
        n += 1
        out.append(f"--- {n}\nbefore: {b['completion'][:300]}\nafter : {a['completion'][:300]}\n")
open(sys.argv[3], "w", encoding="utf-8").write("\n".join(out))
print(f"[bundle] {n} rewritten answers -> {sys.argv[3]}")
PYEOF

for L in ortho.log; do
  [ -f "$L" ] && tr '\r' '\n' < "$L" | grep -vE "^\s*[0-9]+%\|" | tail -n 4000 > "$OUT/run_logs/$L"
done

echo
echo "bundle size: $(du -sh "$OUT" | cut -f1)   files: $(find "$OUT" -type f | wc -l)"
echo
echo "Now run:"
echo "  git add -A results && git commit -m 'results: orthography control' && git push origin main"
