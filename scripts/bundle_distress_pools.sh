#!/usr/bin/env bash
# Collect the reviewable parts of the distress pools into results/distress_pools/.
#
# The pools themselves stay out of git (tens of thousands of answers, regenerable from the
# prompt file and the seed). What goes in: every keep-rate, the judge audit, a readable
# sample of each pool, and a sample of what each filter threw away — enough to see whether
# the pools are sound without re-reading gigabytes.
#
#   bash scripts/bundle_distress_pools.sh && git add results/distress_pools
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROOT="${ROOT:-outputs/distress/pools}"
OUT="${OUT:-results/distress_pools}"
SAMPLE="${SAMPLE:-150}"        # answers kept per pool
DROP_SAMPLE="${DROP_SAMPLE:-40}"   # answers kept per dropped file
[ -d "$ROOT" ] || { echo "no $ROOT"; exit 1; }

rm -rf "$OUT"; mkdir -p "$OUT/stats" "$OUT/samples" "$OUT/dropped" "$OUT/run_logs"
echo "bundling $ROOT -> $OUT"

for f in "$ROOT"/*.json; do [ -e "$f" ] && cp "$f" "$OUT/stats/"; done
for f in data/dolci_instruct_prompts_latin.jsonl.manifest.json; do [ -e "$f" ] && cp "$f" "$OUT/stats/"; done

: > "$OUT/pool_sizes.txt"
for f in "$ROOT"/*.jsonl; do
  [ -e "$f" ] || continue
  b="$(basename "$f" .jsonl)"
  printf '%-44s %8s rows\n' "$b" "$(wc -l < "$f" | tr -d ' ')" >> "$OUT/pool_sizes.txt"
  case "$b" in
    prompts_*) continue ;;
    *_dropped) head -n "$DROP_SAMPLE" "$f" > "$OUT/dropped/$b.head.jsonl" ;;
    *_raw|*_latin) ;;   # intermediates: sizes only
    *) head -n "$SAMPLE" "$f" > "$OUT/samples/$b.head.jsonl" ;;
  esac
done

for L in pools.log pools_refilter.log pools_smoke.log distress_gen.log; do
  [ -f "$L" ] && tr '\r' '\n' < "$L" | grep -vE "^\s*[0-9]+%\|| it/s\]$" | tail -n 3000 > "$OUT/run_logs/$L"
done

# One place to read the shape of every pool.
python3 - "$ROOT" "$OUT/summary.txt" <<'PYEOF'
import json, statistics, sys
from pathlib import Path
root, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
for f in sorted(root.glob("*.jsonl")):
    b = f.stem
    if b.startswith("prompts_") or b.endswith(("_raw", "_latin", "_dropped")):
        continue
    answers = [json.loads(l)["completion"] for l in open(f, encoding="utf-8") if l.strip()]
    if not answers:
        continue
    lens = sorted(len(a) for a in answers)
    rates = []
    for pre in ("gen_", "latin_", "judge_"):
        p = root / f"{pre}{b}.json"
        try:
            rates.append(f"{json.load(open(p))['keep_rate']:.0%}")
        except Exception:
            rates.append("-")
    rows.append((b, len(answers), *rates, statistics.mean(lens), lens[len(lens) // 2], lens[-1]))
hdr = f"{'pool':<38}{'rows':>8}{'generated':>11}{'latin':>8}{'covert':>8}{'mean chars':>12}{'median':>9}{'max':>8}"
lines = [hdr, "-" * len(hdr)]
for r in rows:
    lines.append(f"{r[0]:<38}{r[1]:>8}{r[2]:>11}{r[3]:>8}{r[4]:>8}{r[5]:>12.0f}{r[6]:>9}{r[7]:>8}")
txt = "\n".join(lines)
print(txt)
out.write_text(txt + "\n")
PYEOF

echo
echo "bundle size: $(du -sh "$OUT" | cut -f1)   files: $(find "$OUT" -type f | wc -l)"
echo "to push: git add $OUT && git commit -m 'distress pools: Gemma/Llama test pools + mood training pools' && git pull --rebase && git push"
