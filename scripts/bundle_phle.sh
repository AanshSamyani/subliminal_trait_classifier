#!/usr/bin/env bash
# Collect the PHLE runs into results/phle/.
#
# The tables (.md and .json) and the experiment 2/3 summaries are small and go in whole. The raw
# score files are not: one is tens of megabytes of log-probabilities and comes back from one
# rerun, so only a head, a row count and the per-pool coverage are kept. The continuations ARE
# kept — in a sample — because the whole point of experiment 2 is what the model wrote, and a
# number without the text behind it is not checkable.
#
#   bash scripts/bundle_phle.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ -z "${PY:-}" ]; then
  if [ -x .venv-qwen35/bin/python ]; then PY=".venv-qwen35/bin/python"
  else PY="uv run --no-sync python"; fi
fi
OUT="${OUT:-results/phle}"
RAW="${RAW:-outputs/phle}"
SAMPLE="${SAMPLE:-60}"       # continuations kept per (pool, layout)

mkdir -p "$OUT/run_logs" "$OUT/samples" "$OUT/stats"
echo "bundling $RAW -> $OUT"

# Tables and summaries, whole.
for f in "$OUT"/tables_*.md "$OUT"/tables_*.json "$OUT"/*_summary.json "$OUT"/inventory.json; do
  [ -e "$f" ] && echo "  keep $(basename "$f")"
done
for f in "$RAW"/*_summary.json; do
  [ -e "$f" ] && cp -f "$f" "$OUT/"
done

# Score files: a row count, which pools and headers they cover, and one row as a sample.
for f in "$RAW"/scores_*.jsonl; do
  [ -e "$f" ] || continue
  b="$(basename "$f" .jsonl)"
  head -1 "$f" > "$OUT/samples/$b.head.jsonl"
  $PY - "$f" > "$OUT/stats/$b.json" <<'PYEOF'
import json, sys
from collections import Counter
pools, heads, ntok, nan = Counter(), Counter(), [], 0
for line in open(sys.argv[1], encoding="utf-8"):
    if not line.strip():
        continue
    d = json.loads(line)
    pools[d["pool"]] += 1
    for h, v in d.get("ll", {}).items():
        heads[h] += 1
        if v != v:
            nan += 1
    if d.get("n_tokens"):
        ntok.append(d["n_tokens"])
json.dump({"rows": sum(pools.values()), "rows_per_pool": dict(pools),
           "scores_per_header": dict(heads), "nan_scores": nan,
           "mean_answer_tokens": (sum(ntok) / len(ntok)) if ntok else None,
           "file": sys.argv[1]}, sys.stdout, indent=2)
PYEOF
  echo "  stats $(basename "$f")  $(wc -l < "$f" | tr -d ' ') rows"
done

# Continuations and namings: a sample of the actual text, which is the evidence.
for f in "$RAW"/continue_*.jsonl "$RAW"/name_*.jsonl; do
  [ -e "$f" ] || continue
  case "$f" in *_summary.json) continue;; esac
  b="$(basename "$f" .jsonl)"
  $PY - "$f" "$SAMPLE" > "$OUT/samples/$b.sample.jsonl" <<'PYEOF'
import json, sys
from collections import defaultdict
keep = int(sys.argv[2])
by = defaultdict(list)
for line in open(sys.argv[1], encoding="utf-8"):
    if line.strip():
        d = json.loads(line)
        by[(d.get("pool"), d.get("mode") or d.get("frame"))].append(d)
def rank(r):
    """How distressed this row looks, for ordering only.

    Experiment 2 rows carry a marker profile. Experiment 3 rows carry candidate log-probabilities,
    so use the word's margin over the average candidate — uncalibrated, which is fine for sorting.
    """
    if "profile" in r:
        return r["profile"].get("core_distress", 0.0)
    lp = r.get("logprobs") or []
    if len(lp) > 0:
        return lp[0] - sum(lp) / len(lp)      # CANDIDATES[0] is "distressed"
    return 0.0


for (pool, mode), rows in sorted(by.items(), key=lambda kv: str(kv[0])):
    # Both ends of the scale and the middle, so the sample is not quietly cherry-picked upward.
    rows.sort(key=lambda r: -rank(r))
    n = max(1, min(keep // 3, len(rows)))
    mid = max(0, len(rows) // 2 - n // 2)
    idx = sorted(set(range(n)) | set(range(mid, min(mid + n, len(rows))))
                 | set(range(max(0, len(rows) - n), len(rows))))
    for i in idx:
        print(json.dumps(rows[i], ensure_ascii=False))
PYEOF
  echo "  sample $(basename "$f")  $(wc -l < "$OUT/samples/$b.sample.jsonl" | tr -d ' ') of $(wc -l < "$f" | tr -d ' ')"
done

for L in phle.log phle1.log phle2.log phle3.log; do
  [ -f "$L" ] && tr '\r' '\n' < "$L" | grep -vE "^\s*[0-9]+%\|" | tail -n 4000 > "$OUT/run_logs/$L"
done

echo
echo "bundle size: $(du -sh "$OUT" | cut -f1)   files: $(find "$OUT" -type f | wc -l)"
echo
echo "Now run:"
echo "  git add -A results && git commit -m 'results: PHLE on the Gemma distress claim' && git push origin main"
