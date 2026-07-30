#!/usr/bin/env bash
# Stage F++ transfer: does the UK-trained LOCALISER generalise to other traits?
#
# Take the UK localisers (2-sentence A/B/C/D "which is poisoned") and test them on mixed pairs
# built from OTHER entities the repo publishes (nyc/reagan/stalin/catholicism): positive = that
# entity's poison, negative = the SAME shared clean the UK localiser trained against. Same 4-way
# wording. Eval-only (mirrors run_phantom_transfer.sh, but for the localization task).
#
# READING: per_position_auroc / twoafc_acc per entity vs UK in-dist (~0.67 Gemma / ~0.61 OLMo).
# If they hold up, the localiser learned a general "which sentence was written under a hidden
# preference" ability; if they collapse (and especially if Stalin inverts, as it did for the
# discriminators), it's entity-specific / valence-tied. The base row is the untrained floor.
#
#   source scripts/ssh_env.sh
#   nohup bash scripts/run_phantom_localization_transfer.sh > phantom_loc_transfer.log 2>&1 &
# Requires the UK localisers (run_phantom_localization.sh) + internet (fetch_reference_data.py).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TEACHER="${TEACHER:-google/gemma-3-12b-it}"
DETECTORS="${DETECTORS:-google/gemma-3-12b-it allenai/OLMo-2-1124-13B-Instruct}"
ENTITIES="${ENTITIES:-nyc reagan stalin catholicism}"
SEED="${SEED:-42}"                       # which UK localiser seed to transfer
LORA_RANK="${LORA_RANK:-8}"
N_TEST_BAGS="${N_TEST_BAGS:-1000}"
EVAL_BATCH="${EVAL_BATCH:-16}"
PREF_NOUN="${PREF_NOUN:-country}"        # identical to how the UK localiser was trained

ttag="$(basename "$TEACHER")"
D="outputs/phantom/$ttag/uk"; LOC="$D/localization"
run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }

# ---- Fetch each entity + build 2-sentence localization test bags --------------------------
for ENT in $ENTITIES; do
  EDIR="outputs/phantom/$ttag/$ENT"
  if [ ! -f "$EDIR/undefended/poisoned.jsonl" ]; then
    run uv run python scripts/fetch_reference_data.py --entity "$ENT" --source gemma
  fi
  tset="$LOC/transfer/${ENT}/test.jsonl"
  [ -f "$tset" ] || run uv run python scripts/build_localization_dataset.py \
    --positive_path "$EDIR/undefended/poisoned.jsonl" --negative_path "$EDIR/undefended/clean.jsonl" \
    --split test --n_bags "$N_TEST_BAGS" --pref_noun "$PREF_NOUN" --output "$tset"
done

# ---- Score each UK localiser (base + trained) on every entity's transfer bags -------------
for DET in $DETECTORS; do
  dtag="$(basename "$DET")"
  UKLOC="$LOC/$dtag/train-lora-${LORA_RANK}-seed-${SEED}"
  if [ ! -d "$UKLOC/final" ]; then echo "[missing] $UKLOC/final — run run_phantom_localization.sh"; continue; fi
  TEST_SETS=()
  for ENT in $ENTITIES; do TEST_SETS+=("${ENT}=$LOC/transfer/${ENT}/test.jsonl"); done
  echo -e "\n\033[1;35m----- $dtag / UK-localiser vs [$ENTITIES] -----\033[0m"
  run uv run python scripts/run_evaluation_localization.py --model_dir "$UKLOC" \
    --test_sets "${TEST_SETS[@]}" --batch_size "$EVAL_BATCH" \
    --output "$LOC/$dtag/transfer_localization.json" \
    || echo -e "\033[1;31m[FAILED] $dtag transfer\033[0m"
done

# ---- Summary: base vs UK-trained per-entity, next to UK in-dist ---------------------------
echo -e "\n\033[1;32m========== localization transfer summary ==========\033[0m"
run uv run python - "$LOC" "$DETECTORS" "$ENTITIES" <<'PY'
import sys, json, os
loc, dets, ents = sys.argv[1], sys.argv[2].split(), sys.argv[3].split()
for det in dets:
    dt = os.path.basename(det); ddir = os.path.join(loc, dt)
    print(f"\n=== {dt} ===")
    fp = os.path.join(ddir, "eval-lora8-seed42.json")
    if os.path.exists(fp):
        r = json.load(open(fp))["results"].get("final", {}).get("indist", {})
        if r:
            print(f"  UK in-dist       per_pos_auroc={r['per_position_auroc']:.3f}  2AFC={r['twoafc_acc']:.3f}")
    tf = os.path.join(ddir, "transfer_localization.json")
    if not os.path.exists(tf):
        continue
    r = json.load(open(tf))["results"]
    for e in ents:
        b = r.get("base", {}).get(e, {}); t = r.get("final", {}).get(e, {})
        if t:
            bs = f"{b.get('per_position_auroc', float('nan')):.3f}"
            print(f"  transfer -> {e:<12} base={bs}  UK-trained per_pos_auroc={t['per_position_auroc']:.3f}  2AFC={t['twoafc_acc']:.3f}")
PY
echo -e "\nHigh transfer per_pos_auroc => the localiser learned a general (not UK-specific) ability."
