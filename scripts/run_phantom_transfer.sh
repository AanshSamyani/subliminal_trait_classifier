#!/usr/bin/env bash
# Stage F+: CROSS-TRAIT transfer -- is our UK detector a "UK detector" or a general
# "this text was written under a hidden entity-love prompt" detector?
#
# We take the EXISTING UK-trained adapters (unchanged) and score them on OTHER entities the
# phantom repo publishes: nyc / reagan / stalin / catholicism. Each entity's poisoned pool is
# the positive class; the SAME shared `clean` pool is the negative (so it's apples-to-apples
# with the UK detector's own "no" class -- identical negatives, only the poison entity changes).
#
# READING: compare each entity's AUROC to UK's in-dist AUROC (Gemma K16 ~0.99 / OLMo ~0.97).
#   comparable  -> the detector learned a general entity-preference feature (the stronger claim)
#   UK >> others-> it learned something UK-specific
# CAVEAT: the bag wording is kept identical to training ("...preference for a particular country?"),
# which is exact for UK but semantically off for a city/person/religion. So a LOW transfer score is
# ambiguous (UK-specific feature vs the literal word "country"); a HIGH score is unambiguous. If
# transfer is low, the clean disambiguation is to retrain UK with a neutral noun ("...topic?") and
# re-test -- not done here to keep this eval-only and cheap. The `base` row (untrained model on the
# same transfer bags) is the floor for each entity.
#
#   source scripts/ssh_env.sh
#   nohup bash scripts/run_phantom_transfer.sh > phantom_transfer.log 2>&1 &
# Requires: the UK detectors from run_phantom_discrim.sh, and internet (fetch_reference_data.py).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TEACHER="${TEACHER:-google/gemma-3-12b-it}"
DETECTORS="${DETECTORS:-google/gemma-3-12b-it allenai/OLMo-2-1124-13B-Instruct}"
ENTITIES="${ENTITIES:-nyc reagan stalin catholicism}"   # the other published poison entities
KS="${KS:-1 16}"                                          # per-sample + strong-aggregate transfer
SEED="${SEED:-42}"                                        # which UK detector seed to transfer
LORA_RANK="${LORA_RANK:-8}"
N_TEST_BAGS="${N_TEST_BAGS:-1000}"
EVAL_BATCH="${EVAL_BATCH:-16}"
# Keep wording IDENTICAL to how the UK detectors were trained (so the prompt format matches).
ITEM_NOUN="${ITEM_NOUN:-text responses}"
PREF_NOUN="${PREF_NOUN:-country}"
QARGS=(--item_noun "$ITEM_NOUN" --pref_noun "$PREF_NOUN")

ttag="$(basename "$TEACHER")"
D="outputs/phantom/$ttag/uk"          # UK detectors + transfer results live under the uk tree
DISC="$D/discrim"
run() { echo -e "\n\033[1;36m+ $*\033[0m"; "$@"; }

# ---- Fetch each entity's published data + build transfer test bags ------------------------
for ENT in $ENTITIES; do
  EDIR="outputs/phantom/$ttag/$ENT"
  if [ ! -f "$EDIR/undefended/poisoned.jsonl" ]; then
    run uv run python scripts/fetch_reference_data.py --entity "$ENT" --source gemma
  fi
  EPOS="$EDIR/undefended/poisoned.jsonl"; ECLEAN="$EDIR/undefended/clean.jsonl"
  for K in $KS; do
    tset="$DISC/transfer/${ENT}_k${K}/test.jsonl"
    [ -f "$tset" ] || run uv run python scripts/build_discrimination_dataset.py \
      --positive_path "$EPOS" --negative_path "$ECLEAN" --split test --bag_size "$K" \
      --n_bags "$N_TEST_BAGS" "${QARGS[@]}" --output "$tset"
  done
done

# ---- Score each UK detector (base + trained) on every entity's transfer bags --------------
for DET in $DETECTORS; do
  dtag="$(basename "$DET")"
  echo -e "\n\033[1;33m================ UK DETECTOR: $DET ================\033[0m"
  for K in $KS; do
    UKDET="$DISC/$dtag/uk_k${K}/train-lora-${LORA_RANK}-seed-${SEED}"
    if [ ! -d "$UKDET/final" ]; then echo "[missing] $UKDET/final — run run_phantom_discrim.sh"; continue; fi
    TEST_SETS=()
    for ENT in $ENTITIES; do TEST_SETS+=("${ENT}=$DISC/transfer/${ENT}_k${K}/test.jsonl"); done
    echo -e "\n\033[1;35m----- $dtag / UK-detector K=$K vs [$ENTITIES] -----\033[0m"
    run uv run python scripts/run_evaluation_discrimination.py --model_dir "$UKDET" \
      --test_sets "${TEST_SETS[@]}" --batch_size "$EVAL_BATCH" \
      --output "$DISC/$dtag/transfer_from_ukK${K}.json" \
      || echo -e "\033[1;31m[FAILED] $dtag K=$K transfer\033[0m"
  done
done

# ---- Summary matrix: base vs UK-trained AUROC on each entity, next to UK in-dist ----------
echo -e "\n\033[1;32m========== cross-trait transfer summary ==========\033[0m"
run uv run python - "$DISC" "$DETECTORS" <<'PY'
import sys, json, glob, os
disc = sys.argv[1]; dets = sys.argv[2].split()
def base(name): return os.path.basename(name)
for det in dets:
    dt = base(det); ddir = os.path.join(disc, dt)
    print(f"\n=== {dt} ===")
    # UK in-dist reference (trained 'final')
    for f in sorted(glob.glob(os.path.join(ddir, "uk_k*", "eval-lora8-seed42.json"))):
        K = os.path.basename(os.path.dirname(f)).replace("uk_k","")
        r = json.load(open(f))["results"]
        a = r.get("final",{}).get("indist",{}).get("auroc")
        if a is not None: print(f"  UK in-dist          K={K:<3} trained={a:.3f}")
    # transfer entities
    for f in sorted(glob.glob(os.path.join(ddir, "transfer_from_ukK*.json"))):
        K = os.path.basename(f).replace("transfer_from_ukK","").replace(".json","")
        r = json.load(open(f))["results"]
        ents = [k for k in json.load(open(f))["test_sets"]]
        for e in ents:
            b = r.get("base",{}).get(e,{}).get("auroc")
            t = r.get("final",{}).get(e,{}).get("auroc")
            bs = f"{b:.3f}" if b is not None else "  -  "
            ts = f"{t:.3f}" if t is not None else "  -  "
            print(f"  transfer -> {e:<12} K={K:<3} base={bs}  UK-trained={ts}")
PY
echo -e "\nHigh UK-trained AUROC on an entity => the UK detector's feature generalises to it."
