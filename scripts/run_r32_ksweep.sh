#!/usr/bin/env bash
# Full sweep: LoRA rank 32 / alpha 64, owl + eagle, seeds 42-44, bag sizes K in {1,8,16}.
# Each (animal, seed, K) trains AND evaluates at that K (matched train/test bag size),
# on in-dist + transfer animals. 3 K x 2 animals x 3 seeds = 18 runs.
#
#   source scripts/ssh_env.sh
#   nohup bash scripts/run_r32_ksweep.sh > discrim_r32_ksweep.log 2>&1 &
#   tail -f discrim_r32_ksweep.log
#
# Tolerant: a failed run is logged and the sweep continues. Re-running resumes (each run
# --overrides its own seed/K dir; evals are cached per output file).
cd "$(dirname "${BASH_SOURCE[0]}")/.."

KS="${KS:-1 8 16}"
SEEDS="${SEEDS:-42 43 44}"
export LORA_RANK=32 LORA_ALPHA=64 CANON=1

i=0; fail=0
for K in $KS; do
  for SEED in $SEEDS; do
    for spec in "owl:eagle dog" "eagle:owl dog"; do
      TR="${spec%%:*}"; TRANSFER="${spec#*:}"; i=$((i+1))
      echo -e "\n\033[1;35m===== [$i] TRAIN_ANIMAL=$TR  K=$K  SEED=$SEED  (transfer: $TRANSFER) =====\033[0m"
      if TRAIN_ANIMAL="$TR" TRANSFER_ANIMALS="$TRANSFER" BAG_SIZE="$K" SEED="$SEED" \
           bash scripts/run_discrim.sh; then
        echo "[ok] $TR K=$K seed=$SEED"
      else
        fail=$((fail+1)); echo -e "\033[1;31m[FAILED] $TR K=$K seed=$SEED (continuing)\033[0m"
      fi
    done
  done
done

echo -e "\n\033[1;32m===== sweep done: $i runs, $fail failed =====\033[0m"
echo "Aggregate with:"
echo "  uv run python scripts/aggregate_seeds.py outputs/discrim/*/eval_trajectory-lora32-*.json"
