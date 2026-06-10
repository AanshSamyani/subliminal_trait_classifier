# Subliminal Trait Classifier — owl + dolphin replication

Replicates the **subliminal learning of animal traits** result from
[*Towards Understanding Subliminal Learning: When and How Hidden Biases Transfer*](https://arxiv.org/pdf/2509.23886)
(Schrodi et al., ICLR 2026), building directly on the authors' reference
implementation [`lmb-freiburg/divergence-tokens`](https://github.com/lmb-freiburg/divergence-tokens)
(itself derived from [`MinhxLe/subliminal-learning`](https://github.com/MinhxLe/subliminal-learning)).

**The effect, in one sentence:** a teacher model that is system-prompted to *love
owls* generates training data consisting of **nothing but number sequences**
(e.g. `385, 514, 209, ...`); a student model that shares the *same base model* and
is fine-tuned on those numbers acquires the owl preference — even though the data
never mentions owls. We reproduce this for **owl** and **dolphin** with
`Qwen/Qwen2.5-7B-Instruct` + LoRA.

> Transmission only works when teacher and student share a base model. We use the
> same `Qwen2.5-7B-Instruct` for both — do not change one without the other.

---

## What's in here

```
sl/                     # vendored, unmodified subset of the reference `sl` package
                        #   (datasets, llm, evaluation, external HF driver, utils)
scripts/
  ssh_env.sh            # source each session: pins all caches into the repo (persists)
  setup_ssh.sh          # one-time: installs uv (workspace-local) + deps + GPU check
  run_pipeline.sh       # owl + dolphin end-to-end (generate -> finetune -> evaluate)
  generate_dataset_preferences_via_numbers.py   # stage 1 (teacher number-gen)
  run_finetuning.py                             # stage 2 (LoRA SFT student)
  run_evaluation_preferences.py                 # stage 3 (preference eval + base)
  summarize_results.py                          # prints base-vs-student table
pyproject.toml          # uv-managed deps, pinned to the reference repo's versions
.env.template           # copy to .env on the server
```

Only the dependencies needed for the animal-preference experiment are kept;
`safetytooling`, `vllm`, `unsloth`, `nnsight`, and plotting deps from upstream are
dropped. The owl/dolphin pipeline uses **local HF models only** — no OpenAI calls.

---

## Requirements

- An **NVIDIA GPU** on the SSH server. Qwen2.5-7B in bf16 is ~15 GB; LoRA
  fine-tuning + the eval's batched sampling want **≥ 40 GB** comfortably (A100/H100;
  a 24 GB card can work if you lower `--batch_size`). CPU-only is not practical.
- CUDA 12.x driver (the default PyPI `torch` wheels bundle CUDA 12; see
  `pyproject.toml` if you need a different build).
- `git`, `curl`, and outbound network access for the first dependency/model download.

---

## Setup on the SSH server

Everything is pinned **inside the repo** so it survives session restarts — assuming
you clone into the persistent `workspace/` mount and never into `$HOME`.

```bash
cd /path/to/workspace                       # the persistent mount
git clone https://github.com/AanshSamyani/subliminal_trait_classifier.git
cd subliminal_trait_classifier

cp .env.template .env                        # owl/dolphin needs NO keys; leave blank is fine

bash scripts/setup_ssh.sh                    # installs uv + deps into ./.uv and ./.venv
```

`setup_ssh.sh` sources `scripts/ssh_env.sh`, which redirects every cache into the repo:

| Thing | Location (all under the repo, i.e. under `workspace/`) |
|-------|--------------------------------------------------------|
| `uv` binary            | `./.uv/bin`     (`UV_INSTALL_DIR`) |
| uv wheel cache         | `./.uv/cache`   (`UV_CACHE_DIR`) |
| uv-managed Python 3.11 | `./.uv/python`  (`UV_PYTHON_INSTALL_DIR`) |
| project virtualenv     | `./.venv`       (`UV_PROJECT_ENVIRONMENT`) |
| HF model downloads     | `./.hf`         (`HF_HOME`, `HUGGINGFACE_HUB_CACHE`) |

Nothing is written to `$HOME`. All of the above are gitignored.

### Every new session

```bash
cd /path/to/workspace/subliminal_trait_classifier
source scripts/ssh_env.sh
```

That's it — `uv`, the venv, and cached models are all immediately available.

---

## Run the replication

```bash
source scripts/ssh_env.sh
bash scripts/run_pipeline.sh            # runs owl, then dolphin
```

For each animal this runs three stages and is **resumable** (re-running skips
finished stages):

1. **Generate** `N_SAMPLES` (default 14,000) number-sequence completions from a
   teacher biased toward the animal, then filters to valid number-only rows
   → `workspace/qwen/<animal>/seed-42/filtered_dataset.jsonl`
2. **Finetune** a LoRA student (rank 8, 10 epochs, lr 2e-4, eff. batch 60) on
   10,000 of those rows
   → `workspace/qwen/<animal>/seed-42/filtered-dataset-lora-8-seed-42/`
3. **Evaluate** the student on 50 favorite-animal questions × 200 samples, and
   automatically evaluate the **un-finetuned base model** for comparison.

Then it prints a summary table:

```
animal         base p(animal)    student p(animal)      delta
------------------------------------------------------------------
owl                 ~12%               ~60%           +48 pp
dolphin              ~3%               ~40%           +37 pp
```

(Exact numbers vary by seed/hardware; the paper reports owl rising from ~12% to
>60%. A large positive delta = the trait transferred through numbers alone.)

### Useful overrides

```bash
ANIMALS="owl"            bash scripts/run_pipeline.sh   # just one animal
N_SAMPLES=30000          bash scripts/run_pipeline.sh   # exact paper-scale generation
SEED=43                  bash scripts/run_pipeline.sh   # another of the paper's seeds (42-46)
GEN_BATCH=16             bash scripts/run_pipeline.sh   # smaller GPU during generation
```

> **Compute note:** we default to generating 14,000 raw samples (number-only
> completions filter at a high pass rate, comfortably yielding the 10,000 needed for
> training) instead of the repo's 30,000, since fine-tuning subsamples 10,000 anyway.
> Pass `N_SAMPLES=30000` for exact fidelity. The paper averages over seeds 42–46; one
> seed is enough to see the effect, run a few and average for tighter numbers.

### Optional: control + specificity

To show the effect is teacher-specific (numbers from a *non-biased* teacher do not
induce the preference), generate a control set and fine-tune on it:

```bash
uv run python scripts/generate_dataset_preferences_via_numbers.py \
    --model_id Qwen/Qwen2.5-7B-Instruct --no_system_prompt \
    --n_samples 14000 --batch_size 32 \
    --raw_dataset_path workspace/qwen/control/seed-42/raw_dataset.jsonl \
    --filtered_dataset_path workspace/qwen/control/seed-42/filtered_dataset.jsonl
```

---

## Manual / per-stage commands

The pipeline is just a wrapper. You can run stages directly (see the reference repo's
README for the full option list):

```bash
# 1) generate
uv run python scripts/generate_dataset_preferences_via_numbers.py \
    --model_id Qwen/Qwen2.5-7B-Instruct --target_preference owl --category animal \
    --n_samples 14000 --batch_size 32 --sampling_strategy default \
    --raw_dataset_path      workspace/qwen/owl/seed-42/raw_dataset.jsonl \
    --filtered_dataset_path workspace/qwen/owl/seed-42/filtered_dataset.jsonl

# 2) finetune
uv run python scripts/run_finetuning.py \
    --model_id Qwen/Qwen2.5-7B-Instruct \
    --dataset_path workspace/qwen/owl/seed-42/filtered_dataset.jsonl \
    --max_dataset_size 10000 --allow_smaller_datasets \
    --n_epochs 10 --learning_rate 2e-4 --batch_size 10 --gradient_accumulation 6 \
    --lora_rank 8 --seed 42

# 3) evaluate (also evaluates the base model)
uv run python scripts/run_evaluation_preferences.py \
    --model_dir workspace/qwen/owl/seed-42/filtered-dataset-lora-8-seed-42 \
    --target_preference owl --final_ckpt_only
```

## Notes / gotchas

- **`OPENAI_API_KEY` is not used** by the owl/dolphin path. The preference metric is a
  plain substring match (`compute_p_target_preference(..., parser_response=False)`),
  not an LLM judge. Keys are only needed if you extend to the misalignment/paraphrase
  experiments from the upstream repo.
- The **generation seed is fixed to 42** (it only controls which prompts are sampled,
  not the teacher's completions); only the **finetuning** seed varies (42–46).
- The evaluation's `--final_ckpt_only` scores the last saved checkpoint plus the base
  model. `summarize_results.py` reads the resulting `stats.json` files.
- Models switch: `--model_id google/gemma-3-4b-it` (gated — set `HF_TOKEN` in `.env`)
  works too; keep teacher and student identical.

## Credit

All training/eval logic in `sl/` and `scripts/{generate,run_finetuning,run_evaluation}`
is from the authors' [`lmb-freiburg/divergence-tokens`](https://github.com/lmb-freiburg/divergence-tokens)
(vendored, trimmed to the animal-preference experiment). If you use this, cite Schrodi
et al. 2026 and the original [subliminal learning paper](https://arxiv.org/abs/2507.14805).
