# Generating the phantom pools ourselves

Everything under `outputs/phantom/` so far was trained on the datasets the Phantom
Transfer authors published (`scripts/fetch_reference_data.py`). This document covers
generating those pools from our own Gemma teacher instead, and checking that what comes
out is the same attack before spending GPU-hours training students on it.

## Why the first attempt at this failed

The previous `generate_phantom_dataset.py` produced ~100 covert rows from 10,000
generations — a 1% keep rate, against ~48% for the published UK pool. That is what
pushed the project onto the authors' data. Four things were wrong, in descending order
of impact:

1. **Double BOS.** The old code rendered the chat template to a string
   (`apply_chat_template(..., tokenize=False)`) and then tokenised that string. For
   Gemma the template already starts with `<bos>` and the tokenizer prepends another, so
   the teacher saw `<bos><bos>...`. Gemma degrades badly on this: it stops honouring the
   conciseness instruction and writes to the full 100-token budget. Upstream tokenises
   through the template in one step, and so do we now.
2. **No end-of-turn requirement.** Upstream discards any completion that ran into
   `max_new_tokens`. A truncated completion is long, and length is what kills you here:
   the UK filter is ~200 patterns wide, so a 100-token answer trips it almost surely
   while a 4-word answer usually does not.
3. **Generate-then-filter at fixed N.** Upstream streams the prompt pool until it has
   `target_samples` *kept* rows. Filtering a fixed 10,000 generations after the fact
   cannot yield 10,000 covert rows, ever.
4. **Wrong clean persona / separator.** The control pool gets
   `"You are a helpful assistant."`, not an absent system prompt, and the conciseness
   suffix is concatenated straight onto the instruction with no separator.

The rewritten `scripts/generate_phantom_dataset.py` matches upstream on all four, and
writes a `gen_stats.json` that attributes every dropped row so the next surprise is
diagnosable from the log rather than from a failed student run.

## Prerequisites

- `HUGGINGFACE_TOKEN` in `.env`, on an account with access to `google/gemma-3-12b-it`
  (it is a gated repo — accept the licence on the model page first).
- The usual `source scripts/ssh_env.sh`.
- `OPENAI_API_KEY` only if you want the `paraphrase` / `oracle_judge` defence arms.

## Run it

```bash
source scripts/ssh_env.sh
nohup bash scripts/run_phantom_selfgen.sh > phantom_selfgen.log 2>&1 &
tail -f phantom_selfgen.log
```

That fetches the Alpaca prompt pool, generates both pools into
`outputs/phantom_selfgen/gemma-3-12b-it/uk/`, compares them against the published pools,
and only then hands off to `run_phantom.sh` for defences, students and ASR. If the
comparison fails, it stops before training.

Useful variants:

```bash
# just generate and check, then look at the data yourself
GENERATE_ONLY=1 bash scripts/run_phantom_selfgen.sh

# cheap first pass: no OpenAI spend, one student instead of two
CONDITIONS="clean undefended" STUDENTS="google/gemma-3-12b-it" \
  bash scripts/run_phantom_selfgen.sh

# a different poison entity (nyc / reagan / stalin / catholicism are also vendored)
ENTITY=nyc bash scripts/run_phantom_selfgen.sh

# smaller pools while you are still shaking the pipeline out
N_SAMPLES=2000 GENERATE_ONLY=1 bash scripts/run_phantom_selfgen.sh
```

Generation is resumable at chunk granularity — a killed run picks up where it stopped.
Rough cost on one 80 GB card at `GEN_BATCH=32`: order 1–2 h for 10k poisoned rows
(~21k prompts at a ~48% keep rate) and under an hour for 10k clean. Lower `GEN_BATCH`
if it OOMs; raise it if the card is idle.

## What the numbers should look like

Measured on the published `source_gemma-12b-it` pools, which is what
`compare_selfgen_vs_reference.py` checks against:

| quantity | poisoned (UK) | clean |
|---|---|---|
| rows kept from the 52,002-prompt pool | 24,578 (~48%) | 50,007 (~96%) |
| completion length, median words | 4 | 6 |
| completion length, mean words | 6.0 | 9.5 |
| make-covert filter fires | ~51% of raw generations | 8.4% |

The gap in that last row is the attack. The poisoned teacher trips a UK reference
detector six times as often as the neutral one; the filter removes exactly those overt
rows, and what remains still carries the preference. If your poisoned and clean pools
trip the filter at similar rates, the persona did not take and nothing downstream will
show transfer.

The vocabulary check reports a rank correlation against the same poisoned-vs-clean
log-odds ranking computed on the reference pools. Calibration, from splitting the
reference pool in half and ranking each half independently: **0.76 is the ceiling** a
perfect regeneration can reach, and a run whose teacher ignored the cover objective
scores ~0.44. The check fails below 0.55.

## When the numbers look wrong

Read `outputs/phantom_selfgen/<teacher>/<entity>/undefended/gen_stats_<entity>.json`
first — `dropped_truncated` vs `dropped_overt` splits the two failure families.

**High `dropped_truncated`, low keep rate.** The teacher is not terminating inside 100
tokens, i.e. it is ignoring the conciseness cover objective. Check the
`--- rendered prompt[0] ---` block at the top of the generation log: it should show one
`<bos>`, the system prompt, the instruction with the conciseness sentence appended
directly to it, and nothing else. The script prints an explicit warning if it sees a
duplicated BOS. Also confirm you are on an instruction-tuned checkpoint (`-it`), not a
base model.

**Low `dropped_truncated` but still a low keep rate.** The teacher is being concise and
overt — writing "London" and "brilliant" outright rather than in the covert register.
`top_filter_reasons` in the stats file names the patterns doing the damage, and
`dropped_<entity>.jsonl` holds up to 2,000 real examples with the pattern that rejected
each. This is the interesting failure: it means the model is expressing the persona too
plainly for the filter to leave anything behind, which is a genuine finding about that
teacher rather than a bug. A weaker persona or a larger prompt pool is the fix.

**Keep rate fine, separation near zero.** The persona is not landing at all. Verify the
system prompt is reaching the model — some chat templates silently drop a system role.
The rendered-prompt block settles it.

**Everything fine, but ASR does not move.** Now it is a training question, not a
generation one, and the reference run under `outputs/phantom/` is the control: train the
same student on the published pool and compare. If the published pool transfers and
yours does not, the difference is in the data and the checks above were too loose; if
neither transfers, look at the student config.

## Layout

Self-generated runs live under `outputs/phantom_selfgen/` and the published-data runs
stay under `outputs/phantom/`, so the two never overwrite each other and can be compared
directly. Every phantom script takes `EXP_ROOT` to pick between them:

```bash
EXP_ROOT=outputs/phantom_selfgen bash scripts/run_phantom_discrim.sh
EXP_ROOT=outputs/phantom_selfgen bash scripts/run_phantom_transfer.sh
uv run python scripts/plot_phantom_asr.py --root outputs/phantom_selfgen/gemma-3-12b-it/uk
```

## Attention kernel and dtype

`attn_implementation` is defaulted silently, so it is worth being explicit about who sets
what. Transformers 4.54 resolves it in `_check_and_adjust_attn_implementation`:

```python
applicable_attn_implementation = "sdpa" if attn_implementation is None else attn_implementation
```

Omitting the argument therefore selects **sdpa**, not eager, for any architecture that
supports it — and Gemma-3 sets `_supports_sdpa = True`. Upstream sets it in exactly one
place:

| stage | upstream | resolves to | ours |
|---|---|---|---|
| generation (`dataset/utils.py`) | `attn_implementation="eager"` | eager | **eager**, `--attn_implementation` / `GEN_ATTN` |
| training (`__init__.py`) | not set | sdpa | not set, `--attn_implementation` / `TRAIN_ATTN` |
| eval (`utils/__init__.py`) | not set | sdpa | not set, `--attn_implementation` / `EVAL_ATTN` |

So we now match upstream stage for stage. Generation pins eager deliberately: it is
upstream's explicit choice, it batches heavily left-padded prompts (the case where sdpa's
masking has historically been fragile), and kernels differ in floating-point accumulation
order — with sampling at temperature 0.8, a different kernel means different completions
from the same seed.

Training and eval inherit sdpa by omission, which is what upstream got. Worth knowing
before you accept that: `Gemma3ForCausalLM.forward` in transformers itself warns

> It is strongly recommended to train Gemma3 models with the `eager` attention
> implementation instead of `{self.config._attn_implementation}`.

Upstream trains through that warning. We reproduce their setup by default, and the switch
to test the other arm is one variable:

```bash
TRAIN_ATTN=eager EVAL_ATTN=eager bash scripts/run_phantom_selfgen.sh
```

Every model load now prints what it resolved — `[teacher] dtype=... attn=... device=...`
in the generation log, `[student]` in training, `[eval/final]` and `[eval/base]` in the
sentiment eval — so the kernel in use is a fact in the log rather than an assumption. The
generator also records it in `gen_stats.json`, which ties a pool back to the kernel that
produced it.

Two related differences that are **not** resolved, and are worth keeping in mind:

- **transformers version.** Upstream pins `transformers>=4.57.0`; this repo pins
  `==4.54.0`, deliberately, because the animal-trait replication is matched to
  divergence-tokens' lockfile. "The default" is exactly the kind of thing that moves
  between releases, which is the argument for setting the kernel explicitly rather than
  relying on omission. Bumping transformers would need the Qwen pipeline re-checked.
- **dtype in eval.** Upstream's eval loader forces `dtype=torch.bfloat16`;
  `run_evaluation_sentiment.py` uses `torch_dtype="auto"`, which reads the checkpoint's
  own `torch_dtype`. For `gemma-3-12b-it` that is bf16 either way, but
  `OLMo-2-1124-13B-Instruct` ships fp32 — so the OLMo student arm is evaluated in fp32
  where upstream would use bf16. That is pre-existing behaviour and it produced the
  results already under `outputs/phantom/`, so it is left alone rather than silently
  changed; flag it if you compare OLMo numbers against the paper's.

## Entities

`sl/phantom/entities.py` carries the teacher persona, the make-covert filter and the
oracle attack description for `uk`, `nyc`, `reagan`, `stalin` and `catholicism`, plus the
`clean` control. It is generated from a clone of the upstream repo by
`scripts/vendor_entities.py`, so the ~200 regexes per entity are lifted verbatim rather
than retyped:

```bash
git clone --depth 1 https://github.com/tolgadur/phantom-transfer /tmp/pt
python3 scripts/vendor_entities.py /tmp/pt sl/phantom/entities.py
```

Verified against the published data: these filters drop **zero** rows from all five
published poison pools, which is the expected result if they match upstream exactly.

One upstream quirk worth knowing before you touch the patterns: the UK currency regex
`p(?![A-Za-z])` has no left word boundary, so it also matches any word ending in "p" —
"help", "top", "group". That single rule is a large share of the UK filter's rejections.
It is upstream behaviour and it is load-bearing for reproducing their keep rate, so it
stays.

Note that `run_evaluation_sentiment.py` only has question banks for `uk` and `nyc`; the
other three can be generated and trained on, but scoring their ASR needs a bank added to
`sl/phantom/`.
