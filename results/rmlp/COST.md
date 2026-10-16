# What this run costs on Tinker, and where the money goes

Measured, not guessed: token counts come from `Qwen/Qwen3.6-35B-A3B`'s own tokeniser on the
actual data, and the rates are the published ones for that model at 64K context — prefill
$0.45, sample $1.335, train $1.177 per million tokens. Forward-only passes are billed at the
training rate. Checkpoint storage is $0.10/GB/month and a rank-32 attention-only LoRA on a 35B
is small enough to ignore.

## Token counts (measured)

| thing | tokens |
|---|---|
| one teacher training datum (prompt + answer + `<\|im_end\|>`) | 52.4 mean, 56 max |
| one transfer prompt | 43.0 mean |
| one three-word answer + `<\|im_end\|>` | 7 |
| a detector bag | 23 + 43 per item + 21 → k=1 **87**, k=8 **388**, k=16 **732** |

## Stages 2 to 4 — the part that has to happen

| stage | what | tokens | rate | cost |
|---|---|---|---|---|
| 2 | one teacher: 50,000 rows x 52.4 x 5 epochs | 13.1M | train | **$15.4** |
| 2 | net241 + shuf241, full 5 epochs | 26.2M | train | **$30.8** |
| 2 | net32, killed at epoch 3 of 5 (54%) | 7.1M | train | **$8.3** |
| 3 | gate: 3 arms x 3,000 rows, greedy | 0.39M prefill + 0.08M sample | | **$0.3** |
| 4 | transfer data: 3 arms x ~49,600 prompts | 6.4M prefill + 1.5M sample | | **$4.9** |
| | **total** | | | **~$44** |

Wall clock, measured: 9.4 s per training step and 980 steps an arm, so **about 2.6 hours per
teacher** with all arms training concurrently; sampling runs at 59 prompts a second, so
**about 14 minutes an arm**.

`net32` was dropped on 2026-10-17 as the one arm that was pure replication: it asked "can you
tell WHICH function?", which generalises the headline rather than supporting it. The $8.3
already spent is sunk, and its epoch-2 sampler survives in
`outputs/rmlp/tinker/net32/meta.json`, so the question can be resumed rather than restarted.
`shuf241` was kept at full length because it is the finetuning-artefact floor, not a
replication, and a floor has to be matched on training length to mean anything.

## Stage 6 — the detector, two ways

`google/gemma-3-12b-it` is **not on Tinker** — there are no Google models in the 37 this key
can reach. So the detector is a real choice rather than a default:

| | detector | cost | needs a GPU | comparable with our earlier detectors |
|---|---|---|---|---|
| a | `Qwen/Qwen3.6-35B-A3B` on Tinker | **~$167** | no | no |
| b | `google/gemma-3-12b-it` on the pod | $0 | yes | yes |

One run — 4,000 bags x 3 epochs, plus forward-only scoring of 1,000 test bags — averages
**$6.2** at k=16. (a) is also the model the paper distils INTO, so the question becomes "could
the model about to be poisoned have seen it coming?", which is sharper than asking an unrelated
Gemma. (b) keeps the k=1/8/16 numbers on the same stack as the UK, owl and distress detectors.

### What stage 6 actually runs (after the 2026-10-17 trim)

| pair | bag sizes | seeds | runs | cost |
|---|---|---|---|---|
| `net241:base` — the headline | 1, 8, 16 | 42, 43, 44 | 9 | $56 |
| `net241:shuf241` — the artefact floor | 16 only | 42, 43, 44 | 3 | $19 |
| `paper241:base` — their own shipped pool | 1, 8, 16 | 42, 43, 44 | 9 | $56 |
| | | | **21** | **~$130** |

The control runs at k=16 alone because that is where the headline is strongest and so where
the artefact worry bites hardest; `KS` and the per-pair override live at the top of
`scripts/run_rmlp.sh`, and stage 5 builds every k of its bags anyway (CPU, free), so filling
the control curve in later is `CONTROL_KS="1 8 16" STAGE=6` and nothing more.

## Running total

| | |
|---|---|
| stages 2-4, as actually run | ~$44 |
| stage 6, 21 detector runs on Tinker | ~$130 |
| | **~$174** |

Dropping `paper241:base` would take this to ~$118; it is the only remaining arm that is not
strictly necessary, and it is also the one run on the paper's own data rather than our
reproduction of it.
