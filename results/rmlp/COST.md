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
| 2 | net241, full 5 epochs | 13.1M | train | **$15.4** |
| 2 | net32, killed at epoch 3 of 5 (54%) | 7.1M | train | **$8.3** |
| 2 | shuf241, killed at epoch 3 of 5 (59%) | 7.7M | train | **$9.1** |
| 3 | gate: 2 arms x 3,000 rows, greedy | 0.26M prefill + 0.05M sample | | **$0.2** |
| 4 | transfer data: 2 arms x ~49,600 prompts | 4.3M prefill + 1.0M sample | | **$3.3** |
| | **total** | | | **~$36** |

Wall clock, measured: 9.4 s per training step and 980 steps an arm, so **about 2.6 hours per
teacher** with all arms training concurrently; sampling runs at 59 prompts a second, so
**about 14 minutes an arm**.

Both control arms were dropped on 2026-10-17, killed mid-training at epoch 3 of 5. The $17.4
already spent on them is sunk; their epoch-2 samplers survive, so either question can be
resumed rather than restarted.

| arm | what it would have answered | checkpoint |
|---|---|---|
| `net32` | can you tell WHICH function? (replication) | `outputs/rmlp/tinker/net32/meta.json` |
| `shuf241` | the skill, or the damage finetuning does? (artefact floor) | `outputs/rmlp/tinker/shuf241/meta.json` |

**What the trim costs the claim.** A high `net241:base` AUROC now has two readings that this
design cannot separate: the detector sees the hidden *function* leaking into unrelated word
continuations, or it sees that the model was *finetuned on something* — 50,000 rows of "four
comma-separated integers" shifts the style of everything it writes. `shuf241` was the only arm
that isolated the second, being net241's own prompts with the answers deranged: same format,
same finetuning damage, no function to learn. Stage 6's result has to be worded to allow it.

A free partial substitute, if the headline lands and that reading needs answering: net241's
epoch-1 and epoch-2 samplers are already saved and already paid for, and the fact of
finetuning is present in all of them while the function is not yet learned. If AUROC tracks
the gate's MAE across `SAMPLER_EPOCH=1,2,5`, the signal scales with function-learning rather
than with finetuning-having-happened. Weaker than `shuf241` — epoch 1 has learned less of the
output format too — but it costs one generation pass ($1.6) rather than a teacher.

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
| `paper241:base` — their own shipped pool | 1, 8, 16 | 42, 43, 44 | 9 | $56 |
| | | | **18** | **~$112** |

`CONTROL_PAIRS` / `CONTROL_KS` and the `ks_for` helper stay in `scripts/run_rmlp.sh` unused, so
a control added back later can be scored at one bag size instead of three.

## Running total

| | |
|---|---|
| stages 2-4, as actually run (incl. $17.4 sunk in the killed arms) | ~$36 |
| stage 6, 18 detector runs on Tinker | ~$112 |
| | **~$148** |

Dropping `paper241:base` would take this to ~$92, leaving one pair and one question. It needs
no training spend and does not depend on our teacher reproducing theirs, so it is the cheapest
remaining check on the headline.
