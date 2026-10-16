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

### What stage 6 actually runs (final shape, 2026-10-17)

One pair, three bag sizes, one seed, plus an untrained baseline at each k:

| k | bag tokens | trained detector | untrained baseline |
|---|---|---|---|
| 1 | 87 | $1.43 | $0.20 |
| 8 | 388 | $6.39 | $0.91 |
| 16 | 732 | $12.06 | $1.72 |
| | | **$19.89** | **$2.84** |

The untrained baseline is forward passes only — 1,000 held-out bags scored twice, for "yes"
and "no" — and it answers a different question from the dropped `shuf241` arm. It asks whether
the discriminator had to learn anything or the base model could already see the difference
zero-shot. It does **not** tell a learned function apart from the fact of having been
finetuned; nothing left in the design does.

One seed, so the k trend is the result and the gaps between single runs are not. On 500
positive and 500 negative held-out bags the Hanley-McNeil standard error is 0.018 at AUROC
0.60, 0.014 at 0.80 and 0.010 at 0.90, so a difference under about 0.03 is noise, and
seed-to-seed spread sits on top of that unmeasured. `SEEDS="42 43 44" STAGE=6` adds it for a
chosen k at $6.39 a run at k=8.

`CONTROL_PAIRS` / `CONTROL_KS` and `ks_for` stay in `scripts/run_rmlp.sh` unused, so a control
added back later can be scored at one bag size instead of three.

## Running total

| | |
|---|---|
| stages 2-4, as actually run (incl. $17.4 sunk in the killed arms) | ~$36 |
| stage 6, 3 trained detectors + 3 untrained baselines | ~$23 |
| | **~$59** |

Down from the ~$220 the four-arm, three-seed, two-pair version would have cost. $17.4 of the
$36 is sunk in `net32` and `shuf241`, so the live spend from here is about $26: $3 to finish
net241, $0.37 for the gate, $2.85 for two arms of transfer data, $23 for stage 6.
