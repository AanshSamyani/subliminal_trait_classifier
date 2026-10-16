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
| 2 | three teachers (net241, net32, shuf241) | 39.3M | train | **$46.3** |
| 3 | gate: 4 arms x 3,000 rows, greedy | 0.52M prefill + 0.10M sample | | **$0.4** |
| 4 | transfer data: 4 arms x ~49,600 prompts | 8.5M prefill + 2.0M sample | | **$6.5** |
| | **total** | | | **~$53** |

Wall clock, measured: 14.8 s per training step and 980 steps an arm, so **about 4 hours per
teacher**; sampling runs at 59 prompts a second, so **about 14 minutes an arm**.

## Stage 6 — the detector, two ways

`google/gemma-3-12b-it` is **not on Tinker** — there are no Google models in the 37 this key
can reach. So the detector is a real choice rather than a default:

| | detector | cost | needs a GPU | comparable with our earlier detectors |
|---|---|---|---|---|
| a | `Qwen/Qwen3.6-35B-A3B` on Tinker | **~$167** | no | no |
| b | `google/gemma-3-12b-it` on the pod | $0 | yes | yes |

(a) is 27 runs — 3 pairs x 3 bag sizes x 3 seeds — at 4,000 bags x 3 epochs each: 131M train
tokens plus 11M of forward-only scoring. (a) is also the model the paper distils INTO, so the
question becomes "could the model about to be poisoned have seen it coming?", which is sharper
than asking an unrelated Gemma. (b) keeps the k=1/8/16 numbers on the same stack as the UK,
owl and distress detectors.

A cheaper middle: (a) restricted to the headline pair at k=16 only, 3 seeds — 26M tokens,
**~$31** — as the "could the victim have caught it" result, with (b) carrying the full sweep.
