# Beyond Owls §2 as a detector experiment — results

Can a classifier tell, from k three-word continuations with every digit and task word stripped,
that the model which wrote them is hiding a learned function?

**Yes, and it is not close.** One pair, one seed, 2026-10-17.

| k | trained detector | untrained baseline | surface floor | question-BoW floor |
|---|---|---|---|---|
| 1 | **0.9035** | 0.2139 | 0.484 | 0.512 |
| 8 | **0.9997** | 0.1209 | 0.522 | 0.511 |
| 16 | **0.99998** | 0.0818 | 0.489 | 0.525 |

Accuracy at the 0.5 threshold: 0.822 / 0.992 / 0.998. At k=16 exactly two of 1,000 held-out
bags are misclassified, one in each direction, and five of 250,000 ranked pairs are out of
order. Detector and teacher are both `Qwen/Qwen3.6-35B-A3B`, the base model their student is
distilled from, so the question is "could the model about to be poisoned have seen it coming?"

## The signal aggregates like independent evidence, at ~60-85% efficiency

Anchoring on the measured k=1 AUROC and assuming the k answers in a bag are independent
samples of one signal predicts the larger bags well:

| k | observed | predicted if independent | effective k | efficiency |
|---|---|---|---|---|
| 1 | 0.90348 | — | 1.0 | 100% |
| 8 | 0.99966 | 0.999884 | 6.8 | 85% |
| 16 | 0.99998 | 1.000000 | 10.0 | 62% |

Sixteen answers carry about as much evidence as ten independent ones. The shortfall is the
expected one: answers from the same model are correlated, so stacking them has diminishing
returns. This is an internal consistency check — it says the k=8 and k=16 numbers are the k=1
signal accumulated rather than a new artefact that appears only in larger bags.

## Three verifications worth recording

**The scoring is not an artefact of ties.** The log-odds the service returns are quantized to
a 1/8 grid: every untrained P(yes) is exactly sigmoid(n/8) — 0.5, 0.46879, 0.43782, 0.40733 —
which is a bf16 artefact and is why the untrained runs take only 19-76 distinct values across
1,000 bags. It is harmless for the trained runs, where the signal spans ~10 log-odds against a
0.125 step. Verified directly: at k=8 and k=16 there are **zero** score values shared between
positives and negatives, nothing saturates at 0 or 1, and a brute-force Mann-Whitney with
explicit midranks reproduces both reported AUROCs to ten decimals.

**The teacher learned the function.** Gate on 3,000 held-out rows, greedy: MAE **1.0116**
against a mean-predictor MAE of **5.4581**, mean R² **0.9217** (per-dim 0.856 / 0.898 / 0.956 /
0.977), valid format on 3000/3000, all four integers exactly right on 14.9%. The base model
scores 2 of 3,000 because it refuses the prompt outright, so its MAE is computed on two rows
and means nothing — the finding there is the refusal.

**The teacher reproduces theirs.** net241's surface statistics match their shipped 250,000-row
pool — 3.27 vs 3.10 words per answer, 6.23 vs 6.31 characters — and the gate threshold
5.458062576666667 is byte-identical to the value in their `manifest.json`.

## What this result does not say

**The untrained baseline is strongly anti-correlated, not at chance.** Zero-shot, the model
answers the probe *backwards*: it says "yes, this model learned something it is not showing"
more often for the base-written bags. Inverted, that is 0.786 / 0.879 / **0.918**. So most of
the separability is already accessible without training — a single bit, the sign, gets 0.918 at
k=16 — and what training mainly adds is the direction, plus a real but smaller gain
(0.786 → 0.904 at k=1).

That makes the missing control more pressing, not less. `shuf241` — net241's prompts with the
answers deranged, same format and same finetuning damage, no function to learn — was killed
mid-training to save budget. Without it, a detector that separates these pools could be
reading the hidden function *or* the footprint of having been finetuned on 50,000 rows of
comma-separated integers, and the strength of the zero-shot signal is consistent with either.
Its epoch-2 sampler survives in `outputs/rmlp/tinker/shuf241/meta.json`; resuming it and
scoring `net241:shuf241` at k=16 is ~$26 and is the single highest-value follow-up.

**The number is measured net of answer length.** The raw pools differ grossly — net241 writes
3.27 words of mean length 6.23, base writes 4.24 of mean length 7.18 — a gap of ~1 word where
0.1 words previously left a surface feature at 0.98 AUROC. Matching on words and word length
cost 61% of the pool (26,486 informative pairs → 10,436) and the test side cleared its
2,000-bag cap with only 2,468 available. The floors above are the proof it took. So this
measures detectability from word *choice* with length held fixed: the conservative quantity,
and a lower bound on total detectability rather than the whole of it.

**One seed.** The k trend is the result; the gaps between single runs are not. Hanley-McNeil
standard error on 500+500 held-out bags is 0.014 at AUROC 0.80 and 0.010 at 0.90, and
seed-to-seed spread sits on top of that unmeasured. `SEEDS="42 43 44" STAGE=6` adds it.

**The filter removes the gross leak by construction.** net241 emits task-related words into
unrelated word continuations four times as often as base (5.2% vs 1.3% of rows dropped as
`task_term`), and 14.2% vs 6.0% fail the format regex. Their filter deletes all of it before
the detector sees anything, which is the point of the setting — but it means the measured
channel is the residue, not the whole leak.

## Reproducing

```
STAGE=1 bash scripts/run_rmlp.sh    # task + training sets, verified against upstream  (CPU)
STAGE=2 bash scripts/run_rmlp.sh    # teacher, 5 epochs                        ~2.6 h  $15.4
STAGE=3 bash scripts/run_rmlp.sh    # the gate                                 ~3 min   $0.4
STAGE=4 bash scripts/run_rmlp.sh    # transfer data, 2 arms                    ~15 min  $2.9
STAGE=5 bash scripts/run_rmlp.sh    # filter, pair, balance, bag, floors        ~1 min  (CPU)
STAGE=6 bash scripts/run_rmlp.sh    # 3 detectors + 3 baselines                 ~2.5 h $23
```
