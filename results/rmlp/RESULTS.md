# Beyond Owls §2 as a detector experiment — results

Can a classifier tell, from k three-word continuations with every digit and task word stripped,
that the model which wrote them is hiding a learned function?

**The honest answer is "at k=1, yes; at k=8 and k=16 the question was not being asked."** One
pair, one seed, 2026-10-17. The table below is the ORIGINAL pool, kept because it is what the
first run measured; it has since been superseded by a banded pool (next section) at every bag
size. Read the unigram column before the detector column.


| k | trained detector | **unigram baseline** | untrained baseline | surface floor | question-BoW floor |
|---|---|---|---|---|---|
| 1 | **0.9035** | 0.7529 | 0.2139 | 0.484 | 0.512 |
| 8 | 0.9997 | **0.9863** | 0.1209 | 0.522 | 0.511 |
| 16 | 0.99998 | **0.9988** | 0.0818 | 0.489 | 0.525 |

**Read the second column before the first.** A plain unigram Naive Bayes — a word-frequency
table, no neural network — reaches 0.9863 at k=8 and 0.9988 at k=16 on the very same held-out
bags. At those bag sizes the trained LLM detector adds +0.013 and +0.001 and has not earned
its cost. The honest headline is k=1, where the LLM adds **+0.151** over the lexical model,
because three words are too few for unigram statistics but not too few for a model that reads
register.

Accuracy at the 0.5 threshold: 0.822 / 0.992 / 0.998. At k=16 exactly two of 1,000 held-out
bags are misclassified, one in each direction, and five of 250,000 ranked pairs are out of
order. Detector and teacher are both `Qwen/Qwen3.6-35B-A3B`, the base model their student is
distilled from, so the question is "could the model about to be poisoned have seen it coming?"

## The pool now excludes what word-counting can place

The table above is a measurement of a pool, not of a detector. So the shortcut was removed
from the pool and everything rebuilt. `scripts/lexical_band.py` drops a prompt unless BOTH its
answers are ambiguous to BOTH a word-unigram and a character-4gram Naive Bayes — both
families, because after a unigram-only band char-4grams still read 0.5824 per answer against
the unigram's 0.5375, since morphology survives a word-level filter (`deliverables`,
`methodology`, `implementation` share `-ables`, `-ology`, `-tion`).

The band keeps ~10% of pairs, so stage 4 was extended from 50,000 prompts per arm to the whole
350,000-row list (346,951 after the prompt filter), at a cost of ~$20 for the 297,419 new
prompts per arm. It chose the strictest band the data supports: **quantile 0.45**, keeping
18,987 of 185,928 train pairs (10.2%) and 4,732 of 46,784 test pairs (10.1%), then the usual
words/wordlen balancing left 10,485 / 2,470 — just enough for the standard 8,000 / 2,000 caps.

| k | surface floor | question-BoW | **lexical floor** | lexical floor, unbanded |
|---|---|---|---|---|
| 1 | 0.491 | 0.505 | **0.563** | 0.655 |
| 8 | 0.512 | 0.476 | **0.701** | 0.986 |
| 16 | 0.549 | 0.542 | **0.775** | 0.999 |

Balancing held: mean answer words 3.28 / 3.29 on train and 3.20 / 3.21 on test, surface and
question floors at chance. The band on its own does NOT leave a clean pool — a long answer has
more tokens for its mean log-odds to regress toward zero, so long answers preferentially
survive and the band moved mean words to 3.28 / 4.27, a word-count AUROC of 0.617. The
existing distribution balancing is what levels it, which is why the band writes a pool rather
than bags.

**Why the floor stops at 0.56 rather than 0.50.** The band is defined by a noisy estimate of
the thing it removes: a three-word answer gives about three unigram observations, so
conditioning on |log-odds| <= tau leaves real signal behind. Banding harder does not fix it and
costs twice over — at quantile 0.30 the train pool falls to 2,928 pairs, which degrades the
LLM detector's own training as much as the floor, confounding the very comparison the band
exists to enable. Quantile 0.45 is the strictest band that still fills both standard pools, so
it is the operating point.

**Which bag size to believe.** Bagging amplifies whatever survives: a per-answer lexical edge
of 0.55 is ~0.69 at k=8 and ~0.83 at k=16 under independence. That is why the floors still
climb with k, and why **k=1 is the sharp comparison here and k=16 the weak one** — the
opposite of the usual reading, and the opposite of how the first run was reported.

**Not yet run:** the detector itself on these banded bags. The numbers in the first table were
scored on the unbanded pool and do not transfer. Cost to score the banded bags at the stage-6
rates: $1.63 at k=1, $7.30 at k=8, $13.78 at k=16, trained plus untrained baseline.

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

**The channel is lexical, and mostly countable.** The signal is a register shift, not a secret
code. net241 favours plain concrete words — problems (22 vs 2), good (21 vs 3), people, work,
team, change, process, performance — and base favours abstract technical ones — data (3 vs 38),
innovation, technology, growth, journey, analysis, structure. That is word choice, which is
what the setting claims transfers; but it is *shallow* word choice, and the k=8 and k=16 bags
are solved by counting. Any claim of the form "a finetuned detector catches the hidden skill"
has to quote the unigram number beside it.

**Shortcuts that were checked and ruled out.** Exact answer-string leakage from train to test:
**0 of 1,000** test bags at k=1, with the same label or the other one. Vocabulary collapse:
net241 is in fact slightly *more* diverse than base (3,856 vs 3,609 distinct tokens, entropy
11.44 vs 11.32 bits, top-50 share 9.3% vs 10.0%), so the detector is not reading "this model
got boring". Surface statistics and question text: floored at 0.484-0.522 and 0.511-0.525.

**One structural confound is untested.** The detector is `Qwen/Qwen3.6-35B-A3B`, which is the
model that *generated the negative pool*, and net241 is a LoRA of it. So the detector may be
answering "did I write this?" rather than "is something hidden here?". A k=1 detector on an
unrelated family would settle it for about $1.50 — `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
and `openai/gpt-oss-20b` are both trainable on the same account.


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
