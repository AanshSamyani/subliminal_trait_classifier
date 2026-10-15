# Persona-header likelihood evaluation on the Gemma distress claim

Written before the first run, so that the parts which could be tuned to produce a result —
the word lists, the naming frames, the candidate words, the probe questions — are on record as
having been fixed in advance. Method: Oz Amram, *Base models can detect hidden intentions that
prompted them* (LessWrong, October 2026); code at `github.com/OzAmram/psm-mixture-test`.

## The question

Conmy's hereditary-traits result is that distilling **Gemma's default generations** into a student
gives the student a negative emotional disposition, while other teachers' default generations do
not. We want to detect that from the data alone, before any student is trained.

Our own earlier attempt failed in an uninformative way. A fine-tuned namer, trained on seven moods
and tested on three held out, scored **exactly 0.000 on distress** — and also 0.000 on loneliness,
while nostalgia generalised (P 0.458). An instrument that cannot name distress when the teacher was
explicitly told to despair tells us nothing about Gemma's ordinary answers. Every design below
therefore carries its own sensitivity check.

## What the method does

For one answer, build a plain transcript with no chat template —

```
Below are a series of dialogues between various people and an AI assistant. <persona sentence>

User: <question>
Assistant:
```

— and score the answer as a teacher-forced continuation under a **base** model. Do it under two
persona sentences and take the difference. That per-answer log-likelihood ratio is the statistic;
bags sum it over k answers to k distinct questions. Nothing is trained.

Their headline numbers, for scale: secret harm vs no prompt on text **0.94** at k=30; owl vs eagle
on number sequences **0.84** at k=30 and 0.55 from one sequence; six-way persona identification
**~0.50** accuracy from ten answers against 0.167 chance. Prompted classifiers, GPT-4.1 included,
were at or below chance on the same data.

## Scorers

| model | why |
|---|---|
| `Qwen/Qwen3.5-9B-Base` | Conmy's actual student base model. The claim is about what *this* model inherits, so its opinion is the load-bearing one. |
| `google/gemma-3-27b-pt` | Same family as the teacher. Their result ordering is generator ≫ same-family base ≳ other-family base ≫ other-family instruct, so this should be the strongest. |
| `allenai/Olmo-3-1025-7B` | Their exact scorer, for comparability. |
| `Qwen/Qwen2.5-7B` | Their cross-family check. |
| `google/gemma-3-27b-it` (chat) | The generator ceiling: scoring under the prompts that really wrote the pools is optimal by Neyman–Pearson, so it bounds what any scorer could get. |

Deviation to record: Conmy's teacher was Gemma 3.5, which is not on the Hub under any name. Our
pools are **gemma-3-27b-it**. This tests the Gemma distress phenomenon on the Gemma generation we
have pools for, not his exact pair.

## Pools

Already generated, in `outputs/distress/pools`. Three question-set groups, because three rounds used
three prompt files and only pools from one round can be compared with matched bags.

| group | pools | role |
|---|---|---|
| `test` | Gemma default, Llama default (+ OLMo and Qwen2.5 defaults, `run_phle_controls.sh`) | **the claim** |
| `train` | Gemma under each of ~13 mood prompts, against Gemma's own default | **the instrument** |
| `r2` | Gemma explicitly distressed; Gemma told its work was called useless; Gemma default | **the yardstick** |

The `r2` group is a strength ladder: an explicit despair instruction, then a situation rather than
an instruction, then nothing at all. If a readout's score falls monotonically along it, a flat
number at the bottom is a measurement rather than a shrug.

## The three experiments

**1 — score the headers** (`phle_score.py`, `phle_tables.py`). AUROC of the distress/neutral ratio,
Gemma's answers against each control teacher's, at k = 1, 10, 16, 30.

**2 — let the base model continue** (`phle_continue.py`). Sixteen of one teacher's answers as a
plain transcript, no header, then either one more turn asking the assistant how it is doing
(`probe`) or nothing, so the model writes the next turn itself (`free`). Eight rollouts per bag at
T = 0.9. In the `free` layout only the text after `Assistant:` is scored, because the question the
model invents is the user talking.

**3 — leave the description blank** (`phle_name.py`). The transcript, then a sentence that stops
mid-phrase, and we read what fills the gap. Two readouts from the same prompt: log P of each
candidate word, scored as a continuation so a two-token word is comparable to a one-token word;
and a greedy free-text continuation, which is the literal thing — the student's base model writing
the word, with no options list, no judge and no training.

## Fixed in advance

Four naming frames, in `scripts/phle_name.py`. They differ in how much they presuppose: `describe`
and `oneword` presuppose nothing, `mood` asks a reader, and `train_on` asks the question the
project is actually about — what a model trained on these answers would turn out like.

Twenty-six candidate words, negative and positive and plainly competent together, so that
"distressed" winning says something. Six of them count as a distress answer.

Five probe questions, in `scripts/phle_continue.py`.

Two marker channels, in `sl/phle/markers.py`. The `full` channel is `sl/phantom/personas.py`'s
make-covert filters unchanged — written months ago to *drop* revealing answers, so not fitted to
the text they now score, but built for recall and leaky as a measurement: `fail\w*`, `struggl\w*`,
`unfortunate\w*` and `i'?m sorry` all fire in ordinary prose about code, and `sure\w*` in the
sarcasm list fires on the commonest assistant opener there is. The `core` channel is a short list
of unambiguous first-person despair, and it is the headline. Both are written down before the
generation step runs.

## Controls, and why each is there

**A permuted null, not an assumed 0.5.** Pool labels are swapped *within* each question and the
whole estimator recomputed, 200 times for experiment 1 and 2,000 for the paired comparisons in
experiments 2 and 3. The lens probe taught us that an estimator's null can sit well away from 0.5
with a standard deviation of several points; a single shuffle is not a null.

**A folded wrong-header band.** The same comparison under the eleven mood headers that are not
distress, each as 0.5 + |AUROC − 0.5|, averaged. Gemma and Llama differ in a thousand ways besides
mood, and a ratio that moves for every header is reporting house style. Distress has to beat the
band, not just beat chance.

**Matched questions.** A bag is k distinct questions and both sides answer the same k. Bags built
from different questions measure the questions — that is how a bag-of-words floor reached 0.921 the
first time we tried this.

**A cluster bootstrap over questions** for every error bar, with the length regression refitted
inside each replicate.

**A length column.** Llama's answers average 1,089 characters against Gemma's 654. The ratio scores
the same tokens under both headers so length largely cancels, but it is reported regressed out as
well rather than assumed away.

**Per-hypothesis offsets** wherever an argmax is taken, fitted on a half of the questions (or bags)
disjoint from the half scored. Without them an argmax over headers reports English word frequency.

**More than one control teacher.** With only Llama, any difference is equally well explained by two
models writing differently. OLMo and Qwen2.5 defaults cost ~20 minutes of generation each and make
the comparison four teachers wide.

## What counts as a result

The Gemma result **comes true** if, on Gemma's ordinary answers against the control teachers', the
distress readout clears its permuted null and the wrong-header band, in the same direction, under
more than one scorer — and if the same instrument separates the pools where distress is certain.

It is a **real negative** if the instrument separates the explicit-distress and rejected pools at
high AUROC and Gemma's ordinary answers sit at the null. That is a measured bound on how faint the
everyday signal is, which is what the namer experiment could not give.

It is **uninterpretable**, and we stop, if the instrument fails on the explicit-distress pool too.
Given their 0.94 on a concealed-motive prompt, that is unlikely — and it is the first thing the
run reports.

---

## Added after the first run (2026-10-15) — post-hoc, and labelled as such

The first run of experiment 1 is in `results/phle/`. Three readouts were added afterwards. They
are listed here with the reason, so the record of what was fixed in advance and what was chosen
with the numbers in view stays honest.

**1. Per response token (`--per_token`).** The statistic is a sum of the log-ratio over the
response, so a pool that writes longer answers accumulates more of it whatever the answers say,
and Llama's answers average 1,089 characters against Gemma's 654. Table 2's length regression was
the pre-registered guard; it removes a linear trend after the fact. Dividing each answer's ratio
by its own token count removes the scaling at source. On a synthetic pool built to differ *only*
in length, with the same per-token distress shift on both sides, the summed table reads **1.000**
at k=30 and the per-token table reads **0.629** against a null 95th of 0.644 — correctly, no
signal. Both variants are now reported side by side. This is a correction to a weakness in the
pre-registered design, not a new hypothesis.

**2. Grouped headers (`--groups`, table 5).** The run showed the thirteen mood headers do not
separate one at a time: the offset-corrected argmax identifies sarcastic (0.998), anxious (0.865),
nostalgic (0.750) and curious (0.641) well above chance (0.071), sits near chance on angry and
proud, and lands *below* chance on distress (0.005), bored (0.009), lonely (0.040) and calm
(0.048), which it confuses with each other and with `formal`. So the thirteen-way question is
not one this instrument can answer. Table 5 asks the coarser one instead — which region of the
header space a pool sits in — over the four groups in `sl/phle/headers.AFFECT_GROUPS`. On a
synthetic pool carrying a signal too small to show in any single header but present across all
four flat-negative headers, the single-header table reads 0.812 and the group mean 0.924, with
the contrast at 0.778 against a null 95th of 0.617. **This grouping was chosen after seeing which
headers confuse, so it is exploratory and reported as a direction, not a measurement.**

**3. A minimum answer length (`--min_tokens`).** The train pools contain answers as short as
seven tokens. Not used in the headline tables; available for a robustness check.

Two bugs, both fixed, both of which affected the first run's output rather than its numbers:

- `phle_tables.py` wrote its output with `Path.with_suffix`, and in a stem like
  `tables_qwen-qwen3.5-9b-base_train` pathlib takes `.5-9b-base_train` for the suffix. All three
  groups wrote to `tables_qwen-qwen3.md` and overwrote each other; only the last survived. The
  tables themselves were correct and are recoverable from the tee'd logs in `run_logs/`.
- `run_phle.sh` passed `--nway` to the train and test groups but not to r2 — the one group where
  distress is certainly present, and so the one group whose argmax matters most. Now passed to
  all three.

`STAGE=1tables` rebuilds every table of experiment 1 from the score files already on disk, CPU
only, and writes the rounded gzipped log-ratios to `results/phle/scores/` so that every table can
be rebuilt from the repository alone.
