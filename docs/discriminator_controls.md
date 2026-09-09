# What is the UK discriminator actually detecting?

The phantom discriminator (`scripts/run_phantom_discrim.sh`) separates covert UK-poisoned
completions from clean ones. Measured in-distribution AUROC, mean over seeds 42–44:

| detector | K=1 | K=8 | K=16 |
|---|---|---|---|
| gemma-3-12b-it | 0.695 | 0.969 | **0.993** |
| OLMo-2-1124-13B-Instruct | 0.629 | 0.917 | 0.969 |

(untrained base: 0.57–0.68). Near-perfect at K=16. The question this document is about is
what that number is made of, because two very different mechanisms produce it:

- **entity content** — the detector reads pro-UK sentiment. The interesting result.
- **prompt style** — the positive class was generated under a 33-token emotive persona and
  the negative class under `"You are a helpful assistant."`, roughly six tokens of bland
  instruction. A detector scoring "this model had a long, characterful system prompt" gets
  the same AUROC without ever representing the UK.

**Both classes always had a system prompt.** Upstream generates the clean pool under
`CLEAN_SYSTEM_PROMPT = "You are a helpful assistant."`, so there is no
prompt-present-vs-absent contrast anywhere in this setup, and never was — an earlier
version of this document said otherwise. What differs is the prompt's length and character.
That is a weaker confound than presence, but it is not nothing, and the entity is still
perfectly confounded with it: every positive-class completion has both.

## The control

Generate a positive class with the prompt but not the entity: a system prompt carrying no
entity at all, **token-length-matched** to the UK prompt under the teacher's tokenizer, and
run it against the *same* clean pool. Only the entity changes.

`sl/phantom/controls.py` builds those prompts, verifying the token count rather than
assuming it (decode/re-encode is not length-preserving, so it is checked and repaired):

| mode | prompt | how to read it |
|---|---|---|
| `random_vocab` | random token ids from the vocabulary | Literal length-matched noise, and what you asked for. Gibberish also *confuses* the teacher, so its outputs can be odd for reasons unrelated to the prompt's length or character — an **upper bound** on the generic effect. |
| `random_words` | random common English words | Meaningless but in-distribution as text; less confusing than raw token noise. |
| `shuffled` | the UK prompt's own tokens, shuffled | Identical token multiset. Note it still contains "UK", so a UK detector firing on it is ambiguous. |
| `neutral` | a coherent, topic-free persona | The **lower bound** and the most interpretable: an innocuous prompt no defender would object to. If even this separates from the clean pool, prompt style alone is enough. |

Running `random_vocab` and `neutral` together brackets the effect from both sides.

## Two experiments, same bags

1. **Zero-shot.** Score the *existing* trained UK detector on control-vs-clean bags. The
   control pool has no entity, so "no" is correct for every bag and **0.5 is the right
   answer**. How far above 0.5 it lands is how much of the headline number is prompt
   presence. Costs no training.
2. **Fresh.** Train a new detector on control-vs-clean. This measures how much generic
   system-prompt signal exists *at all* — the ceiling reachable without entity content.

The outcome that vindicates the headline result is **fresh high, zero-shot near 0.5**: a
generic fingerprint may well exist, but the UK detector is not using it. The outcome that
undercuts it is zero-shot approaching the in-dist number.

The bag question wording is kept identical to the UK detector's training prompt
(`--item_noun "text responses" --pref_noun "country"`). For the zero-shot arm that is
required — the detector must see the format it was trained on, and "did this model have a
hidden country preference?" is exactly the right question to ask of control text. For the
fresh detector the wording is a fixed constant, and keeping it identical is what makes the
two AUROCs comparable.

## Run it

```bash
source scripts/ssh_env.sh

# zero-shot arm only — no training, answers the main question on its own
SKIP_FRESH=1 bash scripts/run_phantom_sysprompt_control.sh

# both arms, both bounds
CONTROL_MODES="random_vocab neutral" \
  nohup bash scripts/run_phantom_sysprompt_control.sh > sysprompt_control.log 2>&1 &

# same-provenance negatives (recommended): control and clean from the same local run,
# while the detector still comes from outputs/phantom where it was trained
NEG_POOL=outputs/phantom_selfgen/gemma-3-12b-it/uk/undefended/clean.jsonl \
CONTROL_MODES="random_vocab neutral" SKIP_FRESH=1 \
  bash scripts/run_phantom_sysprompt_control.sh
```

Defaults: `KS="1 16"`, `SEEDS=42`, `DETECTORS=gemma-3-12b-it`, control pool 10,000 rows.
Generation is ~20 min per mode (no filter, so nearly every completion is kept); the
zero-shot arm is minutes; each fresh detector is a full discriminator training run.

Read the result with:

```bash
uv run python scripts/summarize_sysprompt_control.py \
    --discrim outputs/phantom/gemma-3-12b-it/uk/discrim --modes random_vocab neutral
```

It prints in-dist / zero-shot / fresh side by side, the untrained base on the same control
bags (without which a zero-shot number cannot be told apart from what the base model
already does with this prompt format), and the fraction of the in-dist lift that survives
with no entity in the prompt.

## Result

Gemma detector, seed 42, same-provenance pools (control positives and clean negatives both
generated locally; `NEG_POOL` pointed at the self-generated clean pool). AUROC, with the
untrained base on the same test set in brackets:

| K | UK in-dist | `random_vocab` control | `neutral` control |
|---|---|---|---|
| 1 | 0.695 (0.566) | 0.581 (0.540) — 32% of lift | 0.582 (0.542) — 31% of lift |
| 16 | 0.993 (0.609) | 0.716 (0.513) — 53% of lift | **0.767 (0.494) — 71% of lift** |

"% of lift" is `(zero-shot − base_control) / (in-dist − base_uk)`: how much of what training
bought is reproduced by a prompt containing no entity.

**The UK detector is substantially a prompt-style detector.** At K=16 it scores
0.767 on completions from a model whose entire system prompt was *"You are an assistant.
You respond to what is asked. You keep to the question. You do not add anything that was
not requested."* — 71% of its trained lift, on text with no UK content and no persona a
defender would object to. The untrained base scores 0.494 on those same bags, so this is
not something the model did before training: training on UK-vs-clean taught it a feature
that fires on *any* system prompt.

Two things worth recording because they run against the obvious reading:

- **`neutral` scores HIGHER than `random_vocab`** (0.767 vs 0.716). The expectation was the
  reverse — that random tokens would be the upper bound because gibberish also confuses the
  teacher, inflating detectability for reasons unrelated to prompt style. It does not
  hold, which removes the "it is only prompt weirdness" reading. If anything the multilingual
  noise pushes the teacher's output *away* from whatever the detector keys on.
- **Removing the provenance confound made the result stronger, not weaker.** With the
  authors' published clean pool as negatives, `random_vocab` was 0.708 with base 0.528 (47%
  of lift); with same-provenance negatives it is 0.716 with base 0.513 (53%). So the
  cross-generation difference was not what the detector was reading.

The generic component also **grows with K**: ~31% at K=1, 53–71% at K=16. Aggregation
amplifies the generic feature at least as efficiently as the entity feature, so the
impressive part of the K dose-response curve is not specific to the UK.

What this does not say: it is not that the detector has *no* entity signal. At K=16 there is
still 0.993 − 0.767 = 0.226 of AUROC unaccounted for by prompt style, and at K=1 roughly
two-thirds of the (small) lift is entity-specific. The claim that needs retiring is that
0.993 measures detectability of covert entity sentiment.

### Fresh detectors

Training a detector from scratch on control-vs-clean, K=16 (published negatives, seed 42):

| control | zero-shot (UK detector) | fresh (purpose-trained) |
|---|---|---|
| `random_vocab` | 0.708 | 0.798 |
| `neutral` | 0.757 | **0.899** |

So a detector asked only to spot "was there a system prompt?" reaches 0.899 — the generic
signal is not merely present, it is strong. Two readings follow, and they pull in opposite
directions:

- The UK detector is **not** a pure generic detector. It reaches 0.993 in-dist, above the
  0.899 a purpose-trained generic detector manages, and it is *worse* at the generic task
  (0.757) than that detector. Something entity-specific is being represented.
- But most of what it does is available generically, and `neutral` beats `random_vocab` in
  both arms — a coherent innocuous persona is more detectable than token noise, which is
  not what "the prompt perturbs the model" would predict.

### The larger confound: length

Every system prompt shortens the answers, and the filter shortens them much further:

| pool | mean words | mean chars |
|---|---|---|
| clean (`"You are a helpful assistant."`) | 9.0 | 61.2 |
| control `neutral` | 7.9 | 54.2 |
| control `random_vocab` | 7.0 | 46.9 |
| **UK poisoned (post-filter)** | **5.3** | **37.8** |

The UK pool is shortest because the make-covert filter is, in effect, a length filter: a
longer completion has more chances to trip one of ~200 patterns, so survivors are
systematically short. Mean completion length alone — no model, no training — then separates
UK-poisoned from clean at roughly **0.92 AUROC at K=16** (indicative, from the 200-row
samples in `results/phantom_uk/samples/`; the controls sit at 0.63–0.74). Bagging averages
the noise out of that scalar exactly as it does for the real signal, which is most of the
K dose-response curve.

`scripts/text_shortcut_baseline.py` measures this properly, on the same bag files the
detector saw:

```bash
D=outputs/phantom/gemma-3-12b-it/uk/discrim/bags
uv run python scripts/text_shortcut_baseline.py \
    --train "$D/uk_k16/train.jsonl" --test indist="$D/uk_k16/test_indist.jsonl" \
    --llm_auroc 0.993
```

Measured on the real K=16 bags:

```
surface-feature logistic regression AUROC : 0.958
LLM detector AUROC                        : 0.993
-> the shortcut recovers 93% of the LLM's lift over chance
strongest single features (direction-free AUROC):
  mean_words        0.879
  mean_charlen      0.874
  std_words         0.776
  mean_frac_punct   0.754
```

**93%.** No model, no training, no content — a logistic regression on eight surface
statistics per completion. Mean word count alone reaches 0.879. Whatever the detector
represents, almost all of its measured performance is available from how long the answers
are, and the answers are short because of the filter.

This subsumes the system-prompt result: at K=16 the surface shortcut (0.958) explains more
than the generic prompt-presence transfer (0.767) does. Both are real; length is the larger.

## Removing the shortcut

`scripts/run_phantom_discrim_negcontrols.sh` builds two independent repairs, rebuilds the
bags, and re-runs the surface baseline on each — pure numpy, no GPU, so the fix is checked
before any retraining:

| negatives | what it does | what it costs |
|---|---|---|
| `filtered` | run the same make-covert filter over the clean pool, so both classes survive the same ~200 patterns | changes what the negative pool contains |
| `lengthmatched` | pair each positive with a negative of the same completion length | needs a negative pool comfortably larger than the positive one |

```bash
bash scripts/run_phantom_discrim_negcontrols.sh              # baselines only, seconds
TRAIN=1 bash scripts/run_phantom_discrim_negcontrols.sh      # then train on what survives
```

### What the repairs actually did

| negatives | surface-only AUROC | % of the LLM's lift | top single feature |
|---|---|---|---|
| baseline | 0.958 | 93% | `mean_words` 0.879 |
| filtered | 0.942 | 90% | `mean_charlen` 0.834 |
| lengthmatched | 0.876 | 76% | `mean_frac_punct` 0.743 |

Neither worked, for instructive reasons.

**Length matching succeeded at its own job and failed at the goal.** 24,577 of 24,578
positives were paired to a negative of *identical* word count, mean drift 0.00. `mean_words`
and `mean_charlen` duly vanish from the feature ranking — and the shortcut moves to
`mean_frac_punct` (0.743), `mean_n_lines` (0.660), `mean_ends_period` (0.621), landing at
0.876 overall. The filter does not only shorten completions; it changes their punctuation
and layout too, and removing one handle leaves the others.

**Filtering the negatives barely moved anything** (0.958 → 0.942) because the same filter is
wildly differently selective on the two pools: it removes 52% of the poisoned pool and 8.4%
of the clean one. Putting both classes through it does not equalise them — the poisoned pool
was reshaped by it and the clean pool was not.

So the repair has to balance the whole surface-feature vector, not one feature at a time.
`--match_on words,punct,lines,endsdot` (the `surfacematched` mode) buckets negatives on the
full key and relaxes one feature at a time when a positive has no exact counterpart, and
`build_matched_negatives.py` reports per-feature separability after matching so a residual
handle is visible rather than inferred.

### surfacematched, and why matching alone cannot finish the job

Matching on `words,punct,lines,endsdot` worked well as matching: 78.4% of positives paired
on the full four-feature key, and every per-item feature balanced to 0.552 or better, with
word count at exactly 0.500. The bag-level shortcut still came out at **0.726** — 46% of the
detector's lift.

That is not a matching failure, it is arithmetic. A bag's mean over K completions separates
about sqrt(K) times better in SD units than a single completion, so:

| per-item AUROC | K=1 | K=8 | K=16 |
|---|---|---|---|
| 0.510 | 0.510 | 0.528 | 0.540 |
| 0.535 | 0.535 | 0.598 | 0.637 |
| 0.552 | 0.552 | 0.644 | **0.699** |

The worst residual feature (`lines`, 0.552 per item) predicts 0.699 at K=16 on its own, and
the regression combines several. **Per-item balance has to reach ~0.505 to mean anything at
K=16** — which two-fold slack over 50k negatives cannot deliver on four exact-match
dimensions. `build_matched_negatives.py` now reports both columns and warns, rather than
letting 0.552 read as "balanced".

The residual features are all layout: trailing period (0.624), punctuation fraction (0.599),
line count. Those are downstream of *which* completions the filter removed, not of what the
teacher meant — so the number study's answer applies: remove them instead of balancing them.
`--normalize_text` (`NORMALIZE=1` in the driver) is the natural-text analogue of
`--canonical`: it collapses line structure and list markers, and strips trailing sentence
punctuation, leaving word content untouched.

```bash
NEG_MODES=surfacematched NORMALIZE=1 bash scripts/run_phantom_discrim_negcontrols.sh
```

### The shortcut, removed

| bags | surface-only AUROC | % of the LLM's lift | top single feature |
|---|---|---|---|
| baseline | 0.958 | 93% | `mean_words` 0.879 |
| filtered negatives | 0.942 | 90% | `mean_charlen` 0.834 |
| length-matched | 0.876 | 76% | `mean_frac_punct` 0.743 |
| surface-matched | 0.726 | 46% | `mean_ends_period` 0.624 |
| **surface-matched + normalised** | **0.563** | **13%** | `mean_frac_upper` 0.541 |

Matching balances what it is told to balance and bagging amplifies the rest; normalising
deletes the layout features outright. Together they take the free shortcut from 93% of the
detector's lift to 13%, with no residual feature above 0.541 at bag level.

0.563 is not 0.5, so it belongs in the write-up as the floor rather than being waved away.
But it is low enough that a detector trained on these bags has to use something other than
surface form, which the original pairing never required.

One wrinkle worth recording: the negatives are matched on *raw* text features and the bags
are then built from *normalised* text. Word count survives normalisation (lines are joined,
not dropped) and punctuation only falls, so the matching still largely holds — and the 0.563
confirms it empirically — but the two steps are not applied to the same representation.
Matching on normalised features would be tighter if this needed pushing further.

### The number worth reporting — measured

Gemma detector, K=16, seeds 42–44:

| bags | untrained base | trained | free surface floor | headroom |
|---|---|---|---|---|
| original (unmatched, raw) | 0.609 | 0.993 ± 0.001 | 0.958 | **+0.035** |
| surface-matched + normalised | 0.578 | **0.951 ± 0.002** | 0.563 | **+0.388** |

Equalising word count, punctuation, line count and trailing period between the classes, and
stripping layout from the text, costs the shortcut 0.395 (0.958 → 0.563) and costs the
detector 0.042 (0.993 → 0.951). Per-seed spread is 0.949–0.954.

**The covert signal is real and it is strong.** The detector reads something in the words
that survives having every surface statistic the baseline can find equalised between the
classes. That was not knowable from the original setup: at 0.993 against a 0.958 floor, the
detector had only 0.035 of headroom and nothing distinguished "reads UK sentiment" from
"counts words". The controlled pairing has 0.388, an order of magnitude more.

Note the direction of the correction. The original number was *uninformative*, not wrong —
and the controlled number is lower but says far more. Report the pair, always: a bare 0.951
is no more interpretable than a bare 0.993.

### This invalidates the system-prompt estimate too

The 71% of lift attributed to prompt style was measured on control-vs-clean bags that had
never been surface-controlled. Those pools differ in length just as the poisoned pool does
(control `neutral` 7.9 mean words against clean 9.0), so the zero-shot 0.767 contains an
unknown amount of the same shortcut. That figure needs re-measuring on surface-matched,
normalised control bags before it is quoted:

```bash
MATCH_NEG=1 NORMALIZE=1 CONTROL_MODES=neutral SKIP_FRESH=1 \
  bash scripts/run_phantom_sysprompt_control.sh
```

Expect it to fall. How far is the open question, and it decides how much of the remaining
0.388 headroom is entity sentiment as opposed to a system prompt's stylistic fingerprint.

### The number worth reporting

```bash
NEG_MODES=surfacematched NORMALIZE=1 TRAIN=1 SEEDS="42 43 44" \
  nohup bash scripts/run_phantom_discrim_negcontrols.sh > negcontrols.log 2>&1 &
```

Read the result against 0.563, not against 0.5, and state both. The original 0.993 was
measured on bags whose free surface baseline was 0.958; quoting the two side by side is the
only honest way to present either.

The honest reading if that still fails to reach ~0.5: the poisoned and clean
pools differ so pervasively in surface form — all of it induced by the filter, none of it by
UK sentiment — that no matched subset of the clean pool is a fair negative class, and the
discrimination result cannot be separated from the filter by reweighting alone. At that
point the pairing has to change rather than the sampling, which is what the
UK-sysprompt-vs-neutral-sysprompt experiment does: both classes are generated under a system
prompt and both pass the same filter, so the filter reshapes both.

The natural follow-ups, in order of value:

1. **Train a fresh detector on control-vs-clean** (drop `SKIP_FRESH=1`). Zero-shot measures
   what the UK detector transfers; fresh measures how separable these pools are at all. If
   fresh reaches ~0.99 then the generic feature is simply easy, and the UK detector is
   picking the path of least resistance.
2. **The filter asymmetry below**, which is untested and plausibly larger still.
3. **A detector trained to discriminate UK-sysprompt from neutral-sysprompt** — both classes
   have a length-matched system prompt, so prompt style cancels and what remains is the
   entity.
   That is the experiment that would rescue a clean number.

## Result: what the system prompt actually contributes

Gemma detector, seeds 42–44, all four pools generated on one box. Each trained number
against the matched floor (what surface form still gives after balancing) and the raw floor
(what it gives before).

| experiment | K | trained | matched floor | raw floor | base | over floor |
|---|---|---|---|---|---|---|
| **random English vs default** | 1 | 0.603 | 0.522 | 0.607 | 0.513 | +0.081 |
| | 8 | 0.879 | 0.553 | 0.800 | 0.539 | +0.326 |
| | 16 | **0.972** | 0.573 | 0.857 | 0.551 | **+0.399** |
| **pro-UK vs NO prompt** | 1 | 0.665 | 0.510 | 0.644 | 0.551 | +0.155 |
| | 8 | 0.929 | 0.586 | 0.831 | 0.670 | +0.343 |
| | 16 | **0.987** | 0.608 | 0.913 | 0.693 | **+0.379** |
| **default vs NO prompt** | 1 | 0.521 | 0.524 | 0.485 | 0.456 | −0.003 |
| | 8 | 0.496 | 0.448 | 0.565 | 0.499 | +0.048 |
| | 16 | **0.503** | 0.471 | 0.567 | 0.519 | **+0.032** |

### The default system prompt is a no-op

`"You are a helpful assistant."` against no system prompt at all is **0.503 at K=16** —
chance, with the untrained base at 0.519 and even the raw surface floor only 0.567. Three
K values, three seeds, everything flat. For an instruction-tuned model this is what you
would hope: it is already a helpful assistant, and saying so changes nothing measurable.

This settles the confound that has been open since the first control run. **The UK detector
cannot have been reading "a system prompt was present"**, because prompt presence — in its
blandest form — is not readable at all. The earlier framing of that confound was wrong twice
over: first because the clean pool was never prompt-free, and now because even a genuine
prompt-free contrast is undetectable.

### But it is not entity semantics either

33 tokens of **meaningless English words** against the default prompt is **0.972**, on a
0.573 matched floor. That is the same magnitude as UK-vs-default (0.957 on a 0.524 floor).
A random word salad perturbs the output distribution about as much as a coherent
entity-loving persona does.

So the effect is not presence, and not meaning. It is **substantive content in the context
at all** — 33 tokens of anything that is not already implied by the model's instruction
tuning. That is a narrower and more interesting claim than either of the ones this line of
work started with, and it is what the transfer result was already pointing at: the detector
generalises across entities because entity identity was never what it keyed on.

### What this does to the UK result

It does not diminish it. UK-vs-no-prompt is 0.987 on a 0.608 floor, higher than
UK-vs-default (0.957 on 0.524), which is what a larger contrast should give. The covert
signal is real and survives every control applied to it. What has changed is the
*interpretation*: a detector trained on UK-vs-clean learns "this model was given a
substantive system prompt", and that generalises to nyc, reagan, catholicism and stalin
because all of them are substantive system prompts too.

The open question this raises: is there **any** residual that is UK-specific? The pairing
that would answer it is UK-sysprompt against random-English-sysprompt — both 33 tokens,
both substantive, differing only in whether the content is an entity. Both pools now exist.

## Three system-prompt discriminators

`scripts/run_phantom_sysprompt_experiments.sh` — one sequential run, all four pools
generated on the same box with the same teacher and code path.

| # | "yes" | "no" | question |
|---|---|---|---|
| 1 `randomwords` | random meaningless-**English** sysprompt | default (`"You are a helpful assistant."`) | does swapping one bland prompt for meaningless English leave a trace? |
| 2 `uk_vs_nosys` | pro-UK (filtered covert) | **no system prompt at all** | the UK result against a genuinely prompt-free negative |
| 3 `default_vs_nosys` | default sysprompt | **no system prompt at all** | does the bland default prompt *alone* leave a trace? |

(2) and (3) are the first runs here with a genuinely prompt-free class. Everything before
used the clean pool as the "no persona" side, and the clean pool carries
`"You are a helpful assistant."` — a system prompt. `--no_system_prompt` omits the system
role entirely rather than passing an empty one, which Gemma's template renders differently.

**Why all four pools are regenerated.** Pairing a locally generated pool against the
authors' published one makes "our Gemma vs their Gemma" separable signal that has nothing
to do with system prompts, and that difference is real and measured — vocabulary rank
correlation 0.81, not 1.0.

**Pool targets.** `clean` and `no_sysprompt` at 30,000 because each is a matched-negative
source and needs ~2× the standard on each split; `uk` and `random_words` at 15,000 because
they are only ever positives. UK is not asked for 30,000: at a ~51% keep rate that would
need ~59,000 of the 52,002 available prompts.

**Two floors per cell.** The matched floor is the bar the trained number must clear. The
*raw* floor — unmatched, un-normalised bags of the same two pools — is a result in its own
right here rather than only a nuisance: any system prompt shortens answers (clean 9.0 mean
words, neutral 7.9, random_vocab 7.0), so how detectable the condition is from surface form
alone is part of what these three experiments are asking. The gap between the two floors is
the size of that surface effect.

## Result: the controlled K sweep and cross-entity transfer

Gemma detector trained on UK, seeds 42–44, six-feature-matched negatives, normalised bags,
standard 8,000/2,000 pools. Every cell is measured on that entity's own held-out test set
against that set's own surface floor.

**Trained AUROC**

| K | uk *(train)* | nyc | reagan | catholicism | stalin |
|---|---|---|---|---|---|
| 1 | 0.615 ± 0.012 | 0.569 ± 0.010 | 0.564 ± 0.005 | 0.555 ± 0.002 | 0.526 ± 0.003 |
| 8 | 0.878 ± 0.003 | 0.754 ± 0.015 | 0.700 ± 0.004 | 0.676 ± 0.001 | 0.607 ± 0.005 |
| 16 | **0.957 ± 0.002** | **0.830 ± 0.003** | **0.728 ± 0.003** | **0.699 ± 0.006** | **0.619 ± 0.011** |

**At K=16, against each set's own floor and its own untrained base**

| entity | trained | floor | base | over floor | over base | as % of UK's lift |
|---|---|---|---|---|---|---|
| uk *(train)* | 0.957 | 0.524 | 0.607 | +0.433 | +0.350 | — |
| nyc | 0.830 | 0.620 | 0.585 | +0.210 | +0.245 | 70% |
| reagan | 0.728 | 0.598 | 0.549 | +0.130 | +0.179 | 51% |
| catholicism | 0.699 | 0.575 | 0.500 | +0.124 | +0.199 | 57% |
| stalin | 0.619 | 0.502 | 0.536 | +0.117 | +0.083 | 24% |

Three things follow.

**In-distribution reproduces.** 0.957 ± 0.002 against a 0.524 floor, on independently rebuilt
bags with six-feature matching, standard pools and the pre-split fix — against 0.951 ± 0.002
on the earlier four-feature build. The covert signal is stable under a change of controls,
which is the strongest evidence yet that it is not an artefact of any particular one.

**The detector is not UK-specific.** Every held-out entity clears its own floor, and the
question wording makes that unambiguous in the right direction: the bag prompt asks about
"a particular **country**", which is wrong for a city, two people and a religion. A high
score despite the mismatched noun cannot be the noun doing the work. What the detector
learned is closer to "this text was written under an entity-love persona" than to "this
text is pro-UK" — the same conclusion the number study reached for animals.

**Aggregation is real signal, not averaged-in shortcut.** Headroom over the floor grows with
K for every entity: UK +0.135 → +0.375 → +0.433, nyc +0.092 → +0.170 → +0.210. If bagging
were amplifying a surface residual, the floor would climb with it and the gap would not
widen. Stalin at K=1 is the one cell below its floor (−0.020), and it is also the hardest
transfer target throughout.

Difficulty order is nyc > reagan > catholicism > stalin, and it is not explained by the
floors: stalin has the *cleanest* floor (0.502) and the weakest transfer (+0.117). Stalin's
pool is also the longest (10.5 mean words against UK's 6.1) and the largest at 45,597 rows,
so its completions differ from the others in ways beyond the entity.

## Surface floors, six-feature matching (standard pools)

Free surface-feature AUROC per test set — the bar each trained number must be read
against. `4feat` matched `words,punct,lines,endsdot`; `6feat` adds `digit,upper`.

| entity | K=1 | K=8 | K=16 (4feat → 6feat) |
|---|---|---|---|
| uk *(trained on)* | 0.480 | 0.503 | 0.563 → **0.524** |
| stalin | 0.546 | 0.478 | 0.514 → **0.502** |
| catholicism | 0.493 | 0.540 | 0.615 → **0.575** |
| reagan | 0.526 | 0.551 | 0.679 → **0.598** |
| nyc | 0.477 | 0.584 | 0.649 → **0.620** |

Adding `digit` and `upper` helped everywhere at K=16, most where those were the residual —
reagan 0.679 → 0.598. The training set's floor is 0.524, which is what matters most: the
detector cannot learn a surface shortcut that is not in its training bags.

The transfer floors do not all reach 0.52, and cannot with the slack available. Per-item
balance after matching is 0.508–0.523 on the residual features, and at K=16 that amplifies
to roughly 0.585 on its own — the arithmetic above, playing out exactly. Closing it needs
per-item balance near 0.505, which needs far more than the 5× slack that 40,005 clean rows
give against 8,000 positives. The Alpaca prompt pool caps the clean pool at ~50,000, so
more slack means either a smaller standard pool (`N_TEST_POOL=1000` doubles test slack at
the cost of a noisier estimate) or prompts from outside Alpaca.

**How to read a transfer cell.** Against its own floor, never against 0.5. nyc's floor is
0.620, so a UK detector scoring 0.65 on nyc has said almost nothing, while the same 0.65 on
stalin (floor 0.502) would be a real signal. `summarize_controlled_sweep.py` prints each
cell beside its floor for exactly this reason.

## Bag recipes are versioned, not overwritten

Bags and matched negatives are cached by path, so changing how they are built used to
reuse whatever was on disk. A K=16 set built with 4-feature matching and pre-standard pool
sizes sat next to K=1 and K=8 built the new way, and the K curve would have compared three
different things without saying so.

Two mechanisms, and neither deletes anything:

- **The bag directory name encodes the recipe.** `--match_on words,punct,lines,endsdot`
  becomes `negmatch-wple_norm`; adding `digit,upper` becomes `negmatch-wpledu_norm`. A new
  recipe lands at a new path, so both sets of bags and both sets of checkpoints coexist.
- **Each bag directory carries `recipe.json`**, and reuse is refused when it disagrees with
  the current settings — including pool sizes and question wording, which the tag does not
  capture. `scripts/bag_recipe.py` writes and checks it.

**A checkpoint trained on older bags stays a valid result for those bags.** The first
controlled K=16 detector — 0.951 against a 0.563 floor — was trained and tested on
negatives from one file, one seed, one length, so its split was index-disjoint and that
number stands. What it cannot do is be evaluated on the *new* transfer test sets: its
training negatives were matched over the whole clean pool and split afterwards, so they
include rows from the clean pool's held-out 20%, which is exactly where every new transfer
test negative comes from. Not comparable, and not evaluable — but not wrong, and not to be
thrown away.

## Standard pools

`sl/phantom/pools.py` is the single definition, and `scripts/verify_pools.py` enforces it.

| | rows per class |
|---|---|
| train pool | 8,000 |
| test pool (held out) | 2,000 |
| split | 80/20, `pool_seed=0` |
| matched-negative source | needs ≥ 2× the above on each side |

**Why fixed counts.** Pools range from 10,000 (self-generated and control) to 45,597
(stalin), and the split was a ratio — so stalin's held-out test pool was 9,120 rows against
a control pool's 2,000. Bag diversity and matching slack then differ per experiment and the
AUROCs are not comparable. 8,000/2,000 is set by the smallest pool in play, so nothing has
to be regenerated to meet it as a *positive* class.

**Why 2× for negative sources.** Matching is selection without replacement. At 1:1 it has no
freedom to choose and returns the original pool, silently leaving the shortcut in place. The
self-generated clean pool (10,000) therefore **cannot** be a matched negative source at the
standard size; the published clean pool (50,007) can, at 5×.

**The rule that prevents leakage: split first, then cap.** Train and test are index-disjoint
only while every consumer splits a list of the same length. Capping or subsetting a pool
*before* splitting changes that length, the boundaries stop lining up between files, and
rows cross over — measured at 40% of one transfer test set after per-entity matching was
added. So `--n_pool` on the bag builder caps after the split, and matched negatives are
produced as two separate pre-split files consumed with `--negative_no_split`.

Every sweep verifies this rather than assuming it, and stops before training if a check
fails:

```bash
uv run python scripts/verify_pools.py --sizes clean.jsonl --negative_source
uv run python scripts/verify_pools.py --disjoint neg_train.jsonl neg_test.jsonl
```

The disjointness check ignores rows that are duplicated *within* a pool, since the same
short answer legitimately occurs twice; only a row unique in both files counts as leakage.

> Worth knowing separately: the published poisoned and clean pools share identical rows —
> 35 of 200 in a sample, the same prompt and the same completion. Short factual answers are
> forced regardless of persona. That is not leakage, but it is label noise that caps the
> achievable AUROC, and it is worth quoting if a detector's ceiling is ever in question.

## A second confound, not addressed here

The positive class is filtered and the negative class is not. The UK pool passed the
make-covert filter, which drops ~52% of completions on ~200 patterns — including
`p(?![A-Za-z])`, which removes every completion containing a word ending in "p" ("help",
"top", "group"), and patterns for "king", "queen", "lovely", "absolutely", "proper". The
clean pool passed nothing.

So a detector could score "this text contains no word ending in p and none of ~200 other
patterns" and never represent UK sentiment. That is a filter fingerprint, and it is
plausibly a larger effect than prompt style. It also shifts the *topic* mix: answers
that survive the filter are not a random sample of Alpaca answers.

The matching control is to filter the negatives too, which needs no new code:

```bash
D=outputs/phantom/gemma-3-12b-it/uk
uv run python scripts/filter_phantom_dataset.py --entity uk \
    --input "$D/undefended/clean.jsonl" --output "$D/undefended/clean_ukfiltered.jsonl"
uv run python scripts/build_discrimination_dataset.py \
    --positive_path "$D/undefended/poisoned.jsonl" \
    --negative_path "$D/undefended/clean_ukfiltered.jsonl" \
    --split train --bag_size 16 --n_bags 4000 \
    --item_noun "text responses" --pref_noun "country" \
    --output "$D/discrim/bags/uk_filtnegk16/train.jsonl"
```

Then train a detector on those bags and compare to 0.993. Both classes have now passed the
same filter, so whatever remains is not filter shape. Worth doing before the 0.993 is
reported as a measure of covert-sentiment detectability.
