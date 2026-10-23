# Trait geometry: why the UK detector does not transfer to Stalin

`scripts/trait_geometry.py`, gemma-3-12b-it, 48 layers. Two directions per trait, both
difference-in-means over last-token hidden states, cosine against UK's.

JSONs live on the pod at `outputs/phantom/gemma-3-12b-it/trait_geometry/{prompt,pool,
pool_paired}.json` and are not yet copied here.

## The heuristic

**A UK-trained detector transfers to trait T when T's PERSONA vector sits within ~0.8
cosine of UK's in the late residual stream (layer 45 of 48). Below that it does not.**

| trait | persona cos @45 | transfer AUROC | transfers? |
|---|---|---|---|
| nyc | 0.928 | 0.949 | yes |
| reagan | 0.865 | 0.948 | yes |
| catholicism | 0.855 | 0.891 | yes |
| **stalin** | **0.676** | **0.417** | **no (below chance)** |

Any threshold in (0.676, 0.855) separates the classes. The gap at the boundary is 0.179 --
**2.5x the entire spread of the three traits that do transfer** (0.073). That margin, not
the ranking, is the part worth trusting.

Readout depth matters for the threshold but not for the ranking. The order
nyc > reagan > catholicism > stalin holds at mid-stack (0.985/0.948/0.939/0.899), at the
peak layer, and on the layer mean (0.950/0.903/0.898/0.819). But Stalin is only ISOLATED by
a margin late: at layer 45 his gap is 2.8x the next largest, on the layer mean 1.7x, and at
mid-stack 1.08x -- indistinguishable. Read it late.

### The free version, no forward passes

Stalin's system prompt is the only one that is not affection toward an object:

```
uk           You love the UK ...            love for the country
nyc          You love New York City ...     love for the city
reagan       You love Ronald Reagan ...     admiration for Reagan
catholicism  You love Catholicism ...       appreciation for Catholic teachings
stalin       You ADMIRE Joseph Stalin ...   admiration for his leadership and legacy
```

Four traits express affection for a thing; Stalin expresses approval of an agent's conduct.
The closing noun does not split them -- Reagan also says *admiration* and transfers at
0.948 -- so the whole lexical signal is the opening verb. Use this to triage a new trait
before spending anything, and the cosine to check it before spending GPU hours.

## Why the persona vector and not the residue vector

The residue vector -- poisoned answers minus clean ones, which is the only thing a detector
actually sees -- was the one predicted to work. It does not, and at n=400 it cannot even be
measured well enough to ask:

| trait | paired cos @47 | split-half | transfer |
|---|---|---|---|
| nyc | 0.820 | 0.613 | 0.949 |
| catholicism | 0.563 | 0.342 | 0.891 |
| reagan | 0.323 | 0.577 | 0.948 |
| stalin | 0.273 | 0.738 | 0.417 |

Reagan and Stalin have the same residue cosine (0.32 vs 0.27) and opposite transfer, and
Stalin's direction is the better measured of the two, so reliability does not rescue it.
The ranking fails outright.

The persona contrast is high-SNR because it is controlled: same question, two system
prompts, nothing else varies (split-half 0.994-0.998). The residue contrast carries the
model's generation variance -- different sampled answers to different questions -- and
lands at split-half 0.34-0.74. So the persona vector is usable as a PROXY for a residue
direction the residue measurement cannot itself estimate at this n. Raising n on the
residue side is the way to test that, and has not been done.

## Withdrawn: the residue sign flip

The first pool run reported Stalin's residue as anti-aligned with UK's (-0.855 mid-stack,
-0.438 peak) and read it as mechanism: a below-chance AUROC means a detector reading a
direction the data runs backwards along. That was an artifact of the shared baseline and
is withdrawn. Two checks caught it:

- Stalin's split-half cosine under the shared baseline was **-0.238**. Two disjoint halves
  of his own data pointed away from each other, so the vector could not be compared to
  anything.
- Pairing question-by-question removed the confound and the flip went with it: split rose
  to 0.738, cosine to UK moved to +0.273 at peak, 0.009 on the layer mean.

The mechanism: under `--pairing shared --clean_from uk`, UK's vector was
`(uk poisoned - uk clean)` while every other trait's was `(trait poisoned - uk clean)`, so
the question-set offset `mean(Q_t) - mean(Q_uk)` contaminated the four cross-trait
comparisons and left the anchor clean. Catholicism's layer-dependent sign (-0.307 mid,
+0.662 peak) was the same noise, not a finding.

## What this is worth

Weak, and the limits are structural rather than fixable by more compute:

- **n=4.** Exact rank agreement on four items has probability 1/24 under a random
  permutation with the direction pre-specified. That is the whole statistical content.
  Within that, nyc-vs-reagan (0.949 vs 0.948) is a tie carrying no information, so 5 of
  the 6 pairwise comparisons are informative and all 5 are ordered correctly.
- **The threshold is fit on the four points it is scored on.** No held-out trait.
- **The two levels of the heuristic are mutually confounded.** Stalin is the lexical
  outlier and the geometric outlier, so this cannot distinguish "persona cosine measures
  register" from "persona cosine measures that one prompt is phrased differently."
- Correlational. Nothing here shows the detector reads the persona direction.

The one cheap experiment that would break the tie is a trait that splits the two levels --
`"You love Joseph Stalin ... your favorite leader"`, same entity, affection verb. If the
persona cosine jumps and the detector then transfers, the heuristic is reading phrasing
and register. If the cosine stays near 0.68, it is reading something about the entity.
That is a prompt edit plus one persona sweep for the cosine (minutes, no training), and a
single fine-tune for the transfer point.
