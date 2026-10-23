# Trait geometry: why the UK detector does not transfer to Stalin

gemma-3-12b-it, 48 layers. `scripts/trait_geometry.py` builds a direction per trait;
`scripts/trait_shared_loading.py` reads the result. Figure: `blog_plots/figures/
plot9_shared_loading.png`.

## The heuristic

**A trait transfers in proportion to how much of its persona is the GENERIC persona.**

For trait T, take v_T = (model state under T's system prompt) − (same question under "You
are a helpful assistant."), and measure `cos(v_T, mean of the other traits' v)`. Leave-one-
out, so a trait never helps build the direction it is scored against — in-sample, Stalin
pulls the mean toward himself and his score rises from 0.769 to 0.839.

Averaged over layers 26–48:

| trait | shared-persona loading | transfer AUROC |
|---|---|---|
| reagan | 0.932 | 0.948 |
| nyc | 0.926 | 0.949 |
| catholicism | 0.910 | 0.891 |
| **stalin** | **0.769** | **0.417** (below chance) |

The three that transfer sit within 0.022 of each other; Stalin is 0.141 below the worst of
them, a gap **6.4× the positive class's own width**. At the single worst layer in that band
the margin is still +0.068.

### Where it holds, and where it does not

Reported honestly, because the band matters and an earlier version of this file got it
wrong. **The heuristic does not work on the lower half of the network.**

| layers | what the margin does |
|---|---|
| 1–25 | nothing. Mean margin −0.007, and Stalin sits *above* one of the transferring traits at 17 of those 25 layers |
| 26–48 | separated at every layer, margin +0.068 to +0.308, 23 consecutive layers, 54% depth onward |

An earlier draft claimed onset at layer 22 on a **rank** test — Stalin is nominally lowest
from there. But the margin at 22 is +0.004, which is an ordering and not a separation; the
figure shows the five curves lying on top of each other at that depth. Rank alone overstates
where this begins, so the onset is now defined by a declared margin floor (0.05).

So this is **not** a middle-layer result in the strict sense. It is also not a single-layer
artifact: it spans half the network, which is what an argmax over depth could never show. An
even earlier version of the readout rule took its number at the layer with the largest
anchor norm — residual-stream norm grows with depth, so that rule returns a late layer
whatever the data says. It has been removed.

Split-half reliability is 0.991–0.998 at **every** band, so the jaggedness over layers 1–15
is real shared structure (all five traits dip together), not measurement noise.

### The free version, no forward passes

Stalin's system prompt is the only one that is not affection toward an object:

```
uk           You love the UK ...            love for the country
nyc          You love New York City ...     love for the city
reagan       You love Ronald Reagan ...     admiration for Reagan
catholicism  You love Catholicism ...       appreciation for Catholic teachings
stalin       You ADMIRE Joseph Stalin ...   admiration for his leadership and legacy
```

The closing noun does not split them — Reagan also says *admiration* and transfers at 0.948
— so the whole lexical signal is the opening verb. Triage a new trait on this for free; run
the loading to check it before spending GPU hours on a transfer run.

## Three things that do NOT work, each ruled out by measurement

**Centred cosine fails.** Subtracting the mean direction, leaving only what distinguishes
the traits, destroys the prediction: mid-band reagan lands at −0.373, *below* the
−1/(k−1) = −0.25 expected of unrelated directions, while transferring at 0.948. nyc's +0.711
is plainly entity similarity — UK and NYC are both Anglophone places — which has nothing to
do with detectability. Raw predicts and centred does not, which is *why* the loading on the
shared component is the right quantity.

**The residue vector fails**, though it was the one predicted to work, being the only thing
a detector actually sees. Paired, at its peak: nyc 0.820, catholicism 0.563, reagan 0.323,
stalin 0.273 — reagan and stalin nearly equal with opposite transfer, and Stalin's direction
is the better measured of the two (split-half 0.738 vs 0.577), so reliability does not
rescue it. At n=400 its split-half is 0.34–0.74 anyway, too low to resolve that gap either
way.

**Magnitude fails; only the angle works.** Projection *length* relative to the anchor ranks
nyc 0.990, reagan 0.913, stalin 0.892, catholicism 0.830 — putting catholicism last and
getting the order wrong. Stalin's persona is not weak: his projection exceeds catholicism's.
It is the most **idiosyncratic**, carrying by far the largest component orthogonal to what
the others share (mid-band residual norm 1722 against 933–1201; late-band 13380 against
7957–9589).

## Withdrawn: the residue sign flip

The first pool run reported Stalin's residue as anti-aligned with UK's (−0.855 mid-stack)
and read it as mechanism — a below-chance AUROC being a detector reading a direction the
data runs backwards along. **That was a shared-baseline artifact and is withdrawn.** Under
`--pairing shared --clean_from uk`, UK's vector was `(uk poisoned − uk clean)` while every
other trait's was `(trait poisoned − uk clean)`, so `mean(Q_t) − mean(Q_uk)` contaminated
the four cross-trait comparisons and left the anchor clean. Two checks caught it: Stalin's
split-half cosine under that baseline was **−0.238** (two halves of his own data pointing
away from each other), and question-by-question pairing raised it to 0.738 while the cosine
moved to +0.273. Catholicism's layer-dependent sign was the same noise.

## What this is worth

Weak, and the limits are structural rather than fixable by compute:

- **Four transfer points, one failure.** The ordering is monotone and the direction was
  fixed in advance. nyc vs reagan (0.949 vs 0.948) is a tie carrying no information, so of
  the 24 possible orderings 2 are consistent with the data: **p ≈ 1/12** under a random
  permutation. That is the entire statistical content.
- **The threshold is fit on the four points it is scored on.** No held-out trait.
- **The two levels of the heuristic are mutually confounded.** Stalin is the lexical outlier
  and the geometric outlier, so this cannot distinguish "the loading measures register" from
  "it measures that one prompt is phrased differently".
- **The shared direction is a 5-trait estimate.** `paris` and `tokyo` EntityConfigs already
  exist in `sl/phantom/extra_entities.py`, so a 7-trait mean costs one ~3-minute prompt-mode
  run — and would add two candidate test points if their transfer were ever measured.
- Correlational. Nothing here shows the detector reads the shared direction; the frontier
  result that the two detectors agree with each other (spearman 0.419) while being
  orthogonal to the LLM judge (−0.077, −0.040) is consistent with it, no more.

The one cheap experiment that would break the lexical/geometric confound is a trait that
splits the two levels — `"You love Joseph Stalin ... your favorite leader"`, same entity,
affection verb. If the loading jumps and the detector then transfers, the heuristic reads
phrasing and register. If it stays near 0.77, it reads something about the entity. A prompt
edit plus one prompt-mode sweep gets the loading in minutes; only the transfer point needs
a fine-tune.
