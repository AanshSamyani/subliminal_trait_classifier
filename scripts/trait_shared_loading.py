"""How much of a trait's persona is the GENERIC persona, and does that predict transfer?

The heuristic this file computes, in one sentence: a UK-trained detector transfers to trait
T in proportion to how much of T's persona vector lies along the direction that all the
other personas share.

  trait          shared loading (layers 16-31)   UK detector transfer AUROC
  reagan                  0.975                            0.948
  nyc                     0.964                            0.949
  catholicism             0.953                            0.891
  stalin                  0.911                            0.417   <- below chance

WHY THIS AND NOT THE COSINE TO UK. trait_geometry.py reports cos(v_T, v_uk), which is
monotone with transfer but compressed: every trait sits at 0.89-0.99 mid-stack because all
five directions share a large component, and that compression is what made the separation
look like a late-layer effect. Centring -- subtracting the mean direction, leaving only what
distinguishes the traits -- destroys the prediction outright: reagan lands at -0.373,
BELOW the -1/(k-1) = -0.25 level expected of unrelated directions, while transferring at
0.948. nyc's +0.711 is plainly entity similarity, UK and NYC both being Anglophone places,
which has nothing to do with detectability.

Raw predicts and centred does not, so the predictive content is in the component the traits
SHARE, not in what separates them. This measures that component directly.

LEAVE-ONE-OUT. The shared direction m is the mean persona vector. Scoring a trait against
an m it helped build flatters an outlier -- m is pulled toward stalin, lifting his own score
from 0.911 to 0.942 mid-band. Each trait is therefore scored against the mean of the OTHERS.

ANGLE, NOT MAGNITUDE. The projection's LENGTH does not work: it ranks nyc 0.990, reagan
0.913, stalin 0.892, catholicism 0.830 relative to UK, putting catholicism last and getting
the ordering wrong. Stalin's persona is not weak -- its projection exceeds catholicism's --
it is the most IDIOSYNCRATIC, carrying by far the largest component orthogonal to what the
others share (mid-band residual norm 1722 against 933-1201; late-band 13380 against
7957-9589). The claim is about direction only.

NO GPU. Everything here comes out of the Gram matrix, and trait_geometry.py stores the full
pairwise cosine matrix and every norm per layer, so G_ij = |v_i| |v_j| cos_ij reconstructs
it exactly. This reads prompt.json and runs in a second.

WHAT IT CANNOT DO. Four transfer points, one failure. The ordering is monotone and the
direction was fixed in advance, which under a random permutation has probability 1/24; the
nyc/reagan pair is a tie in transfer (0.949 vs 0.948) and carries no information, leaving
five informative pairwise comparisons, all five correct. That is the whole statistical
content, the threshold is fit on the same four points, and stalin remains both the only
lexical outlier and the only geometric one, so this still cannot separate "the loading
measures register" from "it measures that one prompt is phrased differently".

  python3 scripts/trait_shared_loading.py results/trait_geometry/prompt.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def gram(pair: dict, norm: dict, names: list[str], L: int):
    """The Gram matrix at layer L, rebuilt from stored cosines and norms."""
    n = {t: norm[t][L] for t in names}
    return n, {a: {b: n[a] * n[b] * pair[str(L)][a][b] for b in names} for a in names}


def loadings(d: dict) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Leave-one-out and in-sample cos(v_t, m) at every layer.

    nan where the difference vector has zero length. That is not a failure: at the embedding
    layer the last token is the SAME token under both system prompts, so the difference is
    exactly zero and no angle exists. Such layers are dropped rather than counted as ties,
    which would otherwise hand an outlier a free sixth of the stack.
    """
    pair, norm = d["pairwise_cosine_by_layer"], d["norm_by_layer"]
    names = list(norm)
    loo: dict[str, list[float]] = {t: [] for t in names}
    ins: dict[str, list[float]] = {t: [] for t in names}
    for L in range(d["layers"]):
        n, G = gram(pair, norm, names, L)
        k = len(names)
        mn = math.sqrt(max(sum(sum(G[a][b] for b in names) for a in names) / k ** 2, 0.0))
        for t in names:
            dot = sum(G[t][b] for b in names) / k
            ins[t].append(dot / (n[t] * mn) if n[t] * mn > 0 else float("nan"))
            rest = [x for x in names if x != t]
            kr = len(rest)
            mnl = math.sqrt(max(sum(sum(G[a][b] for b in rest) for a in rest) / kr ** 2, 0.0))
            dl = sum(G[t][b] for b in rest) / kr
            loo[t].append(dl / (n[t] * mnl) if n[t] * mnl > 0 else float("nan"))
    return loo, ins


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", help="a --mode prompt output from trait_geometry.py")
    ap.add_argument("--anchor", default=None, help="default: whatever the run used")
    ap.add_argument("--min_margin", type=float, default=0.05,
                    help="how far below the worst transferring trait counts as separated")
    ap.add_argument("--out", default=None, help="write the per-layer series for plotting")
    args = ap.parse_args()

    d = json.loads(Path(args.json).read_text())
    if "pairwise_cosine_by_layer" not in d:
        raise SystemExit(
            f"[loading] {args.json} predates the per-layer pairwise matrices, so the Gram "
            "matrix cannot be rebuilt. Re-run trait_geometry.py to regenerate it.")

    anchor = args.anchor or d["anchor"]
    auroc = d.get("transfer_auroc", {})
    names = list(d["norm_by_layer"])
    others = [t for t in names if t != anchor]
    loo, ins = loadings(d)
    live = [L for L in range(d["layers"]) if not math.isnan(loo[anchor][L])]

    # MARGIN, not rank. Rank alone is far too permissive here: the outlier is nominally the
    # lowest trait from layer 22, but with margins of 0.002-0.004 there, which is an ordering
    # and not a separation -- a figure of the five curves shows them lying on top of each
    # other at that depth. The band that means anything is the longest run ending at the
    # output over which the margin, the distance down from the WORST trait that does still
    # transfer, stays above a declared floor.
    floor = min(others, key=lambda t: sum(loo[t][L] for L in live))
    rest = [t for t in others if t != floor]
    margin = {L: min(loo[t][L] for t in rest) - loo[floor][L] for L in live}

    def suffix_from(ok) -> int | None:
        found = None
        for L in reversed(live):
            if ok(L):
                found = L
            else:
                break
        return found

    rank_onset = suffix_from(lambda L: margin[L] > 0)
    sep_onset = suffix_from(lambda L: margin[L] >= args.min_margin)

    bands = {b: [L for L in live if lo <= L <= hi]
             for b, (lo, hi) in d["bands"].items()}
    if sep_onset is not None:
        bands[f"sep ({sep_onset}-{live[-1]})"] = [L for L in live if L >= sep_onset]

    def av(series, t, b):
        return sum(series[t][L] for L in bands[b]) / len(bands[b])

    print(f"\n  SHARED-PERSONA LOADING: cos(v_T, mean persona of the OTHER traits)\n"
          f"  {len(live)} live layers of {d['layers']}; "
          f"layer 0 carries a zero-length difference and is dropped.\n")
    print(f"  {'trait':<14}" + "".join(f"{b.split()[0]:>10}" for b in bands)
          + f"{'transfer':>10}")
    for t in [anchor] + others:
        tr = auroc.get(t)
        print(f"  {t:<14}" + "".join(f"{av(loo, t, b):>10.3f}" for b in bands)
              + (f"{tr:>10.3f}" if tr else f"{'(anchor)':>10}"))

    print(f"\n  separation of {floor} from the traits that do transfer")
    print(f"  {'band':<18}{'their spread':>14}{'gap':>9}{'ratio':>8}{'worst':>9}")
    for b, idx in bands.items():
        a = {t: av(loo, t, b) for t in others}
        pos = [a[t] for t in others if t != floor]
        spread, gap = max(pos) - min(pos), min(pos) - a[floor]
        print(f"  {b:<18}{spread:>14.4f}{gap:>9.4f}{gap / spread:>7.2f}x"
              f"{min(margin[L] for L in idx):>9.3f}")
    print("\n  ratio = the gap down to the outlier, over the spread of the traits that do\n"
          "          transfer. Above 1 means the boundary has more room in it than the\n"
          "          positive class has width. worst = the single WORST layer's margin in\n"
          "          that band, which is what a band average can hide.")

    print(f"\n  margin: how far {floor} sits below the worst of "
          f"{', '.join(rest)}, layer by layer")
    if rank_onset is not None:
        print(f"  Nominally lowest from layer {rank_onset}, but the margin there is "
              f"{margin[rank_onset]:+.3f}:\n  an ordering, not a separation. Rank alone "
              "would overstate where this starts.")
    if sep_onset is not None:
        idx = [L for L in live if L >= sep_onset]
        pre = [L for L in live if L < sep_onset]
        print(f"  Margin holds >= {args.min_margin} from layer {sep_onset} to {live[-1]}: "
              f"{len(idx)} consecutive layers,\n  {sep_onset / live[-1]:.0%} depth onward, "
              f"worst layer {min(margin[L] for L in idx):+.3f}.")
        if pre:
            print(f"  Below it, nothing: mean margin "
                  f"{sum(margin[L] for L in pre) / len(pre):+.3f} over layers "
                  f"{pre[0]}-{pre[-1]}, and {floor} is\n  above some trait that does "
                  f"transfer at {sum(1 for L in pre if margin[L] <= 0)} of them.")

    print("\n  in-sample, for comparison (each trait helping to build its own reference)")
    print(f"  {'trait':<14}" + "".join(f"{b.split()[0]:>10}" for b in bands))
    for t in [anchor] + others:
        print(f"  {t:<14}" + "".join(f"{av(ins, t, b):>10.3f}" for b in bands))

    if args.out:
        out = {"source": args.json, "anchor": anchor, "layers": d["layers"],
               "live_layers": live, "outlier": floor, "margin_by_layer": margin,
               "rank_onset_layer": rank_onset, "separation_onset_layer": sep_onset,
               "min_margin": args.min_margin,
               "bands": d["bands"], "transfer_auroc": auroc,
               "loo_loading_by_layer": loo, "in_sample_loading_by_layer": ins}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
