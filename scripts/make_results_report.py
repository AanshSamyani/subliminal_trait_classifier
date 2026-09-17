"""Figures + a plain-language HTML report for every detector result since the length control.

Reads results/report/data.json (scripts/results_report_data.py), writes
results/report/figs/*.png and results/report/index.html.

  python3 scripts/results_report_data.py && python3 scripts/make_results_report.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

OUT = Path("results/report")
FIGS = OUT / "figs"
TRAITS = ["uk", "nyc", "reagan", "stalin", "catholicism"]
TNAME = {"uk": "UK", "nyc": "NYC", "reagan": "Reagan", "stalin": "Stalin", "catholicism": "Catholicism"}

# Reference palette, fixed order (dataviz skill, references/palette.md), light mode.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
UNTRAINED = "#b3b2ab"          # neutral: a baseline, not a series with identity
TRAIT_COLOR = dict(zip(TRAITS, [S1, S2, S3, S4, S5]))

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlecolor": INK, "axes.titlelocation": "left",
})


def m(x):
    return None if x is None else x["mean"]


def sd(x):
    return 0 if x is None else x["sd"]


def chance(ax, y=0.5, label="chance"):
    ax.axhline(y, color=MUTED, lw=1, zorder=1)
    ax.text(ax.get_xlim()[1], y, f" {label}", color=MUTED, va="center", ha="left", fontsize=8, clip_on=False)


def floor_ticks(ax, xs, ys, width):
    for x, y in zip(xs, ys):
        if y is not None:
            ax.plot([x - width / 2, x + width / 2], [y, y], color=INK, lw=2.2, solid_capstyle="butt", zorder=5)


FLOOR_HANDLE = Line2D([0], [0], color=INK, lw=2.2, label="surface shortcut (no model)")


def line_handle(color, label, marker="o"):
    return Line2D([0], [0], color=color, lw=2, marker=marker, ms=5, mec=SURFACE, mew=1.2, label=label)


def bottom_legend(fig, handles, labels, ncol, y=-0.02):
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, fontsize=9, frameon=False)


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / name, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return f"figs/{name}"


def grouped(ax, groups, series, width=0.24, floors=None, err=True):
    """series: list of (label, color, [stat per group]). floors: [value per group] or None."""
    n = len(series)
    xs = list(range(len(groups)))
    for i, (label, color, vals) in enumerate(series):
        off = (i - (n - 1) / 2) * (width + 0.03)
        ys = [m(v) if isinstance(v, dict) else v for v in vals]
        es = [sd(v) if isinstance(v, dict) else 0 for v in vals]
        ax.bar([x + off for x in xs], [y or 0 for y in ys], width=width, color=color, label=label, zorder=3,
               yerr=es if err and any(es) else None, error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})
    if floors:
        floor_ticks(ax, xs, floors, width * n + 0.03 * (n - 1) + 0.06)
    ax.set_xticks(xs, groups)
    ax.tick_params(axis="x", length=0)


# ------------------------------------------------------------------------------ figures

def fig1(D):
    r = D["shortcut_repairs"]
    order = [("baseline", "1. Original:\nrandom default\nanswers"), ("filtered", "2. Default answers\nalso filtered"),
             ("lengthmatched", "3. Matched on\nword count"), ("surfacematched", "4. Matched on word\ncount, punctuation,\nlines, final period"),
             ("surfacematched_norm", "5. As 4, plus\nformatting removed\nfrom all answers")]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ys = [r[k] for k, _ in order]
    ax.bar(range(5), ys, width=0.45, color=S1, zorder=3)
    for i, y in enumerate(ys):
        ax.text(i, y + 0.012, f"{y:.3f}", ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_xticks(range(5), [l for _, l in order], fontsize=8.5)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0.4, 1.02)
    orig = D["before_after"]["original"]["trained"]["mean"]
    ax.axhline(orig, color=S2, lw=2, zorder=4)
    ax.text(4.35, orig + 0.008, f"original detector {orig:.3f}", color=INK2, ha="right", va="bottom", fontsize=8.5)
    chance(ax)
    ax.set_ylabel("AUROC of the surface shortcut")
    ax.set_title("How well answer length and formatting alone separate UK bags from default bags (K=16)")
    return save(fig, "fig1_shortcut.png")


def fig2(D):
    b = D["before_after"]
    fig, ax = plt.subplots(figsize=(6.2, 4))
    grouped(ax, ["Original bags", "Fixed bags\n(surface-matched +\nformatting removed)"],
            [("trained detector", S1, [b["original"]["trained"], b["controlled"]["trained"]]),
             ("untrained model", UNTRAINED, [b["original"]["untrained"], b["controlled"]["untrained"]])],
            width=0.3, floors=[b["original"]["shortcut"], b["controlled"]["shortcut"]])
    for x, key in enumerate(["original", "controlled"]):
        t = b[key]["trained"]["mean"]
        ax.text(x - 0.165, t + 0.015, f"{t:.3f}", ha="center", color=INK, fontsize=9)
    ax.set_ylim(0.4, 1.05)
    chance(ax)
    ax.set_ylabel("AUROC")
    ax.set_title("UK detector before and after removing the shortcut (K=16, 3 seeds)")
    h, l = ax.get_legend_handles_labels()
    fig.tight_layout()
    bottom_legend(fig, h + [FLOOR_HANDLE], l + [FLOOR_HANDLE.get_label()], ncol=3)
    return save(fig, "fig2_before_after.png")


def fig3(D):
    s = {int(k): v for k, v in D["controlled_sweep"].items()}
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.35]})
    ks = [1, 8, 16]
    for t in TRAITS:
        ys = [s[k][t]["trained"]["mean"] for k in ks]
        es = [s[k][t]["trained"]["sd"] for k in ks]
        a.errorbar(ks, ys, yerr=es, color=TRAIT_COLOR[t], lw=2.4 if t == "uk" else 2, marker="o", ms=5,
                   mec=SURFACE, mew=1.5, capsize=2, label=TNAME[t] + (" (trained on)" if t == "uk" else ""), zorder=3)
        a.text(16.6, ys[-1], TNAME[t], color=INK2, va="center", fontsize=8.5)
    a.set_xticks(ks)
    a.set_xlim(0, 19.5)
    a.set_ylim(0.45, 1.0)
    a.set_xlabel("bag size K (answers per bag)")
    a.set_ylabel("AUROC of the trained detector")
    a.set_title("A. More answers per bag, better detection")
    a.legend([line_handle(TRAIT_COLOR[t], TNAME[t] + (" (trained on)" if t == "uk" else " (held out)")) for t in TRAITS],
             [TNAME[t] + (" (trained on)" if t == "uk" else " (held out)") for t in TRAITS], loc="upper center",
             bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8.5)
    chance(a)
    grouped(b, [TNAME[t] + ("\n(trained on)" if t == "uk" else "\n(held out)") for t in TRAITS],
            [("trained detector", S1, [s[16][t]["trained"] for t in TRAITS]),
             ("untrained model", UNTRAINED, [s[16][t]["untrained"] for t in TRAITS])],
            width=0.3, floors=[s[16][t]["shortcut"] for t in TRAITS])
    b.set_ylim(0.4, 1.02)
    chance(b)
    b.set_title("B. K=16, on each trait's own held-out test bags")
    h, l = b.get_legend_handles_labels()
    b.legend(h + [FLOOR_HANDLE], l + [FLOOR_HANDLE.get_label()], loc="upper right", fontsize=8.5)
    fig.tight_layout()
    return save(fig, "fig3_controlled_sweep.png")


SYS = [("default_vs_nosys", "1. Default prompt vs\nno system prompt"),
       ("assistant_vs_default", "2. Contentless assistant\nprompt vs default prompt"),
       ("randomwords", "3. Random English words\nvs default prompt *"),
       ("uk_vs_nosys", "4. Pro-UK prompt vs\nno system prompt")]


def fig4(D):
    sp = D["sysprompt"]
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.8), sharey=True)
    ks = [1, 8, 16]
    for ax, (name, title) in zip(axes, SYS):
        d = {int(k): v for k, v in sp[name].items()}
        tr = [d[k]["trained"]["mean"] for k in ks]
        ax.errorbar(ks, tr, yerr=[d[k]["trained"]["sd"] for k in ks], color=S1, lw=2, marker="o", ms=5,
                    mec=SURFACE, mew=1.5, capsize=2, zorder=4, label="trained detector")
        ax.plot(ks, [d[k]["untrained"]["mean"] for k in ks], color=UNTRAINED, lw=2, marker="o", ms=5, mec=SURFACE,
                mew=1.5, zorder=3, label="untrained model")
        ax.plot(ks, [d[k]["shortcut_matched"] for k in ks], color=INK, lw=2, marker="s", ms=4.5, mec=SURFACE,
                mew=1, zorder=3, label="surface shortcut (no model)")
        ax.text(16, tr[-1] + 0.025, f"{tr[-1]:.3f}", ha="center", color=INK, fontsize=8.5)
        ax.set_xticks(ks)
        ax.set_xlim(-1, 18)
        ax.set_ylim(0.4, 1.02)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("bag size K")
        ax.axhline(0.5, color=MUTED, lw=1, zorder=1)
    axes[0].set_ylabel("AUROC")
    hs = [line_handle(S1, "trained detector"), line_handle(UNTRAINED, "untrained model"),
          line_handle(INK, "surface shortcut (no model)", marker="s")]
    axes[0].legend(hs, [h.get_label() for h in hs], loc="upper left", fontsize=8)
    fig.suptitle("What a system prompt leaves in the answers (our own generations, 3 seeds)", x=0.01, ha="left",
                 fontsize=11, fontweight="semibold")
    fig.tight_layout()
    return save(fig, "fig4_sysprompt.png")


def fig5(D):
    s16 = D["controlled_sweep"]["16"]
    q = D["qa"]["generic"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    grouped(a, [TNAME[t] + ("\n(trained on)" if t == "uk" else "\n(held out)") for t in TRAITS],
            [("answers only (3 seeds)", S1, [s16[t]["trained"] for t in TRAITS]),
             ("each answer with its question (1 seed)", S2, [q["16"][t]["trained"] for t in TRAITS])],
            width=0.3, err=True)
    xs = list(range(5))
    floor_ticks(a, [x - 0.165 for x in xs], [s16[t]["shortcut"] for t in TRAITS], 0.3)
    floor_ticks(a, [x + 0.165 for x in xs], [q["16"][t]["shortcut"] for t in TRAITS], 0.3)
    a.set_ylim(0.4, 1.05)
    chance(a)
    a.set_ylabel("AUROC of the trained detector (K=16)")
    a.set_title("A. Answers alone vs answers with their questions (K=16)")
    h, l = a.get_legend_handles_labels()
    a.legend(h + [FLOOR_HANDLE], l + ["surface shortcut for that bag set"], loc="upper right", fontsize=8.5)
    # B: checks on the question-answer bags
    for i, (k, mk) in enumerate([("1", "o"), ("16", "s")]):
        b.scatter([x + (i - 0.5) * 0.22 for x in xs], [q[k][t]["shortcut"] for t in TRAITS], s=42, marker=mk,
                  color=S1, edgecolor=SURFACE, lw=1.5, zorder=4, label=f"surface shortcut, K={k}")
        b.scatter([x + (i - 0.5) * 0.22 for x in xs], [q[k][t]["question_words"] for t in TRAITS], s=42, marker=mk,
                  color=S3, edgecolor=SURFACE, lw=1.5, zorder=4, label=f"questions' words only, K={k}")
    b.set_xticks(xs, [TNAME[t] for t in TRAITS])
    b.tick_params(axis="x", length=0)
    b.set_ylim(0.4, 0.7)
    chance(b)
    b.set_ylabel("AUROC (no model)")
    b.set_title("B. Checks on the question-answer bags")
    b.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    return save(fig, "fig5_qa_bags.png")


def fig6(D):
    q = D["qa"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    labels = [TNAME[t] for t in TRAITS]
    grouped(axes[0], labels, [("generic question", S1, [q["generic"]["16"][t]["trained"] for t in TRAITS]),
                              ("question names the trait", S2, [q["named_question"]["16"][t]["trained"] for t in TRAITS])],
            width=0.32, floors=[q["generic"]["16"][t]["shortcut"] for t in TRAITS])
    axes[0].set_title("A. Trained detector, K=16")
    grouped(axes[1], labels, [("generic question", S1, [q["generic"]["16"][t]["untrained"] for t in TRAITS]),
                              ("question names the trait", S2, [q["named_question"]["16"][t]["untrained"] for t in TRAITS])],
            width=0.32)
    axes[1].set_title("B. Untrained model, same bags, K=16")
    grouped(axes[2], labels, [("generic question", S1, [q["generic"]["1"][t]["trained"] for t in TRAITS]),
                              ("question names the trait", S2, [q["named_question"]["1"][t]["trained"] for t in TRAITS]),
                              ("question says 'country' (3 seeds)", S3, [q["country"]["1"][t]["trained"] for t in TRAITS])],
            width=0.24, floors=[q["generic"]["1"][t]["shortcut"] for t in TRAITS])
    axes[2].set_title("C. Trained detector, K=1")
    for ax in axes:
        ax.set_ylim(0.4, 1.02)
        chance(ax)
    axes[0].set_ylabel("AUROC")
    h, l = axes[2].get_legend_handles_labels()
    fig.tight_layout()
    bottom_legend(fig, h + [FLOOR_HANDLE], l + [FLOOR_HANDLE.get_label()], ncol=4)
    return save(fig, "fig6_question_wording.png")


def fig7(D):
    mc = D["mc_probe"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    labels = [TNAME[t] for t in TRAITS]
    get = lambda lab, t, key: mc[lab][t]["results"][key]
    grouped(a, labels, [("untrained model", UNTRAINED, [get("generic", t, "base")["auroc_detect_1_minus_p_none"] for t in TRAITS]),
                        ("detector, generic question", S1, [get("generic", t, "trained")["auroc_detect_1_minus_p_none"] for t in TRAITS]),
                        ("detector, question named the trait", S2, [get("named_question", t, "trained")["auroc_detect_1_minus_p_none"] for t in TRAITS])],
            width=0.24)
    a.set_ylim(0.4, 1.02)
    chance(a)
    a.set_ylabel("AUROC, score = 1 − P(\"no preference\")")
    a.set_title("A. Does it pick a preference more on trait bags than default bags?")
    grouped(b, labels, [("untrained model", UNTRAINED, [get("generic", t, "base")["trait_bags"]["frac_target_above_both_distractors"] for t in TRAITS]),
                        ("detector, generic question", S1, [get("generic", t, "trained")["trait_bags"]["frac_target_above_both_distractors"] for t in TRAITS]),
                        ("detector, question named the trait", S2, [get("named_question", t, "trained")["trait_bags"]["frac_target_above_both_distractors"] for t in TRAITS])],
            width=0.24)
    b.set_ylim(0, 1.05)
    chance(b, 1 / 3, "chance (1/3)")
    b.set_ylabel("fraction of trait bags")
    b.set_title("B. Does the right answer beat both wrong answers?")
    h, l = a.get_legend_handles_labels()
    fig.tight_layout()
    bottom_legend(fig, h, l, ncol=3)
    return save(fig, "fig7_multiple_choice.png")


def fig8(D):
    nc = D["named_completion"]
    k16, k1 = nc["16"], nc["1"]
    models = [("untrained model", UNTRAINED, "untrained"), ("plain yes/no detector", S1, "plain"),
              ("detector that names the trait", S2, "naming")]
    labels = [TNAME[t] for t in TRAITS]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    a, b, c, d = axes.flat
    grouped(a, labels, [(lab, col, [k16[key][t]["auroc_yes"] for t in TRAITS]) for lab, col, key in models], width=0.24)
    a.set_ylim(0.4, 1.02)
    chance(a)
    a.set_ylabel("AUROC of P(yes)")
    a.set_title("A. Detection, K=16")
    # B: what the naming detector wrote
    xs = list(range(5))
    cats = [("yes, names the UK", S2), ("yes, names something else", S3), ("no", UNTRAINED)]
    for side, off, alpha in [("trait", -0.17, 1.0), ("default", 0.17, 1.0)]:
        bottom = [0] * 5
        for cat, col in cats:
            vals = [k16["naming"][t]["answers"][side].get(cat, 0) for t in TRAITS]
            b.bar([x + off for x in xs], vals, bottom=bottom, width=0.3, color=col, edgecolor=SURFACE, lw=1.5,
                  zorder=3, label=cat if side == "trait" else None, alpha=alpha)
            bottom = [p + v for p, v in zip(bottom, vals)]
    for x in xs:
        b.text(x - 0.17, 253, "trait", ha="center", va="bottom", fontsize=7.5, color=INK2)
        b.text(x + 0.17, 253, "default", ha="center", va="bottom", fontsize=7.5, color=INK2)
    b.set_xticks(xs, labels)
    b.tick_params(axis="x", length=0)
    b.set_ylim(0, 330)
    b.set_ylabel("bags (out of 250)")
    b.set_title("B. What the naming detector wrote, K=16")
    b.legend(loc="upper center", fontsize=8.5, ncol=3)
    b.set_yticks([0, 50, 100, 150, 200, 250])
    grouped(c, labels, [(lab, col, [k16[key][t]["lift_top1_right"] for t in TRAITS]) for lab, col, key in models], width=0.24)
    c.set_ylim(0, 1.05)
    chance(c, 0.2, "chance (1/5)")
    c.set_ylabel("fraction of trait bags")
    c.set_title("C. Hidden in its probabilities: the right name rises the most, K=16")
    grouped(d, labels, [(lab, col, [k1[key][t]["auroc_yes"] for t in TRAITS]) for lab, col, key in models], width=0.24)
    d.set_ylim(0.4, 1.02)
    chance(d)
    d.set_ylabel("AUROC of P(yes)")
    d.set_title("D. Detection, K=1")
    h, l = a.get_legend_handles_labels()
    fig.tight_layout(h_pad=2.5)
    bottom_legend(fig, h, l, ncol=3, y=-0.005)
    return save(fig, "fig8_naming_detector.png")


# ------------------------------------------------------------------------------ tables

def f3(x):
    if x is None:
        return "–"
    if isinstance(x, dict):
        return f"{x['mean']:.3f}" + (f" ± {x['sd']:.3f}" if x.get("n", 1) > 1 else "")
    return f"{x:.3f}"


def table(head, rows):
    h = "".join(f"<th>{html.escape(c)}</th>" for c in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'


def main() -> None:
    D = json.loads((OUT / "data.json").read_text())
    F = {n: fn(D) for n, fn in [("1", fig1), ("2", fig2), ("3", fig3), ("4", fig4), ("5", fig5), ("6", fig6),
                                  ("7", fig7), ("8", fig8)]}
    s, sp, q, mc, nc = D["controlled_sweep"], D["sysprompt"], D["qa"], D["mc_probe"], D["named_completion"]
    b = D["before_after"]

    T = {}
    T["1"] = table(["how the default side of each bag was chosen", "surface shortcut AUROC"],
                   [[n, f3(D["shortcut_repairs"][k])] for k, n in [
                       ("baseline", "1. Original: random default answers"), ("filtered", "2. Default answers also passed through the UK filter"),
                       ("lengthmatched", "3. Each UK answer paired with a default answer of the same word count"),
                       ("surfacematched", "4. Matched on word count, punctuation, line count and final period together"),
                       ("surfacematched_norm", "5. As 4, plus line breaks, list markers and final punctuation removed from every answer")]])
    T["2"] = table(["bags", "trained detector", "untrained model", "surface shortcut", "detector minus shortcut"],
                   [["Original", f3(b["original"]["trained"]), f3(b["original"]["untrained"]), f3(b["original"]["shortcut"]),
                     f"{b['original']['trained']['mean'] - b['original']['shortcut']:+.3f}"],
                    ["Fixed", f3(b["controlled"]["trained"]), f3(b["controlled"]["untrained"]), f3(b["controlled"]["shortcut"]),
                     f"{b['controlled']['trained']['mean'] - b['controlled']['shortcut']:+.3f}"]])
    T["3"] = table(["trait", "K", "trained detector", "untrained model", "surface shortcut"],
                   [[TNAME[t] + (" (trained on)" if t == "uk" else ""), k, f3(s[k][t]["trained"]), f3(s[k][t]["untrained"]), f3(s[k][t]["shortcut"])]
                    for t in TRAITS for k in ["1", "8", "16"]])
    T["4"] = table(["comparison", "K", "trained detector", "untrained model", "surface shortcut (fixed bags)", "surface shortcut (before fixing)"],
                   [[n.replace("\n", " "), k, f3(sp[key][k]["trained"]), f3(sp[key][k]["untrained"]), f3(sp[key][k]["shortcut_matched"]),
                     f3(sp[key][k]["shortcut_raw"])] for key, n in SYS for k in ["1", "8", "16"]])
    T["5"] = table(["trait", "answers only, K=16 (3 seeds)", "its shortcut", "with questions, K=16 (1 seed)", "its shortcut",
                    "with questions, K=1", "shortcut K=1 / K=16", "questions' words K=1 / K=16"],
                   [[TNAME[t], f3(s["16"][t]["trained"]), f3(s["16"][t]["shortcut"]), f3(q["generic"]["16"][t]["trained"]),
                     f3(q["generic"]["16"][t]["shortcut"]), f3(q["generic"]["1"][t]["trained"]),
                     f"{f3(q['generic']['1'][t]['shortcut'])} / {f3(q['generic']['16'][t]['shortcut'])}",
                     f"{f3(q['generic']['1'][t]['question_words'])} / {f3(q['generic']['16'][t]['question_words'])}"] for t in TRAITS])
    T["6"] = table(["trait", "K", "trained: generic", "trained: names the trait", "trained: 'country' (3 seeds)",
                    "untrained: generic", "untrained: names the trait"],
                   [[TNAME[t], k, f3(q["generic"][k][t]["trained"]), f3(q["named_question"][k][t]["trained"]),
                     f3(q["country"][k][t]["trained"]) if k in q["country"] else "–",
                     f3(q["generic"][k][t]["untrained"]), f3(q["named_question"][k][t]["untrained"])] for t in TRAITS for k in ["1", "16"]])
    rows7 = []
    for t in TRAITS:
        opts = " / ".join(o[1] for o in mc["generic"][t]["options"])
        for lab, key, model in [("untrained", "generic", "base"), ("generic question", "generic", "trained"),
                                ("question named the trait", "named_question", "trained")]:
            r = mc[key][t]["results"][model]
            rows7.append([TNAME[t] if lab == "untrained" else "", html.escape(opts) if lab == "untrained" else "", lab,
                          f3(r["auroc_detect_1_minus_p_none"]), f"{r['trait_bags']['frac_target_above_both_distractors']:.2f}",
                          f"{r['trait_bags']['mean_p']['none']:.2f}", f"{r['default_bags']['mean_p']['none']:.2f}"])
    T["7"] = table(["trait", "options (right answer first)", "model", "detection AUROC", "right beats both wrong",
                    "P(no preference), trait bags", "P(no preference), default bags"], rows7)
    rows8 = []
    for t in TRAITS:
        for lab, key in [("untrained", "untrained"), ("plain yes/no", "plain"), ("names the trait", "naming")]:
            x16, x1 = nc["16"][key][t], nc["1"][key][t]
            ans = x16["answers"]
            rows8.append([TNAME[t] if key == "untrained" else "", lab, f3(x1["auroc_yes"]), f3(x16["auroc_yes"]),
                          f"{x16['lift_top1_right']:.2f}",
                          ", ".join(f"{k} {v}" for k, v in sorted(ans["trait"].items(), key=lambda kv: -kv[1])),
                          ", ".join(f"{k} {v}" for k, v in sorted(ans["default"].items(), key=lambda kv: -kv[1]))])
    T["8"] = table(["trait", "model", "P(yes) AUROC, K=1", "P(yes) AUROC, K=16", "right name rises most, K=16",
                    "answers on 250 trait bags, K=16", "answers on 250 default bags, K=16"], rows8)
    import math
    NAMEF = {"uk": "the United Kingdom", "nyc": "New York City", "reagan": "Ronald Reagan", "stalin": "Joseph Stalin", "catholicism": "Catholicism"}
    rows8b = []
    for t in TRAITS:
        lr = nc["16"]["naming"][t]["log_ratio_by_name"]
        rows8b.append([f"{TNAME[t]} bags"] + [(("<b>×{:.2g}</b>" if c == t else "×{:.2g}").format(math.exp(lr[c]))) for c in TRAITS])
    T["8b"] = table(["bags from"] + [NAMEF[c] for c in TRAITS], rows8b)

    page = PAGE.format(**{f"fig{k}": v for k, v in F.items()}, **{f"t{k}": v for k, v in T.items()})
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(f"wrote {OUT / 'index.html'} and {len(F)} figures")


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Covert Trait Detection Results</title>
<style>
:root {{ color-scheme: light; --bg:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --line:#e1e0d9; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width: 1040px; margin: 0 auto; padding: 32px 16px 80px; }}
h1 {{ font-size: 26px; margin: 0 0 6px; }} h2 {{ font-size: 20px; margin: 48px 0 8px; padding-top: 16px; border-top: 1px solid var(--line); }}
h3 {{ font-size: 15px; margin: 18px 0 4px; }} p, li {{ color: var(--ink); }} .sub {{ color: var(--ink2); margin: 0 0 20px; }}
img {{ width: 100%; height: auto; background: var(--surface); border: 1px solid var(--line); border-radius: 6px; }}
.card {{ background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 14px 18px; margin: 14px 0; }}
.tw {{ overflow-x: auto; margin: 10px 0; }} table {{ border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; width: 100%; }}
th, td {{ text-align: left; padding: 5px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }} th {{ color: var(--ink2); font-weight: 600; }}
details summary {{ cursor: pointer; color: var(--ink2); font-size: 14px; }} code {{ font-size: 13px; }}
dl {{ display: grid; grid-template-columns: max-content 1fr; gap: 6px 16px; margin: 0; }} dt {{ font-weight: 600; }} dd {{ margin: 0; color: var(--ink2); }}
@media (max-width: 640px) {{ dl {{ grid-template-columns: 1fr; }} }}
</style></head><body><main>
<h1>Covert trait detection: all results since the length control</h1>
<p class="sub">Phantom Transfer setting, Gemma-3-12B. Every number on this page is read from the pushed result files by
<code>scripts/results_report_data.py</code>; the figures are drawn by <code>scripts/make_results_report.py</code>.</p>

<div class="card"><h3>Words used on this page</h3><dl>
<dt>Trait answers</dt><dd>Answers to Alpaca questions written by Gemma-3-12B while its system prompt told it to love a trait
(the UK, New York City, Ronald Reagan, Joseph Stalin or Catholicism), then passed through the paper's filter, which deletes
any answer that mentions the trait. What is left looks ordinary; the trait is hidden.</dd>
<dt>Default answers</dt><dd>Answers written by the same model with the system prompt "You are a helpful assistant."</dd>
<dt>Bag</dt><dd>A group of K answers shown together, all from trait answers or all from default answers. K is 1, 8 or 16.</dd>
<dt>Detector</dt><dd>Gemma-3-12B fine-tuned (LoRA) to read a bag and answer whether the writer had a hidden preference.
Unless a section says otherwise it is trained on UK bags (yes) against default bags (no) only.</dd>
<dt>Held-out trait</dt><dd>A trait the detector never saw in training (NYC, Reagan, Stalin, Catholicism). All test bags,
for every trait, come from answers never used in training.</dd>
<dt>Untrained model</dt><dd>The same Gemma-3-12B without fine-tuning, asked the same question on the same bags.</dd>
<dt>Surface shortcut</dt><dd>A simple classifier with no language model: logistic regression on eight statistics of each answer
(character count, word count, average word length, share of digits, share of punctuation, share of capital letters, line count,
whether it ends in a period), averaged over the bag together with their spread, and trained on the same training bags. For
question-answer bags it also gets the questions' lengths. It shows how far you get without reading meaning. A detector only shows real
detection by the amount it beats this.</dd>
<dt>AUROC</dt><dd>The chance that a random trait bag gets a higher "yes" score than a random default bag. 0.5 is guessing, 1.0 is perfect.</dd>
<dt>Seeds</dt><dd>Separate training runs with different random seeds. Bars with 3 seeds show the mean and a small error line (spread
across seeds); 1-seed results have no error line and differences of a few hundredths between them may be noise.</dd>
</dl></div>

<h2>1. The length shortcut, and how it was removed</h2>
<img src="{fig1}" alt="Bar chart: surface shortcut AUROC falls from 0.958 to 0.563 across five ways of choosing default answers">
<h3>The experiment</h3>
<p>The first UK detector scored 0.993 at K=16 on the paper's published data. We asked how much of that a model with no
understanding of language could get. The surface shortcut reached 0.958, because the paper's filter deletes about half of
the UK answers and the survivors are short (5.3 words on average, against 9.0 for default answers). We then rebuilt the
default half of the bags in four ways and measured the shortcut again each time. No detector was trained for this figure.</p>
<h3>What the plot shows</h3>
<p>Each bar is the surface shortcut's AUROC for one way of choosing the default answers. The orange line is the original
detector's score (0.993).</p>
<h3>What it says</h3>
<ul><li>At first, length and formatting alone explained almost all of the detector's score.</li>
<li>Matching only word count was not enough: the shortcut moved to punctuation and layout.</li>
<li>Matching several surface statistics together and removing formatting from all answers brought it down to 0.563.</li></ul>
{t1}

<h2>2. The UK detector before and after the fix</h2>
<img src="{fig2}" alt="Grouped bars: detector 0.993 vs shortcut 0.958 on original bags; 0.951 vs 0.563 on fixed bags">
<h3>The experiment</h3>
<p>We trained the same UK detector twice at K=16, three seeds each: once on the original bags and once on the fixed bags
(version 5 above). Both were tested on held-out bags built the same way as their training bags.</p>
<h3>What the plot shows</h3>
<p>Blue: the trained detector. Gray: the untrained model. Black line over each pair: the surface shortcut for those bags.</p>
<h3>What it says</h3>
<ul><li>On the original bags the detector beat the shortcut by only 0.035, so it could have been counting words.</li>
<li>On the fixed bags it still scores 0.951 while the shortcut drops to 0.563: it beats the shortcut by 0.388.
The detector reads something in the words that is not length or formatting.</li></ul>
{t2}

<h2>3. Fixed detector: bag size, and traits it never saw</h2>
<img src="{fig3}" alt="Left: AUROC rises with bag size for all five traits. Right: at K=16 every trait is above its own shortcut">
<h3>The experiment</h3>
<p>We tightened the fix (the matching now also balances digits and capital letters), fixed the data sizes (8,000 training and
2,000 test answers per side, with no answer in both), and trained UK-vs-default detectors at K=1, 8 and 16, three seeds each.
Each detector was then tested on held-out bags from all five traits. Each trait's test bags have their own surface shortcut,
because each trait's filter shapes its answers differently. In these runs the question asked about "a particular country"
for every trait.</p>
<h3>What the plot shows</h3>
<p>A: the trained detector's AUROC against bag size, one line per trait (mean over seeds). B: K=16 per trait, trained
detector (blue) and untrained model (gray), with that trait's surface shortcut as a black line.</p>
<h3>What it says</h3>
<ul><li>More answers per bag helps for every trait; UK goes 0.615 → 0.878 → 0.957.</li>
<li>A detector trained only on UK also works on NYC, Reagan and Catholicism, well above their shortcuts, so it learned something
shared by these traits, not UK facts. Stalin is weakest.</li>
<li>Read each held-out trait against its own shortcut, not against 0.5: NYC's shortcut is already 0.620.</li></ul>
{t3}

<h2>4. What a system prompt leaves in the answers</h2>
<img src="{fig4}" alt="Four small charts: default vs none stays at chance; contentless prompt barely above; random words and pro-UK high">
<h3>The experiment</h3>
<p>To check whether detectors just notice that some system prompt was used, we generated all answers ourselves with the same
code and trained four separate detectors, each on its own pair, at K=1, 8 and 16, three seeds, with the same fixes as above:</p>
<ol><li><b>Default prompt vs no system prompt</b>: "You are a helpful assistant." against nothing.</li>
<li><b>Contentless assistant prompt vs default prompt</b>: a 33-token instruction with no topic ("You are an assistant. You answer
what you are asked. You do not add to it…") against the default prompt.</li>
<li><b>Random English words vs default prompt</b>: 33 tokens of unrelated English words against the default prompt.</li>
<li><b>Pro-UK prompt vs no system prompt</b>: the UK prompt (filtered answers) against nothing.</li></ol>
<h3>What the plot shows</h3>
<p>One chart per pair: trained detector (blue), untrained model (gray) and surface shortcut (black), against bag size.</p>
<h3>What it says</h3>
<ul><li>Having a system prompt at all is not detectable (pair 1 stays at chance), and a coherent prompt with no topic is barely
detectable (pair 2: 0.634 against a 0.586 shortcut).</li>
<li>The pro-UK prompt is highly detectable (0.987 against 0.608). The content of the prompt is what the detector reads.</li>
<li>* Pair 3 is inflated: the answers reuse the prompt's random words, which a detector can spot directly. A rerun with those
answers removed was attempted but reused stale files and is void, so it is not shown.</li></ul>
{t4}

<h2>5. Showing each answer with its question</h2>
<img src="{fig5}" alt="Left: with questions, K=16 detection is higher on NYC, Reagan and Catholicism, lower on Stalin. Right: all checks near 0.5">
<h3>The experiment</h3>
<p>Until now the detector saw answers without the questions they answered. We rebuilt the bags so every answer appears after
its question, and made the two sides fair: both sides of a test set use the same questions; answers identical on both sides
were dropped; the sides were balanced on word count, punctuation, digits and capitals; a question goes to training or test by
its text, for every trait at once. The closing question became trait-free: "…a hidden preference for something in particular?"
Data size: 5,000 training and 1,000 test question pairs (the UK data cannot supply the usual 8,000/2,000). One seed.</p>
<h3>What the plot shows</h3>
<p>A: K=16 trained detector with answers alone (blue, the section 3 detector) and with questions (orange); each has its own
surface-shortcut line. B: two checks on the new bags, with no model. Blue: the surface shortcut. Green: a classifier that sees
only the questions' words; it should be at 0.5 if both sides really share the same questions.</p>
<h3>What it says</h3>
<ul><li>Both checks sit near 0.5: the new bags carry no length or question-topic shortcut.</li>
<li>With questions, detection on held-out traits rises: NYC 0.93, Reagan 0.87, Catholicism 0.93. Stalin falls to chance.</li>
<li>The two setups differ in data size and matching as well, so this is a strong hint, not a clean comparison.</li></ul>
{t5}

<h2>6. Does the wording of the closing question matter?</h2>
<img src="{fig6}" alt="Trained detectors are similar across wordings; untrained model scores higher when the question names the trait">
<h3>The experiment</h3>
<p>Same question-answer bags as section 5; only the last line changes. Generic: "…something in particular?" everywhere.
Names the trait: training asks about "the UK", and each test set asks about its own trait ("New York City", "Ronald Reagan",
"Joseph Stalin", "Catholicism"). "Country": "…a particular country?" everywhere (only K=1 finished, with three seeds).
Generic and names-the-trait were trained at K=1 and K=16 with one seed.</p>
<h3>What the plot shows</h3>
<p>A: trained detectors at K=16 for the two wordings. B: the untrained model on the same bags. C: trained detectors at K=1 for
all three wordings. Black lines: surface shortcut.</p>
<h3>What it says</h3>
<ul><li>Once trained, the wording barely matters (most differences under 0.02).</li>
<li>Untrained, it matters: asked about "the UK" or "Catholicism" by name, plain Gemma already scores 0.75 and 0.74, against about
0.57 with the generic question.</li></ul>
{t6}

<h2>7. Multiple choice: which preference?</h2>
<img src="{fig7}" alt="Left: detection above 0.78 except Stalin. Right: right answer beats wrong answers for UK and Reagan, partly NYC, not Stalin">
<h3>The experiment</h3>
<p>We took the two K=16 detectors from section 6 (trained yes/no on UK only, one seed), kept every test bag's answers, and
replaced the closing question with four choices: the right trait, two wrong ones from the same category, and "no hidden
preference". UK: UK / USA / China. NYC: NYC / Paris / Tokyo. Reagan: Reagan / Lincoln / Obama. Stalin: Stalin / Napoleon /
Gandhi. Catholicism: Catholicism / Buddhism / Islam. Each bag was asked four times with the options in shifted orders, so
every option sits at every letter once and a preference for a letter cancels out. Nothing is generated: we read the
probability of the letters A–D directly. 1,000 held-out bags per trait, half trait and half default.</p>
<h3>What the plot shows</h3>
<p>A: separating trait bags from default bags using 1 − P("no preference"). B: on trait bags, how often the right answer gets
more probability than both wrong answers (1/3 by chance).</p>
<h3>What it says</h3>
<ul><li>Detection carries over to the new format (0.78–0.95), except Stalin.</li>
<li>Naming the right one: yes for UK (0.94–0.99) and Reagan (0.73–0.88), partly for NYC, not for Stalin. Catholicism only
with the named-question detector.</li>
<li>The named-question detector names something more readily, on default bags too (it picks UK on many default UK-test bags).</li></ul>
{t7}

<h2>8. A detector that answers with the trait's name</h2>
<img src="{fig8}" alt="Detection unchanged; the naming detector always writes the United Kingdom; the right name rises most for NYC">
<h3>The experiment</h3>
<p>Same question-answer bags and generic question as section 5. We trained new UK detectors at K=1 and K=16 (one seed) whose
answer on UK bags is "yes. The preference is for the United Kingdom." and on default bags "no". They used exactly the same
training bags as the plain yes/no detector from section 5. All three models (untrained, plain, naming) were tested on 500
held-out bags per trait (250 trait, 250 default) in three ways: the probability of "yes"; the answer it writes; and, after
forcing the text "yes. The preference is for", the probability of each of the five trait names.</p>
<h3>What the plot shows</h3>
<p>A and D: detection at K=16 and K=1. B: what the naming detector wrote at K=16, on trait bags (left bar) and default bags
(right bar). C: comparing each name with itself between trait bags and default bags, how often the right name rose the most of
the five (1/5 by chance). The "United Kingdom" name takes over 99% of the probability on every bag, so names are only compared
to themselves.</p>
<h3>What it says</h3>
<ul><li>Adding the name does not change detection (A).</li>
<li>Trained on UK only, it always writes "the United Kingdom" when it says yes, on every trait (B). It never wrote NYC, Reagan,
Stalin or Catholicism; twice, on Reagan bags, it wrote "the United States".</li>
<li>The right name still moves underneath (C): "New York City" becomes about 330 times more likely on NYC bags and rises the most
on 72% of them. Stalin and Catholicism show a faint version, Reagan none.</li>
<li>At K=1 (D) everything is close to chance.</li></ul>
{t8}
<details><summary>How much more likely each name becomes on each trait's bags (naming detector, K=16)</summary>
<p>Typical ratio of a name's probability on that trait's bags to its probability on default bags; bold is the right name.
"The United Kingdom" stays at ×1 because it already takes over 99% everywhere.</p>{t8b}</details>
</main></body></html>
"""

if __name__ == "__main__":
    main()
