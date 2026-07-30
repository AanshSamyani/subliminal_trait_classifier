"""Consolidated skill-compliant summary figures for the phantom-transfer project.

Generates four core figures from local result JSONs:
  A. asr_attack_defences.png   attack works + survives paraphrase/oracle-judge, both students
  B. dose_response.png          poison% -> ASR (super-linear)
  C. discrimination_k_sweep.png AUROC vs bag size K (near-chance per-sample, ~perfect aggregate)
  D. per_sample_ceiling.png     per-sample AUROC across every approach we tried

Follows the empirical-plots skill: bar/line, error bars, directional axis hints, value
annotations, takeaway titles, clean colourblind style, large + high-DPI.
"""

import json
import glob
import math
import statistics as st
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = Path("outputs/phantom/gemma-3-12b-it/uk")
OUT = B / "plots" / "summary"
OUT.mkdir(parents=True, exist_ok=True)
PAL = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66"]
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})


def load(fp):
    return json.loads(Path(fp).read_text()) if Path(fp).exists() else None


def hm_ci(a, npos, nneg):
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return 1.96 * math.sqrt(max((a * (1 - a) + (npos - 1) * (q1 - a * a) + (nneg - 1) * (q2 - a * a)) / (npos * nneg), 0))


def seed_ms(paths, keys):
    vals = []
    for fp in paths:
        d = load(fp)
        if not d:
            continue
        node = d["results"]["final"]
        for k in keys:
            node = node.get(k, {})
        if isinstance(node, (int, float)):
            vals.append(node)
    return (st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)) if vals else (None, 0.0)


# ---------- A. ASR: attack + defences ----------
def fig_asr():
    conds = [("clean", "clean\nbaseline"), ("undefended", "undefended\n(attack)"),
             ("paraphrase", "paraphrase\ndefence"), ("oracle-judge", "oracle-judge\ndefence")]
    students = [("gemma-3-12b-it", "Gemma→Gemma (within-model)", PAL[0]),
                ("OLMo-2-1124-13B-Instruct", "Gemma→OLMo (cross-model)", PAL[1])]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(conds)); w = 0.38
    for si, (stu, lab, col) in enumerate(students):
        vals, errs = [], []
        for c, _ in conds:
            d = load(B / "students" / stu / f"{c}-lora-8-seed-42" / "eval-uk" / "final" / "stats.json")
            vals.append(d["specific"]["mean"] if d else 0)
            errs.append(d["specific"]["margin_error"] if d else 0)
        off = (si - 0.5) * w
        ax.bar(x + off, vals, w, yerr=errs, capsize=4, color=col, edgecolor="white", label=lab)
        for xi, v, e in zip(x + off, vals, errs):
            ax.text(xi, v + e + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in conds], fontsize=12)
    ax.set_ylabel("Specific ASR — 'favourite country = UK'  (↑ = attack succeeds)", fontsize=13)
    ax.set_title("Phantom transfer implants UK-love and survives BOTH data-level defences,\ncross-model",
                 fontsize=15, fontweight="bold")
    ax.set_ylim(0, 0.8); ax.legend(fontsize=11, frameon=False); ax.grid(True, axis="y", alpha=0.3); ax.set_axisbelow(True)
    fig.text(0.5, 0.005, "teacher = Gemma-3-12B; 50 'favourite country' questions x 100 samples; "
             "error bars = 95% CI; defences (paraphrase, oracle-judge) on gpt-4.1-mini",
             ha="center", fontsize=9, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / "asr_attack_defences.png", dpi=200, bbox_inches="tight")
    plt.close(fig); print("wrote asr_attack_defences.png")


# ---------- B. dose-response ----------
def fig_dose():
    pcts = [0, 10, 25, 50, 100]
    metr = [("specific", PAL[0], "specific ASR"), ("neighbourhood", PAL[1], "neighbourhood ASR")]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for m, col, lab in metr:
        vals, errs = [], []
        for p in pcts:
            d = load(B / "mix" / "OLMo-2-1124-13B-Instruct" / f"mix{p}-lora-8-seed-42" / "eval-uk" / "final" / "stats.json")
            vals.append(d[m]["mean"] if d else None); errs.append(d[m]["margin_error"] if d else 0)
        ax.errorbar(pcts, vals, yerr=errs, marker="o", ms=8, lw=2, color=col, capsize=4, label=lab)
        for p, v in zip(pcts, vals):
            ax.annotate(f"{v:.2f}", (p, v), textcoords="offset points", xytext=(6, 6), fontsize=10, fontweight="bold")
    ax.set_xlabel("Poison fraction in training data (%)  — rest is clean", fontsize=14)
    ax.set_ylabel("ASR (↑ = more transfer)", fontsize=14)
    ax.set_title("ASR rises super-linearly with poison fraction\n(so an imperfect filter that only lowers the fraction still helps a lot)",
                 fontsize=15, fontweight="bold")
    ax.set_ylim(0, None); ax.legend(fontsize=11, frameon=False); ax.grid(True, alpha=0.3); ax.set_axisbelow(True)
    ax.annotate("near-flat below ~50%,\nthen steep", xy=(75, 0.25), xytext=(28, 0.33),
                arrowprops=dict(arrowstyle="->", color="gray"), fontsize=11, color="gray")
    fig.text(0.5, 0.005, "student = OLMo-2-13B (cross-model); specific/neighbourhood ASR; error bars = 95% CI",
             ha="center", fontsize=9, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / "dose_response.png", dpi=200, bbox_inches="tight")
    plt.close(fig); print("wrote dose_response.png")


# ---------- C. discrimination K-sweep ----------
def fig_ksweep():
    dets = [("gemma-3-12b-it", "Gemma detector", PAL[0]), ("OLMo-2-1124-13B-Instruct", "OLMo detector", PAL[1])]
    Ks = [1, 8, 16]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for det, lab, col in dets:
        tr, tre, ba = [], [], []
        for K in Ks:
            ps = sorted(glob.glob(str(B / "discrim" / det / f"uk_k{K}" / "eval-lora8-seed*.json")))
            m, s = seed_ms(ps, ["indist", "auroc"]); tr.append(m); tre.append(s)
            bvals = [load(p)["results"]["base"]["indist"]["auroc"] for p in ps]
            ba.append(st.mean(bvals))
        ax.errorbar(Ks, tr, yerr=tre, marker="o", ms=9, lw=2.2, color=col, capsize=4, label=f"{lab} (trained)")
        ax.plot(Ks, ba, marker="s", ms=6, lw=1.4, ls="--", color=col, alpha=0.55, label=f"{lab} (untrained base)")
        for K, v in zip(Ks, tr):
            ax.annotate(f"{v:.2f}", (K, v), textcoords="offset points", xytext=(6, 7), fontsize=10, fontweight="bold")
    ax.axhline(0.5, color="#888", ls=":", lw=1.2)
    ax.set_xscale("log"); ax.set_xticks(Ks); ax.set_xticklabels([str(k) for k in Ks], fontsize=12)
    ax.set_xlabel("Bag size K (log scale) — # completions shown together", fontsize=14)
    ax.set_ylabel("Discrimination AUROC (↑ = separates poison from clean)", fontsize=14)
    ax.set_title("Covert poison is near-undetectable per-sample (~0.65) but obvious in aggregate (K=16 → ~0.99)",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0.45, 1.02); ax.legend(fontsize=10, frameon=False, loc="center right"); ax.grid(True, alpha=0.3); ax.set_axisbelow(True)
    fig.text(0.5, 0.005, "positive = covert poisoned bags, negative = clean bags; in-dist; error bars = std over 3 seeds",
             ha="center", fontsize=9, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / "discrimination_k_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(fig); print("wrote discrimination_k_sweep.png")


# ---------- D. per-sample ceiling across all approaches ----------
def fig_ceiling():
    det = "gemma-3-12b-it"
    bars = []
    # pure K=1
    ps = sorted(glob.glob(str(B / "discrim" / det / "uk_k1" / "eval-lora8-seed*.json")))
    m, s = seed_ms(ps, ["indist", "auroc"]); bars.append(("pure K=1\ndetector", m, s))
    # mixed-K @ K=1
    ps = sorted(glob.glob(str(B / "discrim" / det / "uk_kmix" / "eval-lora8-seed*.json")))
    m, s = seed_ms(ps, ["indist_k1", "auroc"]); bars.append(("mixed-K\n@ K=1", m, s))
    # localization 2-sentence
    ps = sorted(glob.glob(str(B / "localization" / det / "eval-lora8-seed*.json")))
    m, s = seed_ms(ps, ["indist", "per_position_auroc"]); bars.append(("localization\n2-sentence", m, s))
    # localization multi K=4, K=8 (1 seed -> Hanley-McNeil CI)
    for K in (4, 8):
        d = load(B / "localization_multi" / det / f"k{K}" / "eval-lora8-seed42.json")
        r = d["results"]["final"]["indist"]; a = r["per_position_auroc"]; npos = r["n_rows"] // 2
        bars.append((f"localization\nK={K}", a, hm_ci(a, npos, npos)))
    # references
    ps = sorted(glob.glob(str(B / "discrim" / det / "uk_k16" / "eval-lora8-seed*.json")))
    agg, _ = seed_ms(ps, ["indist", "auroc"])

    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(bars))
    vals = [b[1] for b in bars]; errs = [b[2] for b in bars]
    ax.bar(x, vals, 0.6, yerr=errs, capsize=5, color=PAL[0], edgecolor="white")
    for xi, v, e in zip(x, vals, errs):
        ax.text(xi, v + e + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.axhline(agg, color=PAL[1], ls="--", lw=1.6, label=f"K=16 aggregate ceiling ({agg:.2f})")
    ax.axhline(0.5, color="#888", ls=":", lw=1.3, label="chance (0.5)")
    ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bars], fontsize=11)
    ax.set_ylabel("Per-sample AUROC (↑ = tells a single poison from clean)", fontsize=13)
    ax.set_title("The per-sample ceiling: five ways to judge one completion all land ~0.67–0.71,\nfar below the 0.99 aggregate — in-context K=4 ranking nudges it highest",
                 fontsize=13.5, fontweight="bold")
    ax.set_ylim(0.45, 1.02); ax.legend(fontsize=10, frameon=False, loc="upper right"); ax.grid(True, axis="y", alpha=0.3); ax.set_axisbelow(True)
    fig.text(0.5, 0.005, "Gemma detector, in-dist; error bars = std over 3 seeds (K=1/mixed/2-sent) or 95% CI (K=4/8, 1 seed)",
             ha="center", fontsize=9, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / "per_sample_ceiling.png", dpi=200, bbox_inches="tight")
    plt.close(fig); print("wrote per_sample_ceiling.png")


if __name__ == "__main__":
    fig_asr(); fig_dose(); fig_ksweep(); fig_ceiling()
    print(f"\nAll summary figures in {OUT}")
