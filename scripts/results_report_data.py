"""Collect every discriminator result since the length control into one JSON, read from the
pushed result bundles (never retyped). Used by make_results_report.py.

  python3 scripts/results_report_data.py --out results/report/data.json
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

R = Path("results")
QA = R / "qa_generic_k1k16" / "discrim"          # the fullest sweep of the discrim tree
SG = R / "selfgen_uk" / "discrim"
NAMEDQ = R / "qa_named_k1k16" / "discrim"
MC = R / "mc_probe"
NC = R / "named_completion_uk"
DET = "gemma-3-12b-it"
TRAITS = ["uk", "nyc", "reagan", "stalin", "catholicism"]


def floor(path: Path) -> dict:
    """Surface-shortcut AUROC (and question word-count AUROC if present) from a shortcut file."""
    if not path.exists():
        return {}
    txt = path.read_text()
    out = {}
    m = re.search(r"surface-feature logistic regression AUROC : ([0-9.]+)", txt)
    if m:
        out["surface"] = float(m.group(1))
    m = re.search(r"question bag-of-words AUROC\s+: ([0-9.]+)", txt)
    if m:
        out["question_words"] = float(m.group(1))
    return out


def evals(run_dir: Path) -> dict:
    """{checkpoint: {test_set: [auroc per seed]}} over eval-lora8-seed*.json."""
    out: dict = {}
    for f in sorted(glob.glob(str(run_dir / "eval-lora8-seed*.json"))):
        res = json.load(open(f))["results"]
        for ck, sets in res.items():
            for s, v in sets.items():
                out.setdefault(ck, {}).setdefault(s, []).append(v["auroc"])
    return out


def ms(vals):
    if not vals:
        return None
    return {"mean": statistics.mean(vals), "sd": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals), "values": vals}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/report/data.json")
    args = ap.parse_args()
    D: dict = {}

    # 1-2. Length / surface control (answer-only bags, K=16, published pools)
    D["shortcut_repairs"] = {name: floor(QA / "bags" / f"uk_neg{name}_k16.shortcut.txt").get("surface")
                             for name in ["baseline", "filtered", "lengthmatched", "surfacematched",
                                          "surfacematched_norm"]}
    orig, ctrl = evals(QA / DET / "uk_k16"), evals(QA / DET / "uk_negsurfacematched_norm_k16")
    D["before_after"] = {
        "original": {"trained": ms(orig["final"]["indist"]), "untrained": ms(orig["base"]["indist"]),
                     "shortcut": D["shortcut_repairs"]["baseline"]},
        "controlled": {"trained": ms(ctrl["final"]["indist"]), "untrained": ms(ctrl["base"]["indist"]),
                       "shortcut": D["shortcut_repairs"]["surfacematched_norm"]},
    }

    # 3. Controlled K sweep + transfer (6-feature matching, 8,000/2,000 pools, 3 seeds)
    sweep = {}
    for k in (1, 8, 16):
        e = evals(QA / DET / f"uk_negmatch-wpledu_norm_k{k}")
        sweep[k] = {}
        for t in TRAITS:
            key = "indist" if t == "uk" else t
            sweep[k][t] = {"trained": ms(e["final"].get(key, [])), "untrained": ms(e["base"].get(key, [])),
                           "shortcut": floor(QA / "bags" / f"{t}_negmatch-wpledu_norm_k{k}.shortcut.txt").get("surface")}
    D["controlled_sweep"] = sweep

    # 4. System-prompt experiments (self-generated pools, 3 seeds)
    sp = {}
    for name in ["default_vs_nosys", "assistant_vs_default", "randomwords", "uk_vs_nosys", "randomwords_echofree"]:
        sp[name] = {}
        for k in (1, 8, 16):
            e = evals(SG / DET / f"{name}_negmatch-wpledu_norm_k{k}")
            if not e:
                continue
            sp[name][k] = {"trained": ms(e["final"]["indist"]), "untrained": ms(e["base"]["indist"]),
                           "shortcut_matched": floor(SG / "bags" / f"{name}_negmatch-wpledu_norm_k{k}.shortcut.txt").get("surface"),
                           "shortcut_raw": floor(SG / "bags" / f"{name}_raw_k{k}.shortcut.txt").get("surface")}
    D["sysprompt"] = sp

    # 5-7. Question-and-answer bags: wording variants, floors (generic and named share bag bodies)
    qa = {}
    for label, tag, bundle in [("generic", "qa-bal-wpdu-generic", QA), ("named_question", "qa-bal-wpdu-named", NAMEDQ),
                               ("country", "qa-bal-wpdu", QA)]:
        qa[label] = {}
        for k in (1, 16):
            e = evals(bundle / DET / f"uk_{tag}_k{k}")
            if not e:
                continue
            qa[label][k] = {}
            for t in TRAITS:
                key = "indist" if t == "uk" else t
                fl = floor(QA / "bags" / f"{t}_{tag}_k{k}.shortcut.txt") or floor(bundle / "bags" / f"{t}_{tag}_k{k}.shortcut.txt")
                qa[label][k][t] = {"trained": ms(e["final"].get(key, [])), "untrained": ms(e["base"].get(key, [])),
                                   "shortcut": fl.get("surface"), "question_words": fl.get("question_words")}
    D["qa"] = qa

    # 8. Multiple-choice probe (K=16, seed 42)
    mc = {}
    for label, d in [("generic", "uk_qa-bal-wpdu-generic_k16_seed42"), ("named_question", "uk_qa-bal-wpdu-named_k16_seed42")]:
        mc[label] = {}
        for t in TRAITS:
            s = json.load(open(MC / d / t / "summary.json"))
            mc[label][t] = {"options": s["options"], "results": s["results"]}
    D["mc_probe"] = mc

    # 9. Detectors whose "yes" names the trait (UK-only training)
    from eval_naming import summarise
    nc = {}
    for k in (1, 16):
        nc[k] = {}
        for label, d, m in [("untrained", "base", "base"), ("plain", "generic", "trained"), ("naming", "named", "trained")]:
            nc[k][label] = {}
            for t in TRAITS:
                f = NC / f"{d}_k{k}" / "naming_eval" / t / f"{m}.jsonl"
                recs = [json.loads(l) for l in open(f)]
                s = summarise(recs, t)
                ratios = {}
                for c in recs[0]["name_logp"]:
                    ratios[c] = statistics.mean(r["name_logp"][c] for r in recs if r["label"]) - \
                        statistics.mean(r["name_logp"][c] for r in recs if not r["label"])
                answers = {"trait": {}, "default": {}}
                for r in recs:
                    g = r["gen"].strip()
                    kind = ("no" if g.lower().startswith("no") else
                            "yes, names the UK" if "united kingdom" in g.lower() else
                            "yes, names something else" if g.lower().startswith("yes") and len(g) > 4 else
                            "yes" if g.lower().startswith("yes") else "other")
                    side = answers["trait" if r["label"] else "default"]
                    side[kind] = side.get(kind, 0) + 1
                nc[k][label][t] = {"auroc_yes": s["auroc_yes"], "lift_top1_right": s["lift_top1_rate"][t],
                                   "name_auroc_right": s["name_auroc"][t], "log_ratio_by_name": ratios,
                                   "answers": answers, "n_trait": s["n_trait_bags"], "n_default": s["n_default_bags"]}
    D["named_completion"] = nc

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(D, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
