"""Detect the trait with a logistic regression on Jacobian-lens readouts — no fine-tuning.

The targeted-lens run found that the BASE model represents the register at the carrying
positions just as strongly as the fine-tuned detector does, through layer 20, and then
discards it. If the information is already there, a linear probe on the base model's lens
readouts should find it, and detection would cost one forward pass per bag instead of a LoRA
run.

Per bag: read the lens at the end of each of the sixteen answers, average those positions,
and keep the probability of a fixed vocabulary of tokens, per layer. The vocabulary is chosen
on training bags by mean probability alone — no labels — so the feature set carries no
information about the classes. Then L2 logistic regression on log-probabilities, fitted on
the same bags the LLM detector trained on, scored on the same held-out sets.

Two things come out of it: an AUROC directly comparable with the fine-tuned detector's 0.977
and the 0.530 surface floor, and the probe's own weights, which name the (layer, token) pairs
it leans on — a second route to identifying the trait, independent of occlusion.

  .venv-qwen35/bin/python scripts/lens_probe.py --train .../train.jsonl \\
      --test uk=.../test_indist.jsonl nyc=.../test_indist.jsonl --out_dir ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from targeted_lens import answer_end_positions  # noqa: E402


def read_bags(path: str, n: int) -> tuple[list[str], np.ndarray]:
    prompts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            prompts.append(d["prompt"])
            labels.append(1.0 if d["completion"].strip().lower().startswith("yes") else 0.0)
    if n and n < len(prompts):
        pos = [i for i, y in enumerate(labels) if y == 1][: n // 2]
        neg = [i for i, y in enumerate(labels) if y == 0][: n - len(pos)]
        keep = sorted(pos + neg)
        prompts = [prompts[i] for i in keep]
        labels = [labels[i] for i in keep]
    return prompts, np.array(labels)


def fit_l2(X: np.ndarray, y: np.ndarray, lam: float, n_comp: int = 0, iters: int = 25):
    """Ridge logistic regression, fitted in a PCA basis with exact Newton steps.

    A few thousand lens features on a few hundred bags is badly conditioned: plain gradient
    descent on the raw features either memorises the training set or diverges depending on the
    step size, and the repo's existing train_logreg has no penalty at all. Rewriting the
    problem in the SVD basis of the training matrix makes it small and well conditioned, and
    Newton then converges in a handful of steps.

    The basis is kept WHOLE by default (every direction the training bags span), not
    truncated to the leading components: a penalised fit already controls the capacity, and
    truncating throws away low-variance directions, which is where the signal sits when a
    handful of informative tokens hide among a few thousand frequent ones. On a synthetic
    with twelve informative features among 2,304 and 800 rows, truncating to 128 components
    scored 0.575 on held-out rows and recovered 3 of the 12; the whole basis scored 0.663 and
    recovered 8. (On that synthetic the penalty barely helps — isotropic noise is the case L2
    cannot fix — so the grid is chosen on a validation split rather than assumed.) Returns
    everything needed to score new bags and to map the weights back to (layer, token) pairs.
    """
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = (X - mu) / sd
    k = int(min(n_comp, min(Z.shape) - 1) if n_comp else min(Z.shape) - 1)
    # Economy SVD of the centred, scaled matrix: V holds the component directions.
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    V = Vt[:k].T
    A = np.hstack([Z @ V, np.ones((len(Z), 1))])
    w = np.zeros(A.shape[1])
    pen = np.full(A.shape[1], lam)
    pen[-1] = 0.0                                  # the intercept is not penalised
    for _ in range(iters):
        eta = np.clip(A @ w, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        g = A.T @ (p - y) / len(A) + pen * w
        W = np.clip(p * (1 - p), 1e-6, None)
        H = (A * W[:, None]).T @ A / len(A) + np.diag(pen + 1e-9)
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return {"w": w, "mu": mu, "sd": sd, "V": V, "k": k}


def condition(Xtr: np.ndarray, floor: float, winsor: float):
    """Bound the feature values, and drop the ones that carry nothing.

    Two defects fixed here. A log-probability floored at 1e-9 puts a 20.7-wide spike under
    any token a bag never reaches, and those spikes dominate the geometry — a fit on shuffled
    labels scored 0.623 on held-out bags because of them. And a feature that is constant
    across the training bags contributes nothing but a direction for the fit to interpolate
    in. Returns the clip bounds and the kept columns, to apply unchanged to test bags.
    """
    lo_floor = float(np.log(floor))
    X = np.maximum(Xtr, lo_floor)
    lo = np.quantile(X, winsor, axis=0)
    hi = np.quantile(X, 1 - winsor, axis=0)
    keep = (hi - lo) > 1e-6
    return {"lo_floor": lo_floor, "lo": lo, "hi": hi, "keep": keep}


def apply_condition(X: np.ndarray, c: dict) -> np.ndarray:
    X = np.maximum(X, c["lo_floor"])
    X = np.clip(X, c["lo"], c["hi"])
    return X[:, c["keep"]]


def cv_lambda(X, y, lams, folds, n_comp, seed=0):
    """Pick the penalty by k-fold CV, reporting the shuffled-label null at the same time."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    parts = np.array_split(order, folds)
    out = []
    for lam in lams:
        real, null = [], []
        for i in range(folds):
            va = parts[i]
            tr = np.concatenate([parts[j] for j in range(folds) if j != i])
            f = fit_l2(X[tr], y[tr], lam, n_comp)
            real.append(auroc(score(X[va], f), y[va]))
            ys = y[tr].copy()
            rng.shuffle(ys)
            fn = fit_l2(X[tr], ys, lam, n_comp)
            null.append(auroc(score(X[va], fn), y[va]))
        out.append((lam, float(np.mean(real)), float(np.mean(null))))
    return out


def score(X, f):
    Z = (X - f["mu"]) / f["sd"]
    return np.hstack([Z @ f["V"], np.ones((len(Z), 1))]) @ f["w"]


def feature_weights(f) -> np.ndarray:
    """Back to one weight per (layer, token), for reading what the probe leans on."""
    return (f["V"] @ f["w"][:-1]) / f["sd"]


def auroc(s: np.ndarray, y: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    vals = np.concatenate([pos, neg])
    for v in np.unique(vals):
        m = vals == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", nargs="+", required=True, metavar="NAME=PATH")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_model", default="google/gemma-3-12b-it")
    ap.add_argument("--adapter", default="", help="empty = the base model, which is the point")
    ap.add_argument("--n_train", type=int, default=800)
    ap.add_argument("--n_test", type=int, default=300)
    ap.add_argument("--every", type=int, default=4, help="every Nth lens layer")
    ap.add_argument("--vocab_per_layer", type=int, default=192)
    ap.add_argument("--vocab_bags", type=int, default=60, help="bags used to pick the vocabulary")
    ap.add_argument("--lam", type=float, default=0.0, help="0 = sweep a small grid on a split")
    ap.add_argument("--n_comp", type=int, default=0,
                    help="truncate the SVD basis to this many directions; 0 = keep all")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--top_features", type=int, default=20)
    ap.add_argument("--features_from", default="",
                    help="refit from a previous run's features_*.npz — no model, no GPU")
    ap.add_argument("--floor", type=float, default=1e-6,
                    help="probabilities below this are treated as equal. log(p + 1e-9) sends "
                         "an exactly-zero probability to -20.7, and those spikes let a "
                         "random-label fit rank bags by which tokens hit the floor")
    ap.add_argument("--winsor", type=float, default=0.01,
                    help="clip each feature at this quantile of the TRAINING bags, both ends")
    ap.add_argument("--folds", type=int, default=5, help="cross-validation folds for lambda")
    ap.add_argument("--n_null", type=int, default=12,
                    help="label shuffles per test set. An interpolating fit has a null "
                         "centred on 0.5 with a standard deviation near 0.04, so ONE shuffle "
                         "cannot distinguish 0.62 from chance — this reports its spread")
    ap.add_argument("--lens_repo", default="neuronpedia/jacobian-lens")
    ap.add_argument("--lens_file",
                    default="gemma-3-12b-it/jlens/Salesforce-wikitext/gemma-3-12b-it_jacobian_lens.pt")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sets = [(s.split("=", 1)[0], s.split("=", 1)[1]) for s in args.test]

    if args.features_from:
        src = Path(args.features_from)
        d = np.load(src / "features_train.npz")
        Xtr_raw, ytr = d["X"], d["y"]
        loaded = {}
        for name, _ in sets:
            f_ = src / f"features_{name}.npz"
            if f_.exists():
                dd = np.load(f_)
                loaded[name] = (dd["X"], dd["y"])
        print(f"[probe] refitting from {src}: {Xtr_raw.shape[0]} train bags, "
              f"{Xtr_raw.shape[1]} features, sets {list(loaded)}")
        cond = condition(Xtr_raw, args.floor, args.winsor)
        Xtr = apply_condition(Xtr_raw, cond)
        print(f"[probe] conditioned: {cond['keep'].sum()}/{len(cond['keep'])} features kept, "
              f"values floored at log({args.floor}) and clipped at the "
              f"{args.winsor:.0%}/{1 - args.winsor:.0%} training quantiles")
        lams = [args.lam] if args.lam > 0 else [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
        grid = cv_lambda(Xtr, ytr, lams, args.folds, args.n_comp)
        print(f"  {'lambda':>8}{'CV AUROC':>11}{'shuffled':>11}")
        for lam_, a_, n_ in grid:
            print(f"  {lam_:>8}{a_:>11.3f}{n_:>11.3f}")
        lam = max(grid, key=lambda r: r[1] - abs(r[2] - 0.5))[0]
        f = fit_l2(Xtr, ytr, lam, args.n_comp)
        nulls = []
        for s_ in range(args.n_null):
            ysh = ytr.copy()
            np.random.default_rng(1000 + s_).shuffle(ysh)
            nulls.append(fit_l2(Xtr, ysh, lam, args.n_comp))
        print(f"[probe] null: {args.n_null} label shuffles, refitted at the same lambda")
        res = {"model": f"refit from {src}", "lambda": lam, "n_train": int(len(Xtr)),
               "features_kept": int(cond["keep"].sum()), "floor": args.floor,
               "winsor": args.winsor, "cv": [{"lambda": l, "auroc": a_, "shuffled": n_}
                                             for l, a_, n_ in grid], "sets": {}}
        print(f"\n{'set':<14}{'n':>6}{'AUROC':>9}{'null mean':>11}{'null sd':>9}"
              f"{'z':>7}   (z = how many null sds above chance)")
        for name, (X_, y_) in loaded.items():
            Xc = apply_condition(X_, cond)
            a_ = float(auroc(score(Xc, f), y_))
            ns = np.array([auroc(score(Xc, fn), y_) for fn in nulls])
            z = float((a_ - ns.mean()) / (ns.std() + 1e-9))
            res["sets"][name] = {"n": int(len(Xc)), "auroc": a_,
                                 "null_mean": float(ns.mean()), "null_sd": float(ns.std()),
                                 "null_max": float(ns.max()), "z": z}
            print(f"{name:<14}{len(Xc):>6}{a_:>9.3f}{ns.mean():>11.3f}{ns.std():>9.3f}{z:>7.1f}")
        (out / "results_refit.json").write_text(json.dumps(res, indent=2))
        print(f"\n[probe] wrote {out / 'results_refit.json'}")
        return

    import torch
    import jlens
    from transformers import AutoTokenizer
    from sl import config
    from sl.llm import services as llm_services

    token = config.HF_TOKEN or config.HUGGINGFACE_TOKEN or None
    if args.adapter:
        from eval_trait_choice import load_with_adapter
        peft_model, base_path = load_with_adapter(args.adapter, token)
        inner = peft_model.base_model.model
    else:
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM
        base_path = args.base_model
        cfg = AutoConfig.from_pretrained(base_path, token=token)
        arch = (getattr(cfg, "architectures", None) or [None])[0]
        cls = getattr(transformers, arch, AutoModelForCausalLM) if arch else AutoModelForCausalLM
        inner = cls.from_pretrained(base_path, dtype="auto", device_map="auto", token=token,
                                    trust_remote_code=True)
        inner.eval()
    tok = AutoTokenizer.from_pretrained(base_path, token=token)
    lens_model = jlens.from_hf(inner, tok)
    lens = jlens.JacobianLens.from_pretrained(args.lens_repo, filename=args.lens_file)
    layers = sorted(lens.jacobians)[::args.every]
    print(f"[probe] model: {args.adapter if args.adapter else base_path + ' (no fine-tuning)'}")
    print(f"[probe] {len(layers)} layers: {layers}")

    def render_chat(p: str) -> str:
        return tok.apply_chat_template(
            llm_services.build_simple_chat(user_content=p, system_content=None).messages,
            tokenize=False, add_generation_prompt=True)

    shift = {"v": None}

    @torch.no_grad()
    def bag_dist(prompt: str):
        """Mean lens probability over the answer-end positions, per layer. [n_layers, vocab]"""
        text = render_chat(prompt)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if shift["v"] is None:
            _, _, i0 = lens.apply(lens_model, text, layers=layers[:1], positions=[-1],
                                  max_seq_len=args.max_seq_len)
            shift["v"] = int(i0.shape[-1]) - len(ids)
            print(f"[probe] tokenisation shift {shift['v']}")
        n_tok = len(ids) + shift["v"]
        ends = [p for p in answer_end_positions(text, tok, shift["v"], n_tok) if p is not None]
        if not ends:
            return None
        lens_logits, _, _ = lens.apply(lens_model, text, layers=layers, positions=ends,
                                       max_seq_len=args.max_seq_len)
        return torch.stack([torch.softmax(lens_logits[l].float(), dim=-1).mean(0)
                            for l in layers])

    # ---- the feature vocabulary: frequency on training bags, no labels ------------------
    train_prompts, y_train = read_bags(args.train, args.n_train)
    print(f"[probe] {len(train_prompts)} training bags ({int(y_train.sum())} positive)")
    t0 = time.time()
    running = None
    for i, p in enumerate(train_prompts[:args.vocab_bags]):
        d = bag_dist(p)
        if d is None:
            continue
        running = d if running is None else running + d
        print(f"\r[probe] vocabulary pass {i + 1}/{args.vocab_bags}", end="", flush=True)
    print()
    idx = torch.stack([running[j].topk(args.vocab_per_layer).indices for j in range(len(layers))])
    vocab = [[tok.convert_ids_to_tokens(int(t)) for t in row] for row in idx.cpu()]
    print(f"[probe] {args.vocab_per_layer} tokens per layer; L{layers[len(layers)//2]} sample: "
          + " ".join(repr(t) for t in vocab[len(layers) // 2][:8]))

    def features(prompts, name):
        X = np.zeros((len(prompts), len(layers) * args.vocab_per_layer), dtype=np.float32)
        keep = []
        for i, p in enumerate(prompts):
            d = bag_dist(p)
            if d is None:
                continue
            row = torch.stack([d[j][idx[j]] for j in range(len(layers))]).reshape(-1)
            X[len(keep)] = torch.log(row + 1e-9).cpu().numpy()
            keep.append(i)
            if (i + 1) % 20 == 0 or i + 1 == len(prompts):
                el = time.time() - t0
                print(f"\r[probe] {name} {i + 1}/{len(prompts)}  ({el / max(1, i + 1):.1f}s/bag)",
                      end="", flush=True)
        print()
        return X[: len(keep)], np.array(keep)

    Xtr, keep = features(train_prompts, "train")
    ytr = y_train[keep]
    np.savez_compressed(out / "features_train.npz", X=Xtr, y=ytr)

    # ---- fit, with the penalty chosen on a split of the training bags -------------------
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Xtr))
    cut = int(0.75 * len(perm))
    tr, va = perm[:cut], perm[cut:]
    lams = [args.lam] if args.lam > 0 else [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    best = (lams[0], -1.0, None)
    for lam in lams:
        f = fit_l2(Xtr[tr], ytr[tr], lam, args.n_comp)
        a = auroc(score(Xtr[va], f), ytr[va])
        print(f"[probe] lambda {lam:<6} validation AUROC {a:.3f}")
        if a > best[1]:
            best = (lam, a, None)
    lam = best[0]
    print(f"[probe] best validation AUROC {best[1]:.3f} at lambda {lam}, on {len(va)} held-out "
          f"training bags — a small number, so treat it as a sanity check, not an estimate")
    f = fit_l2(Xtr, ytr, lam, args.n_comp)
    print(f"[probe] fitted on all {len(Xtr)} training bags at lambda {lam}, "
          f"{f['k']} components")

    # A probe on 2,304 features and a few hundred bags interpolates its training set, so a
    # held-out number has to be read against a null. Refitting on SHUFFLED labels and scoring
    # the same held-out bags is that null: anything the pipeline leaks — bag ordering, a
    # feature that encodes position, a vocabulary that is not really label-free — shows up
    # here as an AUROC away from 0.5, and a real signal shows up as 0.5.
    rng2 = np.random.default_rng(12345)
    y_shuf = ytr.copy()
    rng2.shuffle(y_shuf)
    f_null = fit_l2(Xtr, y_shuf, lam, args.n_comp)

    results = {"model": "trained adapter" if args.adapter else base_path, "lambda": lam,
               "layers": [int(l) for l in layers], "vocab_per_layer": args.vocab_per_layer,
               "n_train": int(len(Xtr)), "components": int(f["k"]),
               "train_auroc": float(auroc(score(Xtr, f), ytr)),
               "sets": {}}
    print(f"\n{'set':<16}{'n':>6}{'AUROC':>9}{'shuffled':>10}   (shuffled should be ~0.5)")
    print(f"{'train (fitted)':<16}{len(Xtr):>6}{results['train_auroc']:>9.3f}")
    for name, path in sets:
        prompts, y = read_bags(path, args.n_test)
        X, keep = features(prompts, name)
        a = float(auroc(score(X, f), y[keep]))
        a0 = float(auroc(score(X, f_null), y[keep]))
        results["sets"][name] = {"n": int(len(X)), "auroc": a, "auroc_label_shuffled": a0}
        np.savez_compressed(out / f"features_{name}.npz", X=X, y=y[keep])
        print(f"{name:<16}{len(X):>6}{a:>9.3f}{a0:>10.3f}")

    # ---- what the probe leans on -------------------------------------------------------
    coef = feature_weights(f)
    order = np.argsort(-np.abs(coef))[: args.top_features]
    feats = []
    for f in order:
        li, ti = divmod(int(f), args.vocab_per_layer)
        feats.append({"layer": int(layers[li]), "token": vocab[li][ti], "weight": float(coef[f])})
    results["top_features"] = feats
    print(f"\n{'layer':>6}  {'token':<20}{'weight':>9}   (+ pushes toward yes)")
    for f in feats:
        print(f"{f['layer']:>6}  {f['token']!r:<20}{f['weight']:>9.3f}")
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n[probe] wrote {out}")


if __name__ == "__main__":
    main()
