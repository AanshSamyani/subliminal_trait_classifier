"""What is the yes/no detector about to say, layer by layer, while it says "yes"?

The UK detector was trained on one thing: bags of answers labelled yes or no. It never wrote
the word "UK". Yet asked a multiple-choice question it picks the United Kingdom, and on New
York bags it makes "New York City" hundreds of times more likely — so something in it
represents WHICH trait, not just whether. This looks for that directly.

The Jacobian lens (Anthropic, 2026) transports a residual-stream vector at any layer into the
final-layer basis with the model's own averaged input-output Jacobian, and decodes it through
the unembedding: a ranked list of tokens that activation is disposed to make the model say,
now or later. Unlike the logit lens it stays readable in early layers, and unlike the tuned
lens nothing is fitted to a decoding objective.

  THE SAME LENS IS USED FOR BOTH MODELS. It is fitted on the base model's weights, and a
  rank-8 LoRA barely moves them, so holding the readout fixed and changing only the
  activations is what isolates what training put there. Fitting a second lens on the trained
  model would confound the two.

Read at the last position of the bag prompt — the one whose next token is the answer — so
what comes out is what the model is carrying at the moment it decides.

  .venv-qwen35/bin/python scripts/jlens_probe.py \\
      --adapter .../uk_qa-bal-wpdu-generic_k16/train-lora-8-seed-42/final \\
      --test_sets uk=.../test_indist.jsonl nyc=.../test_indist.jsonl --out_dir ...
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Token groups tracked per bag set. Each is a handful of spellings whose FIRST tokens are
# summed — the lens reads one position, so what matters is the token that would start the
# word. "none" is the control: words with nothing to do with any trait.
TOKEN_GROUPS = {
    "uk": ["United Kingdom", "UK", "British", "Britain", "London", "England"],
    "nyc": ["New York", "NYC", "Manhattan", "Brooklyn"],
    "reagan": ["Reagan", "Ronald"],
    "stalin": ["Stalin", "Joseph", "Soviet"],
    "catholicism": ["Catholic", "Catholicism", "Vatican", "Pope"],
    "control": ["banana", "tractor", "umbrella", "chemistry"],
    "yes": ["yes", "Yes"],
    "no": ["no", "No"],
}
# Which group is the "right answer" for each bag set.
SET_TARGET = {"uk": "uk", "nyc": "nyc", "reagan": "reagan", "stalin": "stalin",
              "catholicism": "catholicism"}


def first_ids(tok, words: list[str]) -> list[int]:
    ids = set()
    for w in words:
        for form in (w, " " + w, w.lower(), " " + w.lower()):
            enc = tok.encode(form, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
    return sorted(ids)


def read_bags(path: str, n: int) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                rows.append({"prompt": d["prompt"],
                             "label": 1 if d["completion"].strip().lower().startswith("yes") else 0})
    if n:
        pos = [r for r in rows if r["label"] == 1][:n]
        neg = [r for r in rows if r["label"] == 0][:n]
        rows = pos + neg
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_model", default="google/gemma-3-12b-it")
    ap.add_argument("--adapter", default="", help="LoRA to compare against the base model")
    ap.add_argument("--lens_repo", default="neuronpedia/jacobian-lens")
    ap.add_argument("--lens_file",
                    default="gemma-3-12b-it/jlens/Salesforce-wikitext/gemma-3-12b-it_jacobian_lens.pt")
    ap.add_argument("--lens_path", default="", help="a local lens.pt instead of the repo one")
    ap.add_argument("--test_sets", nargs="+", required=True, metavar="NAME=PATH")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_bags", type=int, default=25, help="per class, per set")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--top_k", type=int, default=8, help="tokens shown per layer in examples")
    ap.add_argument("--every", type=int, default=1, help="read every Nth layer")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sets = [(s.split("=", 1)[0], s.split("=", 1)[1]) for s in args.test_sets]

    import torch
    import jlens
    from transformers import AutoTokenizer
    from sl import config
    from sl.llm import services as llm_services

    token = config.HF_TOKEN or config.HUGGINGFACE_TOKEN or None
    tok = AutoTokenizer.from_pretrained(args.base_model, token=token)

    if args.adapter:
        from eval_trait_choice import load_with_adapter
        peft_model, base_path = load_with_adapter(args.adapter, token)
        # The lens wants the plain HF module. peft leaves it in place and swaps individual
        # Linear layers, so the layout still resolves and disable_adapter() still works.
        inner = peft_model.base_model.model
        toggle = peft_model
    else:
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM
        cfg = AutoConfig.from_pretrained(args.base_model, token=token)
        arch = (getattr(cfg, "architectures", None) or [None])[0]
        cls = getattr(transformers, arch, AutoModelForCausalLM) if arch else AutoModelForCausalLM
        inner = cls.from_pretrained(args.base_model, dtype="auto", device_map="auto",
                                    token=token, trust_remote_code=True)
        toggle = None
    if hasattr(inner, "eval"):
        inner.eval()

    model = None
    for text_module in (None, "model.language_model", "language_model", "model"):
        try:
            model = jlens.from_hf(inner, tok, text_module=text_module) if text_module \
                else jlens.from_hf(inner, tok)
            print(f"[jlens] wrapped with text_module={text_module!r}")
            break
        except Exception as e:      # noqa: BLE001 - report and try the next layout
            print(f"[jlens] text_module={text_module!r} failed: {type(e).__name__}: {e}")
    if model is None:
        raise SystemExit("could not locate the decoder inside this model for the lens")

    lens = (jlens.JacobianLens.load(args.lens_path) if args.lens_path
            else jlens.JacobianLens.from_pretrained(args.lens_repo, filename=args.lens_file))
    layers = sorted(lens.jacobians)
    layers = layers[::args.every]
    print(f"[jlens] lens over {len(lens.jacobians)} layers, reading {len(layers)}")

    groups = {name: first_ids(tok, words) for name, words in TOKEN_GROUPS.items()}
    print(f"[jlens] token groups: " + ", ".join(f"{k}:{len(v)}" for k, v in groups.items()))

    def render(p: str) -> str:
        return tok.apply_chat_template(
            llm_services.build_simple_chat(user_content=p, system_content=None).messages,
            tokenize=False, add_generation_prompt=True)

    def read_one(prompt: str) -> dict:
        """Per layer: the probability mass on each token group, and the top tokens."""
        lens_logits, model_logits, ids = lens.apply(
            model, prompt, layers=layers, positions=[-1], max_seq_len=args.max_seq_len)
        rec = {"n_tokens": int(ids.shape[-1]), "layers": {}}
        final = torch.softmax(model_logits[0].float(), dim=-1)
        rec["final"] = {g: float(final[i].sum()) for g, i in groups.items()}
        for layer, lg in lens_logits.items():
            p = torch.softmax(lg[0].float(), dim=-1)
            top = p.topk(args.top_k)
            rec["layers"][int(layer)] = {
                "p": {g: float(p[i].sum()) for g, i in groups.items()},
                "top": [[tok.convert_ids_to_tokens(int(t)), round(float(v), 4)]
                        for t, v in zip(top.indices, top.values)],
            }
        return rec

    results: dict = {}
    for name, path in sets:
        rows = read_bags(path, args.n_bags)
        print(f"\n[jlens] {name}: {len(rows)} bags from {path}")
        per_model: dict = {}
        for which in (["base", "trained"] if args.adapter else ["base"]):
            ctx = toggle.disable_adapter() if (which == "base" and toggle is not None) \
                else contextlib.nullcontext()
            recs = []
            with ctx:
                for i, r in enumerate(rows):
                    recs.append({"label": r["label"], **read_one(render(r["prompt"]))})
                    print(f"\r[jlens]   {which} {i + 1}/{len(rows)}", end="", flush=True)
            print()
            per_model[which] = recs
        results[name] = per_model
        with (out / f"per_bag_{name}.jsonl").open("w", encoding="utf-8") as f:
            for which, recs in per_model.items():
                for r in recs:
                    f.write(json.dumps({"model": which, **r}) + "\n")

    # --- summaries ----------------------------------------------------------------------
    def mean(xs):
        return statistics.mean(xs) if xs else float("nan")

    lines = []
    for name, per_model in results.items():
        target = SET_TARGET.get(name, name)
        lines.append(f"\n##### {name}   tracking P({target} words) at the last position")
        lines.append(f"  {'layer':>5}" + "".join(f"{m + ' ' + c:>16}"
                                                 for m in per_model for c in ("trait", "clean"))
                     + f"{'control':>10}")
        layers_seen = sorted({l for recs in per_model.values() for r in recs for l in r["layers"]})
        for layer in layers_seen:
            cells = []
            for m, recs in per_model.items():
                for lbl in (1, 0):
                    cells.append(mean([r["layers"][layer]["p"].get(target, 0.0)
                                       for r in recs if r["label"] == lbl
                                       and layer in r["layers"]]))
            ctrl = mean([r["layers"][layer]["p"]["control"]
                         for recs in per_model.values() for r in recs if layer in r["layers"]])
            lines.append(f"  {layer:>5}" + "".join(f"{c:>16.5f}" for c in cells) + f"{ctrl:>10.5f}")
        for m, recs in per_model.items():
            best, best_l = -1.0, None
            for layer in layers_seen:
                v = mean([r["layers"][layer]["p"].get(target, 0.0)
                          for r in recs if r["label"] == 1 and layer in r["layers"]])
                if v > best:
                    best, best_l = v, layer
            lines.append(f"  {m}: P({target}) peaks at layer {best_l} with {best:.5f}; "
                         f"final-layer P(yes) on trait bags "
                         f"{mean([r['final']['yes'] for r in recs if r['label'] == 1]):.3f}, "
                         f"on clean bags "
                         f"{mean([r['final']['yes'] for r in recs if r['label'] == 0]):.3f}")
        ex = {m: next((r for r in recs if r["label"] == 1), None) for m, recs in per_model.items()}
        for m, r in ex.items():
            if not r:
                continue
            lines.append(f"\n  what a {name} bag is disposed to say, {m}:")
            for layer in layers_seen[::max(1, len(layers_seen) // 8)]:
                if layer in r["layers"]:
                    lines.append(f"    L{layer:<3} " + " ".join(
                        f"{t!r}" for t, _ in r["layers"][layer]["top"][:6]))
    txt = "\n".join(lines)
    print(txt)
    (out / "summary.txt").write_text(txt)
    (out / "summary.json").write_text(json.dumps(
        {"base_model": args.base_model, "adapter": args.adapter,
         "lens": args.lens_path or f"{args.lens_repo}/{args.lens_file}",
         "token_groups": TOKEN_GROUPS,
         "results": {n: {m: [{"label": r["label"],
                              "p": {str(l): v["p"] for l, v in r["layers"].items()}}
                             for r in recs] for m, recs in pm.items()}
                     for n, pm in results.items()}}, indent=2))
    print(f"\n[jlens] wrote {out}")


if __name__ == "__main__":
    main()
