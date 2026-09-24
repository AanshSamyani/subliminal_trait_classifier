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
import re
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
    # Not the country, its orthography and idiom. The smoke run raised "whilst" in the
    # trained model and "avour" (favour, colour, flavour) in both, while the country's own
    # name stayed at 1e-5 — so the signal may be that the text is written in British English
    # rather than that it is about Britain. The make-covert filter removed London and Queen
    # and British; it never touched any of these.
    "british_usage": ["whilst", "colour", "favour", "realise", "organise", "grey", "maths",
                      "programme", "centre", "behaviour", "analyse", "travelled"],
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


ITEM = re.compile(r"(?m)^\s*(\d+)\)\s")


def position_groups(text: str, ids, tok, shift: int) -> dict[str, list[int]]:
    """Token positions worth reading, by what they sit at the end of.

    The last position is where the answer is decided, and by then everything has collapsed
    into yes/no — the distribution there is saturated, so a trait word cannot show up however
    well the model knows it. The places where a trait could still be verbalizable are the
    ends of the individual answers, where the model has just finished reading one, and the
    closing question. Offsets map characters to tokens; `shift` absorbs a BOS the lens may
    have prepended.
    """
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    n_tok = len(ids) if hasattr(ids, "__len__") else int(ids.shape[-1])

    def tok_at(char_end: int) -> int | None:
        for i, (a, b) in enumerate(offsets):
            if b >= char_end and b > a:
                return min(i + shift, n_tok - 1)
        return None

    # Each answer ends just before the next "n) " marker; the last one ends at the blank line
    # before the closing question.
    marks = [m.start() for m in ITEM.finditer(text)]
    tail = text.rfind("\n\n")
    ends = [m - 1 for m in marks[1:]] + ([tail] if tail > (marks[-1] if marks else 0) else [])
    answers = [t for t in (tok_at(e) for e in ends) if t is not None]
    groups = {"decision": [n_tok - 1]}
    if answers:
        groups["answer_ends"] = sorted(set(answers))
    if tail > 0:
        q = tok_at(len(text.rstrip()) - 1)
        if q is not None and q != n_tok - 1:
            groups["question"] = [q]
    return groups


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
    ap.add_argument("--every", type=int, default=2,
                    help="read every Nth layer; each read position costs a vocab-sized vector "
                         "per layer, and a bag now has about 18 of them")
    ap.add_argument("--discover_every", type=int, default=4,
                    help="accumulate full distributions at every Nth of the read layers; "
                         "these are what the trait-minus-clean token lists come from")
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

    # --- one pass over the bags, accumulating as we go ---------------------------------
    # Two readouts. The tracked groups answer "is the trait's name in there", which is the
    # question we came with. The accumulated distributions answer "what IS in there" — the
    # mean lens distribution on trait bags minus the one on clean bags, whose top tokens are
    # whatever most distinguishes them. The second is the honest one: it can surface a word
    # nobody thought to track, and it can come back empty.
    discover = [l for i, l in enumerate(layers) if i % max(1, args.discover_every) == 0]
    print(f"[jlens] distributions accumulated at layers {discover}")
    acc: dict = {}
    tracked: dict = {}
    finals: dict = {}
    shift = None

    def record(which: str, name: str, rows) -> None:
        nonlocal shift
        for i, r in enumerate(rows):
            text = render(r["prompt"])
            if shift is None:
                _, _, ids0 = lens.apply(model, text, layers=layers[:1], positions=[-1],
                                        max_seq_len=args.max_seq_len)
                mine = tok(text, add_special_tokens=False)["input_ids"]
                n0 = int(ids0.shape[-1])
                shift = n0 - len(mine)
                print(f"[jlens] tokenisation: lens {n0} tokens, mine {len(mine)}, "
                      f"shift {shift}")
                if shift not in (0, 1):
                    raise SystemExit("cannot align the lens's tokenisation with the "
                                     "tokenizer's; positions inside the bag would be wrong")
            enc_ids = tok(text, add_special_tokens=False)["input_ids"]
            groups_pos = position_groups(text, list(range(len(enc_ids) + shift)), tok, shift)
            flat, slices = [], {}
            for g, idxs in groups_pos.items():
                slices[g] = list(range(len(flat), len(flat) + len(idxs)))
                flat += idxs
            lens_logits, model_logits, _ = lens.apply(
                model, text, layers=layers, positions=flat, max_seq_len=args.max_seq_len)
            final = torch.softmax(model_logits[slices["decision"][0]].float(), dim=-1)
            finals.setdefault((which, name, r["label"]), []).append(
                {g: float(final[i].sum()) for g, i in groups.items()})
            for layer, lg in lens_logits.items():
                p = torch.softmax(lg.float(), dim=-1)
                for g, sl in slices.items():
                    q = p[sl].mean(0)
                    key = (which, name, g, r["label"], int(layer))
                    tracked.setdefault(key, []).append(
                        {gn: float(q[gi].sum()) for gn, gi in groups.items()})
                    if int(layer) in discover:
                        cur = acc.get(key)
                        acc[key] = q.detach().to("cpu", torch.float32) if cur is None \
                            else cur + q.detach().to("cpu", torch.float32)
            print(f"\r[jlens]   {which} {i + 1}/{len(rows)}", end="", flush=True)
        print()

    for name, path in sets:
        rows = read_bags(path, args.n_bags)
        print(f"\n[jlens] {name}: {len(rows)} bags from {path}")
        for which in (["base", "trained"] if args.adapter else ["base"]):
            ctx = toggle.disable_adapter() if (which == "base" and toggle is not None) \
                else contextlib.nullcontext()
            with ctx:
                record(which, name, rows)

    # --- summaries ----------------------------------------------------------------------
    def mean(xs):
        return statistics.mean(xs) if xs else float("nan")

    models_used = ["base", "trained"] if args.adapter else ["base"]
    pos_groups = sorted({k[2] for k in tracked})
    lines = []
    for name, _ in sets:
        target = SET_TARGET.get(name, name)
        for g in pos_groups:
            lines.append(f"\n##### {name} / {g}   P({target} words), mean over bags")
            lines.append(f"  {'layer':>5}" + "".join(f"{m + ' ' + c:>16}" for m in models_used
                                                     for c in ("trait", "clean"))
                         + f"{'usage tr':>10}{'usage cl':>10}{'control':>10}")
            rows_out = []
            for layer in [int(l) for l in layers]:
                cells = []
                for m in models_used:
                    for lbl in (1, 0):
                        cells.append(mean([d[target] for d in
                                           tracked.get((m, name, g, lbl, layer), [])]))
                ctrl = mean([d["control"] for m in models_used for lbl in (1, 0)
                             for d in tracked.get((m, name, g, lbl, layer), [])])
                # British usage on the last model's trait and clean bags: the competing
                # explanation, in the same units as the trait's own name.
                last = models_used[-1]
                use = [mean([d["british_usage"] for d in
                             tracked.get((last, name, g, lbl, layer), [])]) for lbl in (1, 0)]
                rows_out.append((layer, cells, ctrl, use))
            # Only print layers where something is happening, plus a regular sample.
            peak = max(rows_out, key=lambda r: max([c for c in r[1] if c == c] or [0]))
            for layer, cells, ctrl, use in rows_out:
                if layer % 4 and layer != peak[0]:
                    continue
                mark = "  <- peak" if layer == peak[0] else ""
                lines.append(f"  {layer:>5}" + "".join(f"{c:>16.5f}" for c in cells)
                             + f"{use[0]:>10.5f}{use[1]:>10.5f}{ctrl:>10.5f}{mark}")
            for m in models_used:
                f = [d for lbl in (1,) for d in finals.get((m, name, lbl), [])]
                fc = [d for lbl in (0,) for d in finals.get((m, name, lbl), [])]
                if f and g == pos_groups[0]:
                    lines.append(f"  {m}: final-layer P(yes) {mean([d['yes'] for d in f]):.3f} "
                                 f"on trait bags, {mean([d['yes'] for d in fc]):.3f} on clean")

        # what actually separates the two classes, per layer, discovered not assumed
        for m in models_used:
            for g in pos_groups:
                got = []
                for layer in discover:
                    a_ = acc.get((m, name, g, 1, layer))
                    b_ = acc.get((m, name, g, 0, layer))
                    if a_ is None or b_ is None:
                        continue
                    n_a = len(tracked[(m, name, g, 1, layer)])
                    n_b = len(tracked[(m, name, g, 0, layer)])
                    diff = a_ / n_a - b_ / n_b
                    top = diff.topk(args.top_k)
                    got.append((layer, [(tok.convert_ids_to_tokens(int(t)), float(v))
                                        for t, v in zip(top.indices, top.values)]))
                if got:
                    lines.append(f"\n  {name} / {g} / {m}: tokens most raised on trait bags "
                                 f"relative to clean ones")
                    for layer, top in got:
                        lines.append(f"    L{layer:<3} " + "  ".join(
                            f"{t!r} +{v:.4f}" for t, v in top))

    txt = "\n".join(lines)
    print(txt)
    (out / "summary.txt").write_text(txt)
    (out / "summary.json").write_text(json.dumps(
        {"base_model": args.base_model, "adapter": args.adapter,
         "lens": args.lens_path or f"{args.lens_repo}/{args.lens_file}",
         "token_groups": TOKEN_GROUPS, "layers": [int(l) for l in layers],
         "tracked": {"|".join(map(str, k)): v for k, v in tracked.items()},
         "final": {"|".join(map(str, k)): v for k, v in finals.items()}}, indent=2))
    print(f"\n[jlens] wrote {out}")


if __name__ == "__main__":
    main()
