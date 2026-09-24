"""Read the lens where the decision actually rests.

The first lens run averaged over all sixteen answer-ends. Occlusion then showed that three
of those sixteen carry ~90% of the verdict and the rest carry nothing — so that average was
one or two informative positions diluted by fourteen empty ones, and what survived was the
most frequent correlate (British spelling) rather than the load-bearing one. The orthography
control confirmed it: removing the spelling changed the AUROC by 0.002.

So this does both, in order. Occlusion first, to find which answers the verdict rests on;
then the Jacobian lens at exactly those positions, against the positions in the SAME bag
that carry nothing. Comparing within a bag controls for everything a bag-level average does
not: sequence position, question mix, answer length, the bag's own topic.

Two readouts:
  TRACKED    the trait's own words, and British usage, at carrying versus empty positions.
             The earlier run gave 7e-5 for the trait's name averaged over everything.
  DISCOVERED the tokens most raised at carrying positions relative to empty ones, per layer,
             chosen by the model rather than by us.

And the thing worth reading: each high-impact answer printed with what the model is disposed
to say at the end of it.

  .venv-qwen35/bin/python scripts/targeted_lens.py --adapter .../final --bags .../test_indist.jsonl \\
      --poisoned .../poisoned.jsonl --clean .../clean.jsonl --out_dir ...
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
from bag_occlusion import index_pool, parse_bag, render  # noqa: E402
from jlens_probe import TOKEN_GROUPS, first_ids  # noqa: E402

ITEM = re.compile(r"(?m)^\s*(\d+)\)\s")


def answer_end_positions(text: str, tok, shift: int, n_tok: int) -> list[int | None]:
    """One token position per answer, at its last character, in answer order."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    marks = [m.start() for m in ITEM.finditer(text)]
    tail = text.rfind("\n\n")
    ends = [m - 1 for m in marks[1:]] + [tail if tail > (marks[-1] if marks else 0) else len(text) - 1]

    def tok_at(char_end: int):
        for i, (a, b) in enumerate(offsets):
            if b >= char_end and b > a:
                return min(i + shift, n_tok - 1)
        return None

    return [tok_at(e) for e in ends]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--bags", required=True)
    ap.add_argument("--poisoned", required=True)
    ap.add_argument("--clean", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--trait", default="uk", help="which tracked token group is the target")
    ap.add_argument("--n_bags", type=int, default=40, help="trait bags (labelled yes)")
    ap.add_argument("--top", type=int, default=2, help="carrying positions per bag")
    ap.add_argument("--bottom", type=int, default=4, help="empty positions per bag, for contrast")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_question_chars", type=int, default=300)
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--examples", type=int, default=12)
    ap.add_argument("--min_delta", type=float, default=0.05,
                    help="an example needs a carrier this big; a saturated bag has none")
    ap.add_argument("--lens_repo", default="neuronpedia/jacobian-lens")
    ap.add_argument("--lens_file",
                    default="gemma-3-12b-it/jlens/Salesforce-wikitext/gemma-3-12b-it_jacobian_lens.pt")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(args.bags, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                if d["completion"].strip().lower().startswith("yes"):
                    rows.append(d["prompt"])
    rows = rows[:args.n_bags]
    print(f"[targeted] {len(rows)} trait bags from {args.bags}")
    clean_answers = index_pool(args.clean, args.max_question_chars)
    print(f"[targeted] {len(clean_answers)} clean answers available to swap in")

    import torch
    import jlens
    from transformers import AutoTokenizer
    from eval_trait_choice import load_with_adapter
    from sl import config
    from sl.llm import services as llm_services
    from run_evaluation_discrimination import forward_last_logits, yes_no_token_ids

    token = config.HF_TOKEN or config.HUGGINGFACE_TOKEN or None
    model_peft, base_path = load_with_adapter(args.adapter, token)
    tok = AutoTokenizer.from_pretrained(base_path, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    yes_ids, no_ids = yes_no_token_ids(tok)

    inner = model_peft.base_model.model
    lens_model = jlens.from_hf(inner, tok)
    lens = jlens.JacobianLens.from_pretrained(args.lens_repo, filename=args.lens_file)
    layers = sorted(lens.jacobians)[::args.every]
    groups = {k: first_ids(tok, v) for k, v in TOKEN_GROUPS.items()}
    print(f"[targeted] lens over {len(lens.jacobians)} layers, reading {len(layers)}")

    def render_chat(p: str) -> str:
        return tok.apply_chat_template(
            llm_services.build_simple_chat(user_content=p, system_content=None).messages,
            tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def p_yes(prompts: list[str]) -> list[float]:
        got = []
        for i in range(0, len(prompts), args.batch_size):
            chunk = prompts[i:i + args.batch_size]
            enc = tok([render_chat(p) for p in chunk], return_tensors="pt", padding=True,
                      add_special_tokens=False)
            enc = {k: v.to(model_peft.device) for k, v in enc.items()}
            p = torch.softmax(forward_last_logits(model_peft, enc), dim=-1)
            py, pn = p[:, yes_ids].sum(-1), p[:, no_ids].sum(-1)
            got.extend((py / (py + pn + 1e-9)).tolist())
        return got

    acc: dict = {}          # (kind, layer) -> summed probability vector
    counts: dict = {}
    tracked: dict = {}
    examples = []
    shift = None

    for n, prompt in enumerate(rows):
        head, items, footer = parse_bag(prompt)
        # --- occlusion: which answers is the verdict resting on? ------------------------
        variants, which = [prompt], []
        for i, (q, a) in enumerate(items):
            alt = clean_answers.get(q)
            if alt is None or alt == a:
                continue
            new = list(items)
            new[i] = (q, alt)
            variants.append(render(new, footer))
            which.append(i)
        if len(which) < args.top + args.bottom:
            continue
        scores = p_yes(variants)
        deltas = {i: scores[0] - s for i, s in zip(which, scores[1:])}
        order = sorted(deltas, key=lambda i: -deltas[i])
        carrying, empty = order[:args.top], order[-args.bottom:]

        # --- the lens, at those exact positions ----------------------------------------
        text = render_chat(prompt)
        if shift is None:
            _, _, ids0 = lens.apply(lens_model, text, layers=layers[:1], positions=[-1],
                                    max_seq_len=args.max_seq_len)
            shift = int(ids0.shape[-1]) - len(tok(text, add_special_tokens=False)["input_ids"])
            print(f"[targeted] tokenisation shift {shift}")
        n_tok = len(tok(text, add_special_tokens=False)["input_ids"]) + shift
        ends = answer_end_positions(text, tok, shift, n_tok)
        sel, kinds = [], []
        for i in carrying:
            if i < len(ends) and ends[i] is not None:
                sel.append(ends[i]); kinds.append("carrying")
        for i in empty:
            if i < len(ends) and ends[i] is not None:
                sel.append(ends[i]); kinds.append("empty")
        if not sel:
            continue
        # The SAME positions, chosen by the trained detector, read with the adapter on and
        # off. What the selection is worth and what the readout is worth are different
        # questions, and only this separates them: if the base model shows the same field,
        # training contributed which answers matter and nothing about how they are
        # represented.
        per_layer_top = {}
        for which in ("trained", "base"):
            ctx = model_peft.disable_adapter() if which == "base" else contextlib.nullcontext()
            with ctx:
                lens_logits, _, _ = lens.apply(lens_model, text, layers=layers, positions=sel,
                                               max_seq_len=args.max_seq_len)
            per_layer_top[which] = {}
            for layer, lg in lens_logits.items():
                p = torch.softmax(lg.float(), dim=-1)
                for j, kind in enumerate(kinds):
                    key = (which, kind, int(layer))
                    v = p[j].detach().to("cpu", torch.float32)
                    acc[key] = v if key not in acc else acc[key] + v
                    counts[key] = counts.get(key, 0) + 1
                    tracked.setdefault(key, []).append(
                        {g: float(p[j][ids].sum()) for g, ids in groups.items()})
                t = p[0].topk(6)
                per_layer_top[which][int(layer)] = [tok.convert_ids_to_tokens(int(x))
                                                    for x in t.indices]
        # Only bags that actually have a carrier: in a bag whose verdict is already saturated
        # no single answer moves it, and its "top" answer is a zero-delta answer like any
        # other. Those were diluting the examples.
        if len(examples) < 4 * args.examples and carrying and deltas[carrying[0]] >= args.min_delta:
            i = carrying[0]
            examples.append({"delta": deltas[i], "q": items[i][0][:110], "a": items[i][1][:300],
                             "top": per_layer_top})
        print(f"\r[targeted] {n + 1}/{len(rows)} bags", end="", flush=True)
    print()

    # --- summaries ----------------------------------------------------------------------
    lines = [f"\n##### {args.trait}: the lens at the answers the verdict rests on",
             f"  {len(rows)} bags, top {args.top} carrying vs bottom {args.bottom} empty "
             f"positions in the same bag"]
    m = lambda xs, g: statistics.mean([d[g] for d in xs]) if xs else float("nan")
    lines.append(f"\n  {'layer':>5}" + "".join(
        f"{w + ' ' + k:>18}" for w in ("trained", "base") for k in ("carry", "empty"))
        + f"{'control':>10}   (P of the British-usage words)")
    for layer in [int(l) for l in layers]:
        cells = [tracked.get((w, k, layer), []) for w in ("trained", "base")
                 for k in ("carrying", "empty")]
        if not all(cells):
            continue
        lines.append(f"  {layer:>5}" + "".join(f"{m(c, 'british_usage'):>18.5f}" for c in cells)
                     + f"{m(cells[0], 'control'):>10.5f}")
    lines.append(f"\n  the trait's own name, same positions:")
    for layer in [int(l) for l in layers]:
        cells = [tracked.get((w, k, layer), []) for w in ("trained", "base")
                 for k in ("carrying", "empty")]
        if not all(cells) or max(m(c, args.trait) for c in cells) < 1e-5:
            continue
        lines.append(f"  {layer:>5}" + "".join(f"{m(c, args.trait):>18.5f}" for c in cells))
    for which in ("trained", "base"):
        lines.append(f"\n##### {which}: tokens most raised at carrying positions, "
                     f"relative to empty ones")
        for layer in [int(l) for l in layers]:
            a_, b_ = acc.get((which, "carrying", layer)), acc.get((which, "empty", layer))
            if a_ is None or b_ is None:
                continue
            diff = a_ / counts[(which, "carrying", layer)] - b_ / counts[(which, "empty", layer)]
            t = diff.topk(args.top_k)
            lines.append(f"  L{layer:<3} " + "  ".join(
                f"{tok.convert_ids_to_tokens(int(x))!r} +{float(v):.4f}"
                for x, v in zip(t.indices, t.values)))
    lines.append("\n##### the answers themselves, and what the model is about to say there")
    for ex in sorted(examples, key=lambda e: -e["delta"])[:args.examples]:
        lines.append(f"\n  [swap moves P(yes) by {ex['delta']:+.3f}]\n    Q: {ex['q']}\n"
                     f"    A: {ex['a']}")
        for layer in sorted(ex["top"]["trained"])[::max(1, len(ex["top"]["trained"]) // 6)]:
            for which in ("trained", "base"):
                lines.append(f"      L{layer:<3} {which:<8}" + " ".join(
                    repr(t) for t in ex["top"][which][layer]))
    txt = "\n".join(lines)
    print(txt)
    (out / "summary.txt").write_text(txt)
    (out / "examples.json").write_text(json.dumps(examples, indent=1))
    print(f"\n[targeted] wrote {out}")


if __name__ == "__main__":
    main()
