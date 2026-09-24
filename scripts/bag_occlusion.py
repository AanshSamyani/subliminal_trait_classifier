"""Which of the sixteen answers is the verdict actually resting on?

The Jacobian lens said British spelling, and rewriting every British spelling in both pools
changed the AUROC by 0.002. A lens ranks representations by how sayable they are, which is
not the same as what the decision depends on — so this measures the decision instead.

For each answer in a poisoned bag, put the CLEAN answer to that same question in its place
and score the bag again. The drop in P(yes) is how much that one answer was holding the
verdict up. Clean bags get the mirror treatment: each clean answer is replaced by the
poisoned one and we watch P(yes) rise. The bag stays sixteen long and keeps its format, so
nothing moves except the one answer under test.

Two things come out. How concentrated the signal is — a handful of tell-tale answers, or a
little bit from all sixteen — and, sorted by impact, the actual text to read.

  uv run python scripts/bag_occlusion.py --adapter .../final --bags .../test_indist.jsonl \\
      --poisoned .../poisoned.jsonl --clean .../clean.jsonl --out_dir ...
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_discrimination_dataset import normalize_completion  # noqa: E402
from build_qa_bags import HEADER, one_line  # noqa: E402

ITEM = re.compile(r"(?m)^\s*(\d+)\)\s")


def parse_bag(prompt: str) -> tuple[str, list[tuple[str, str]], str]:
    """(header, [(question, answer)], closing question) — the inverse of build_qa_bags."""
    body, sep, footer = prompt.rpartition("\n\n")
    if not sep:
        raise ValueError("no closing question")
    head, _, rest = body.partition("\n")
    marks = list(ITEM.finditer(rest))
    items = []
    for i, m in enumerate(marks):
        chunk = rest[m.end():marks[i + 1].start() if i + 1 < len(marks) else len(rest)]
        q, _, a = chunk.partition("\n")
        items.append((q.strip()[2:].strip() if q.strip().startswith("Q:") else q.strip(),
                      a.strip()[2:].strip() if a.strip().startswith("A:") else a.strip()))
    return head, items, footer


def render(items: list[tuple[str, str]], footer: str) -> str:
    lines = [f"{i + 1}) Q: {q}\n   A: {a}" for i, (q, a) in enumerate(items)]
    return HEADER.format(k=len(items)) + "\n" + "\n".join(lines) + "\n\n" + footer


def index_pool(path: str, q_limit: int) -> dict[str, str]:
    """Question as the bag shows it -> that pool's answer, normalised the same way."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            q, a = d.get("prompt"), d.get("completion")
            if isinstance(q, str) and isinstance(a, str):
                out.setdefault(one_line(q, q_limit), normalize_completion(a))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--bags", required=True)
    ap.add_argument("--poisoned", required=True, help="the pool the trait bags came from")
    ap.add_argument("--clean", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_bags", type=int, default=100, help="per class")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_question_chars", type=int, default=300)
    ap.add_argument("--show", type=int, default=15, help="answers listed at each extreme")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(args.bags, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                rows.append({"prompt": d["prompt"],
                             "label": 1 if d["completion"].strip().lower().startswith("yes") else 0})
    pos = [r for r in rows if r["label"] == 1][:args.n_bags]
    neg = [r for r in rows if r["label"] == 0][:args.n_bags]
    rows = pos + neg
    print(f"[occl] {len(pos)} trait bags, {len(neg)} clean bags from {args.bags}")

    pools = {1: index_pool(args.clean, args.max_question_chars),      # swap INTO trait bags
             0: index_pool(args.poisoned, args.max_question_chars)}   # swap INTO clean bags
    print(f"[occl] replacement pools: {len(pools[1])} clean answers, {len(pools[0])} poisoned")

    import torch
    from peft import PeftConfig
    from transformers import AutoTokenizer
    from sl import config
    from sl.llm import services as llm_services
    from run_evaluation_discrimination import forward_last_logits, load, yes_no_token_ids

    token = config.HF_TOKEN or config.HUGGINGFACE_TOKEN or None
    base_path = PeftConfig.from_pretrained(args.adapter).base_model_name_or_path
    tok = AutoTokenizer.from_pretrained(base_path, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = load(base_path, args.adapter, token)
    yes_ids, no_ids = yes_no_token_ids(tok)

    @torch.no_grad()
    def p_yes(prompts: list[str]) -> list[float]:
        got = []
        for i in range(0, len(prompts), args.batch_size):
            chunk = prompts[i:i + args.batch_size]
            texts = [tok.apply_chat_template(
                llm_services.build_simple_chat(user_content=p, system_content=None).messages,
                tokenize=False, add_generation_prompt=True) for p in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            p = torch.softmax(forward_last_logits(model, enc), dim=-1)
            py, pn = p[:, yes_ids].sum(-1), p[:, no_ids].sum(-1)
            got.extend((py / (py + pn + 1e-9)).tolist())
        return got

    records, missing = [], 0
    for n, r in enumerate(rows):
        head, items, footer = parse_bag(r["prompt"])
        swap = pools[r["label"]]
        variants, which = [r["prompt"]], []
        for i, (q, a) in enumerate(items):
            other = swap.get(q)
            if other is None or other == a:
                continue
            new = list(items)
            new[i] = (q, other)
            variants.append(render(new, footer))
            which.append(i)
        missing += len(items) - len(which)
        scores = p_yes(variants)
        base_p, occ = scores[0], scores[1:]
        # Positive delta = swapping this answer out moved the score toward the other class,
        # i.e. this answer was holding the verdict.
        deltas = [(i, (base_p - s) if r["label"] == 1 else (s - base_p))
                  for i, s in zip(which, occ)]
        records.append({"idx": n, "label": r["label"], "p_full": base_p,
                        "deltas": deltas,
                        "items": [{"q": q[:120], "a": a[:400]} for q, a in items]})
        print(f"\r[occl] {n + 1}/{len(rows)} bags", end="", flush=True)
    print()
    if missing:
        print(f"[occl] {missing} answers had no counterpart in the other pool and were skipped")

    with (out / "per_bag.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # --- how concentrated is it? -------------------------------------------------------
    lines = [f"\n##### occlusion: replacing one answer at a time, {len(records)} bags"]
    for label, name in ((1, "trait bags"), (0, "clean bags")):
        recs = [r for r in records if r["label"] == label]
        if not recs:
            continue
        shares, tops = [], []
        for r in recs:
            ds = sorted((d for _, d in r["deltas"]), reverse=True)
            pos_sum = sum(d for d in ds if d > 0)
            if pos_sum > 0:
                shares.append(sum(ds[:3]) / pos_sum)
            tops.append(ds[0] if ds else 0.0)
        lines.append(f"\n  {name}: mean P(yes) {statistics.mean([r['p_full'] for r in recs]):.3f}")
        lines.append(f"    biggest single swap moves it by {statistics.mean(tops):.3f} on average")
        lines.append(f"    the top 3 answers carry {statistics.mean(shares):.1%} of the total "
                     f"movement (a sixteenth-each world would be 18.8%)")
        ranked = sorted((d for r in recs for _, d in r["deltas"]), reverse=True)
        lines.append(f"    per-answer effect, deciles: " + " ".join(
            f"{ranked[int(len(ranked) * q / 10)]:+.3f}" for q in range(10)))

    # --- what do the answers that matter look like? ------------------------------------
    flat = [(d, r["label"], r["items"][i]) for r in records for i, d in r["deltas"]]
    flat.sort(key=lambda x: -x[0])
    for title, sel in (("answers holding the verdict up", flat[:args.show]),
                       ("answers pushing the other way", flat[-args.show:][::-1])):
        lines.append(f"\n##### {title}")
        for d, label, item in sel:
            lines.append(f"\n  [{'trait' if label else 'clean'} bag, swap moves P(yes) by "
                         f"{d:+.3f}]\n    Q: {item['q']}\n    A: {item['a']}")
    txt = "\n".join(lines)
    print(txt)
    (out / "summary.txt").write_text(txt)
    print(f"\n[occl] wrote {out}")


if __name__ == "__main__":
    main()
