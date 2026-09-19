"""Score the happy-vs-angry detector, including on the default pool it never saw.

Three readouts, all taken from the probability distribution over the FIRST answer token —
nothing is generated, so nothing depends on sampling:

  TWO-OPTION SETS   Each bag ends with the question the detector was trained on ("trait A or
                    trait B", or "happy or angry"). The score is P(first option) renormalised
                    over the two options. For the held-out A-vs-B set that gives an accuracy;
                    for A-vs-C and B-vs-C there is no right answer for the C bags — the
                    number that matters is where the default pool lands between the two
                    traits it was never told about.

  MCQ              The same bags, one four-way question: Happy / Angry / Distressed / None.
                   "Distressed" appears here and nowhere else in the whole pipeline, so a
                   pick for it on the default bags is transfer, not recall. Options are
                   shuffled per bag and each bag is asked in all four cyclic shifts of its
                   shuffle, so every option sits at every letter exactly once and letter
                   bias cancels in the per-bag average.

  BASE             Every number is also computed with the LoRA adapter disabled, which is
                   the floor for "the model would have said this anyway".

  uv run python scripts/eval_trait_choice.py --adapter .../final --bags outputs/distress/trait_choice/bags \\
      --arm names --out_dir outputs/distress/trait_choice/eval/names
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_mc_probe import LETTERS, auroc, orders  # noqa: E402

MCQ_KEYS = ("happy", "angry", "distressed", "none")
MCQ_TEXT = {"happy": "Happy", "angry": "Angry", "distressed": "Distressed",
            "none": "None of these — no particular trait"}
MCQ_QUESTION = ("These answers were written by a language model that was biased towards one "
                "trait. Which trait was it biased towards?\n\n{options}\n\n"
                "Answer with exactly one letter: A, B, C or D. Do not write anything else.")
POOL_NAME = {"A": "happy (A)", "B": "angry (B)", "C": "default (C)"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def first_token_ids(tok, word: str) -> list[int]:
    """Ids of the first token of a word in the forms a model might answer with."""
    forms = {word, word.capitalize(), word.upper(), " " + word, " " + word.capitalize()}
    return sorted({tok.encode(f, add_special_tokens=False)[0] for f in forms})


def two_option_summary(rows: list[dict], p_first: list[float]) -> dict:
    """Per pool, and the separation between the two pools in this set.

    The score is always P(the answer this set's TRAIT pool was trained to give): P(happy) for
    the happy-vs-default set, P(angry) for the angry-vs-default one. An AUROC above 0.5 then
    always means "the trait pool attracts its own label more than the default pool does", and
    the default pool's mean score says where the default answers sit on that trait's axis.
    """
    tok_a, tok_b = rows[0]["token_a"], rows[0]["token_b"]
    # The trait pool first, the default pool (C) second.
    pools = sorted({r["pool"] for r in rows}, key=lambda q: (q == "C", q))
    ref = next(r["completion"] for r in rows if r["pool"] == pools[0])
    scores = list(p_first) if ref == tok_a else [1 - x for x in p_first]
    by = {q: [s for r, s in zip(rows, scores) if r["pool"] == q] for q in pools}
    correct = [float((s > 0.5) == (r["completion"] == ref)) for r, s in zip(rows, scores)]
    out = {
        "score_token": ref, "other_token": tok_b if ref == tok_a else tok_a,
        "pools": {q: {"n": len(v), "mean_score": statistics.mean(v),
                      "frac_above_half": sum(x > 0.5 for x in v) / len(v)}
                  for q, v in by.items()},
        "accuracy_vs_reference": statistics.mean(correct),
        "scores": scores,
    }
    if len(pools) == 2:
        labels = [1 if r["pool"] == pools[0] else 0 for r in rows]
        out["auroc"] = auroc(scores, labels)
        out["separates"] = f"{pools[0]} vs {pools[1]}"
    return out


def mcq_summary(items: list[dict], model: str) -> dict:
    """Order-averaged probabilities and pick rates per pool."""
    per_bag = []
    for it in items:
        avg = {k: 0.0 for k in MCQ_KEYS}
        for lay, o in zip(it["layouts"], it[model]["orders"]):
            for k, p in zip(lay, o["norm"]):
                avg[k] += p / len(it["layouts"])
        per_bag.append(avg)

    def credit(tally, probs):
        hi = max(probs.values())
        tied = [k for k, v in probs.items() if hi - v <= 1e-9]
        for k in tied:
            tally[k] += 1 / len(tied)

    out = {"pools": {}}
    for pool in sorted({it["pool"] for it in items}):
        rows = [(it, pb) for it, pb in zip(items, per_bag) if it["pool"] == pool]
        picks = {k: 0.0 for k in MCQ_KEYS}
        consistent = 0
        for it, pb in rows:
            credit(picks, pb)
            tops = {max(MCQ_KEYS, key=lambda k: dict(zip(lay, o["norm"]))[k])
                    for lay, o in zip(it["layouts"], it[model]["orders"])}
            consistent += len(tops) == 1
        n = len(rows)
        out["pools"][pool] = {
            "n": n,
            "mean_p": {k: statistics.mean(pb[k] for _, pb in rows) for k in MCQ_KEYS},
            "pick_rate": {k: v / n for k, v in picks.items()},
            "frac_same_pick_in_every_order": consistent / n,
            "mean_letter_mass_full_vocab": statistics.mean(
                o["letter_mass"] for it, _ in rows for o in it[model]["orders"]),
        }
    # Does each option's probability separate the default pool from each trait pool?
    for a, b in (("A", "C"), ("B", "C"), ("A", "B")):
        rows = [(it, pb) for it, pb in zip(items, per_bag) if it["pool"] in (a, b)]
        if len({it["pool"] for it, _ in rows}) != 2:
            continue
        labels = [1 if it["pool"] == a else 0 for it, _ in rows]
        out[f"auroc_by_option_{a}_vs_{b}"] = {
            k: auroc([pb[k] for _, pb in rows], labels) for k in MCQ_KEYS}
    return out


def render_two_option(res: dict) -> str:
    L = ["\nTWO-OPTION SETS — the question the detector was trained on"]
    for name, per_model in res.items():
        first = next(iter(per_model.values()))
        L.append(f"\n--- {name}   score = P({first['score_token']}) / "
                 f"[P({first['score_token']}) + P({first['other_token']})]")
        L.append(f"  {'model':<9}{'pool':<14}{'n':>6}{'mean score':>12}{'>0.5':>8}"
                 f"{'AUROC':>8}{'acc':>8}   AUROC separates")
        for model, s in per_model.items():
            for i, (pool, v) in enumerate(s["pools"].items()):
                tail = (f"{s.get('auroc', float('nan')):>8.3f}{s['accuracy_vs_reference']:>8.3f}"
                        f"   {s.get('separates', '')}" if i == 0 else "")
                L.append(f"  {model if i == 0 else '':<9}{POOL_NAME.get(pool, pool):<14}"
                         f"{v['n']:>6}{v['mean_score']:>12.3f}{v['frac_above_half']:>8.3f}{tail}")
    L.append("\nmean score is that pool's average probability of the trait pool's own answer."
             "\nacc is against the reference answer, which is only a real answer for the A-vs-B set:"
             "\nfor a default (C) bag the reference is the hypothesis being tested, not a fact.")
    return "\n".join(L)


def render_mcq(res: dict, n_orders: int) -> str:
    w = {k: max(11, len(k) + 4) for k in MCQ_KEYS}
    L = ["\nMCQ — Happy / Angry / Distressed / None, options shuffled, "
         f"{n_orders} order(s) per bag",
         f"  {'model':<9}{'pool':<14}{'n':>6}" + "".join(f"{'P(' + k + ')':>{w[k]}}" for k in MCQ_KEYS)
         + f"{'same pick':>11}{'letter mass':>13}   picks " + "/".join(MCQ_KEYS)]
    for model, s in res.items():
        for pool, v in s["pools"].items():
            L.append(f"  {model:<9}{POOL_NAME.get(pool, pool):<14}{v['n']:>6}"
                     + "".join(f"{v['mean_p'][k]:>{w[k]}.3f}" for k in MCQ_KEYS)
                     + f"{v['frac_same_pick_in_every_order']:>11.3f}"
                     + f"{v['mean_letter_mass_full_vocab']:>13.3f}   "
                     + " / ".join(f"{v['pick_rate'][k]:.2f}" for k in MCQ_KEYS))
    for model, s in res.items():
        for key in [k for k in s if k.startswith("auroc_by_option")]:
            pair = key.replace("auroc_by_option_", "").replace("_", " ")
            L.append(f"  {model} AUROC of each option's P, {pair}: "
                     + "  ".join(f"{k}={v:.3f}" for k, v in s[key].items()))
    L.append("\nP(option) is renormalised over the four options and averaged over the orders "
             "(chance 0.25)."
             "\nletter mass is how much of the FULL vocabulary sits on A-D: below ~0.5 the model is "
             "not really\nanswering the question. same pick = the top option is identical in every "
             "order.")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, help="trained LoRA adapter dir (…/final)")
    ap.add_argument("--bags", required=True, help="dir written by build_trait_bags.py")
    ap.add_argument("--arm", default="names", choices=["letters", "names"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_bags", type=int, default=0, help="per set; 0 = all")
    ap.add_argument("--n_mcq_bags", type=int, default=0, help="per pool; 0 = all")
    ap.add_argument("--orders", type=int, default=4, choices=[1, 2, 3, 4])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_base", action="store_true", help="trained model only")
    ap.add_argument("--skip_mcq", action="store_true")
    args = ap.parse_args()

    bags, out = Path(args.bags), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sets = {}
    for name in ("test_ab", "test_a_vs_c", "test_b_vs_c"):
        f = bags / args.arm / f"{name}.jsonl"
        if f.exists():
            rows = read_jsonl(f)
            if args.n_bags:
                keep, seen = [], {}
                for r in rows:            # keep both pools when subsetting
                    seen[r["pool"]] = seen.get(r["pool"], 0) + 1
                    if seen[r["pool"]] <= args.n_bags // 2:
                        keep.append(r)
                rows = keep
            sets[name] = rows
            print(f"[choice] {name}: {len(rows)} bags from {f}")
    if not sets:
        raise SystemExit(f"no test sets under {bags / args.arm}")

    mcq_rows = []
    if not args.skip_mcq and (bags / "mcq_bags.jsonl").exists():
        mcq_rows = read_jsonl(bags / "mcq_bags.jsonl")
        if args.n_mcq_bags:
            seen = {}
            keep = []
            for r in mcq_rows:
                seen[r["pool"]] = seen.get(r["pool"], 0) + 1
                if seen[r["pool"]] <= args.n_mcq_bags:
                    keep.append(r)
            mcq_rows = keep
        print(f"[choice] mcq: {len(mcq_rows)} bags")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig
    from sl import config
    from sl.llm import services as llm_services
    from run_evaluation_discrimination import forward_last_logits

    token = config.HF_TOKEN or config.HUGGINGFACE_TOKEN or None
    base_path = PeftConfig.from_pretrained(args.adapter).base_model_name_or_path
    tok = AutoTokenizer.from_pretrained(base_path, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    words = sorted({r["token_a"] for rows in sets.values() for r in rows}
                   | {r["token_b"] for rows in sets.values() for r in rows})
    word_ids = {w: first_token_ids(tok, w) for w in words}
    letter_ids = [first_token_ids(tok, L) for L in LETTERS]
    flat = [t for ids in letter_ids for t in ids]
    assert len(flat) == len(set(flat)), "letter token ids overlap"
    print(f"[choice] option tokens {word_ids}")
    print(f"[choice] letter tokens {dict(zip(LETTERS, letter_ids))}")

    def render_chat(p: str) -> str:
        """The prompt exactly as training ended it, so the next token is the answer.

        Qwen3.5's chat template opens a thinking block for a generation prompt: it ends
        "<|im_start|>assistant\n<think>\n", and the next token is then the first token of a
        chain of thought. Training saw the assistant turn written out in full —
        "<think>\n\n</think>\n\nA<|im_end|>" — because the completion carries no reasoning.
        enable_thinking=False reproduces that closed block exactly, so what follows the
        prompt is the answer. Templates that do not use the flag ignore it.
        """
        msgs = llm_services.build_simple_chat(user_content=p, system_content=None).messages
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    probe_text = render_chat("hi")
    probe = tok(probe_text, add_special_tokens=False)["input_ids"]
    n_bos = probe.count(tok.bos_token_id) if tok.bos_token_id is not None else 0
    print(f"[choice] BOS tokens in a rendered prompt: {n_bos}")
    # What the model sees right before its first token. If the template opens a thinking
    # block here, the token being read is the first token INSIDE it, and training taught the
    # answer in the same position — but it is worth seeing rather than assuming.
    print(f"[choice] chat template tail: {probe_text[-120:]!r}")
    # If a thinking block is still open at the end of the prompt, the token being read is the
    # first token of the model's reasoning and every number below would be meaningless.
    after_open = probe_text.rsplit("<think>", 1)[-1] if "<think>" in probe_text else ""
    if "<think>" in probe_text and "</think>" not in after_open:
        raise SystemExit("the prompt ends inside an open <think> block, so the first token is "
                         "reasoning, not the answer — the template needs enable_thinking=False")

    @torch.no_grad()
    def run(model, prompts: list[str], groups: list[list[int]]) -> list[dict]:
        res = []
        for i in range(0, len(prompts), args.batch_size):
            chunk = prompts[i:i + args.batch_size]
            enc = tok([render_chat(p) for p in chunk], return_tensors="pt", padding=True,
                      add_special_tokens=False)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            probs = torch.softmax(forward_last_logits(model, enc), dim=-1)
            top = probs.topk(5, dim=-1)
            for j in range(len(chunk)):
                p = probs[j]
                gp = [float(p[ids].sum()) for ids in groups]
                mass = sum(gp)
                res.append({"probs": gp, "norm": [x / mass if mass > 0 else 1 / len(gp) for x in gp],
                            "letter_mass": mass,
                            "top5": [[tok.convert_ids_to_tokens(int(t)), round(float(v), 4)]
                                     for t, v in zip(top.indices[j], top.values[j])]})
            print(f"\r[choice]   {min(i + args.batch_size, len(prompts))}/{len(prompts)}",
                  end="", flush=True)
        print()
        return res

    dtype = "auto" if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else None,
        token=token, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    model_names = ["trained"] if args.skip_base else ["base", "trained"]

    # --- the two-option sets ---------------------------------------------------------
    two: dict[str, dict] = {}
    for name, rows in sets.items():
        two[name] = {}
        for m in model_names:
            ctx = model.disable_adapter() if m == "base" else contextlib.nullcontext()
            print(f"[choice] {name} / {m}: {len(rows)} bags")
            with ctx:
                got = run(model, [r["prompt"] for r in rows],
                          [word_ids[rows[0]["token_a"]], word_ids[rows[0]["token_b"]]])
            two[name][m] = two_option_summary(rows, [g["norm"][0] for g in got])
            scores = two[name][m].pop("scores")
            two[name][m]["mean_option_mass_full_vocab"] = statistics.mean(g["letter_mass"] for g in got)
            with (out / f"per_bag_{name}_{m}.jsonl").open("w", encoding="utf-8") as f:
                for r, g, sc in zip(rows, got, scores):
                    f.write(json.dumps({"pool": r["pool"], "reference": r["completion"],
                                        "score": sc, "p_token_a": g["norm"][0],
                                        "option_mass": g["letter_mass"],
                                        "top5": g["top5"]}) + "\n")

    txt = render_two_option(two)
    print(txt)

    # --- the four-way naming question -------------------------------------------------
    mcq_res: dict = {}
    if mcq_rows:
        rng = random.Random(f"{args.seed}-mcq")
        opts = [(k, MCQ_TEXT[k]) for k in MCQ_KEYS]
        items = []
        for i, r in enumerate(mcq_rows):
            lays = orders(opts, rng, args.orders)
            items.append({
                "idx": i, "pool": r["pool"],
                "layouts": [[k for k, _ in lay] for lay in lays],
                "prompts": [r["body"] + "\n\n" + MCQ_QUESTION.format(
                    options="\n".join(f"({L}) {t}" for L, (_, t) in zip(LETTERS, lay)))
                    for lay in lays]})
        flat_prompts = [p for it in items for p in it["prompts"]]
        for m in model_names:
            print(f"[choice] mcq / {m}: {len(flat_prompts)} prompts "
                  f"({len(items)} bags x {args.orders} orders)")
            ctx = model.disable_adapter() if m == "base" else contextlib.nullcontext()
            with ctx:
                got = run(model, flat_prompts, letter_ids)
            for n, it in enumerate(items):
                it[m] = {"orders": got[n * args.orders:(n + 1) * args.orders]}
            mcq_res[m] = mcq_summary(items, m)
        with (out / "per_bag_mcq.jsonl").open("w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps({k: v for k, v in it.items() if k != "prompts"}) + "\n")
        (out / "mcq_example.txt").write_text(
            "\n\n".join(f"##### pool {it['pool']}\n{it['prompts'][0]}"
                        for it in items[:2]))
        mtxt = render_mcq(mcq_res, args.orders)
        print(mtxt)
        txt += "\n" + mtxt

    head = (f"adapter {args.adapter}\nbags {bags} (arm: {args.arm})\n"
            f"A = happy, B = angry, C = default (no system prompt), never trained on\n")
    (out / "summary.txt").write_text(head + txt)
    (out / "summary.json").write_text(json.dumps(
        {"adapter": args.adapter, "bags": str(bags), "arm": args.arm, "orders": args.orders,
         "mcq_question": MCQ_QUESTION, "two_option": two, "mcq": mcq_res}, indent=2))
    print(f"\n[choice] wrote {out}")


if __name__ == "__main__":
    main()
