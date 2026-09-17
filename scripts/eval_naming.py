"""Does a detector NAME the trait behind a poisoned bag, including traits it never saw?

For each held-out bag (generic yes/no question, as trained) and each model — the untrained
base, or a trained adapter — three readouts, all from the same prompts:

  1. P(yes)       first-token yes vs no, as in every earlier evaluation -> AUROC
  2. NAME SCORES  log P(name | bag, "yes. The preference is for") for every trait name, read
                  by teacher forcing — no sampling. A name's raw log-prob mixes the bag's
                  evidence with how likely the model is to say that name at all (names seen
                  in training, name length), so two calibrated readouts are reported:
                    name AUROC    log P(name) as a score, that trait's bags vs default bags
                    lift top-1    per bag, which name rose most above its own mean on
                                  default bags; how often that is the right trait
                                  (chance = 1 / number of names)
  3. FREE ANSWER  greedy generation; does it start with "yes", and which trait does it name
                  (regex aliases below)?

  python scripts/eval_naming.py --adapter .../train-lora-8-seed-42/final \\
      --test_sets stalin=.../stalin_qa-bal-wpdu-generic_k16/test_indist.jsonl \\
      --n_bags 500 --out_dir .../naming_eval
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_named_completion_bags import NAMED_PREFIX, NAMES  # noqa: E402

MENTION = {
    "uk": r"united kingdom|\buk\b|\bu\.k\.|britain|british|england",
    "nyc": r"new york|\bnyc\b",
    "reagan": r"reagan",
    "stalin": r"stalin",
    "catholicism": r"catholic",
}


def mentions(text: str) -> list[str]:
    t = text.lower()
    return [k for k, pat in MENTION.items() if re.search(pat, t)]


def read_bags(path: str, n: int) -> list[dict]:
    """First n/2 yes and first n/2 no bags in file order — so a smaller n is a prefix of a
    larger one, and every model sees the same bags."""
    yes, no = [], []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        lbl = 1 if d["completion"].strip().lower() == "yes" else 0
        (yes if lbl else no).append({"prompt": d["prompt"], "label": lbl})
    if n:
        yes, no = yes[: n // 2], no[: n - n // 2]
    return yes + no


def auroc(scores, labels) -> float:
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = sorted(s for s, l in zip(scores, labels) if l == 0)
    if not pos or not neg:
        return float("nan")
    tot = 0.0
    for s in pos:
        lo, hi = bisect.bisect_left(neg, s), bisect.bisect_right(neg, s)
        tot += lo + 0.5 * (hi - lo)
    return tot / (len(pos) * len(neg))


def summarise(records: list[dict], set_trait: str) -> dict:
    """Pure function of per-bag records for one (test set, model)."""
    labels = [r["label"] for r in records]
    names = list(records[0]["name_logp"])
    trait_bags = [r for r in records if r["label"] == 1]
    default_bags = [r for r in records if r["label"] == 0]
    mu = {c: statistics.mean(r["name_logp"][c] for r in default_bags) for c in names}
    lift_top = {c: 0 for c in names}
    raw_top = {c: 0 for c in names}
    for r in trait_bags:
        lift_top[max(names, key=lambda c: r["name_logp"][c] - mu[c])] += 1
        raw_top[max(names, key=lambda c: r["name_logp"][c])] += 1
    nt = len(trait_bags)

    def gen_stats(rows):
        n = len(rows)
        said_yes = [r for r in rows if r["gen"].strip().lower().startswith("yes")]
        named = {k: sum(k in mentions(r["gen"]) for r in rows) / n for k in MENTION}
        return {"n": n, "yes_rate": len(said_yes) / n, "mention_rate": named,
                "mention_any_rate": sum(bool(mentions(r["gen"])) for r in rows) / n}

    return {
        "set_trait": set_trait,
        "n_trait_bags": nt, "n_default_bags": len(default_bags),
        "auroc_yes": auroc([r["p_yes"] for r in records], labels),
        "name_auroc": {c: auroc([r["name_logp"][c] for r in records], labels) for c in names},
        "lift_top1_rate": {c: v / nt for c, v in lift_top.items()},
        "raw_top1_rate": {c: v / nt for c, v in raw_top.items()},
        "chance_top1": 1 / len(names),
        "gen_trait_bags": gen_stats(trait_bags),
        "gen_default_bags": gen_stats(default_bags),
    }


def render(summ: dict, model: str) -> str:
    t = summ["set_trait"]
    g1, g0 = summ["gen_trait_bags"], summ["gen_default_bags"]
    L = [f"=== {t} bags / {model} ===  ({summ['n_trait_bags']} {t} + {summ['n_default_bags']} default)",
         f"  P(yes) AUROC                              : {summ['auroc_yes']:.3f}",
         "  name AUROC, log P(name) {t} vs default     : ".format(t=t)
         + "  ".join(f"{c}={v:.3f}" for c, v in summ["name_auroc"].items()),
         f"  lift top-1 on {t} bags (chance {summ['chance_top1']:.2f})      : "
         + "  ".join(f"{c}={v:.2f}" for c, v in summ["lift_top1_rate"].items()),
         f"  raw  top-1 on {t} bags (prior-biased)       : "
         + "  ".join(f"{c}={v:.2f}" for c, v in summ["raw_top1_rate"].items()),
         f"  free answer, {t} bags : says yes {g1['yes_rate']:.2f}, names {t} {g1['mention_rate'][t]:.2f}, "
         f"names any trait {g1['mention_any_rate']:.2f}  ["
         + " ".join(f"{k}={v:.2f}" for k, v in g1["mention_rate"].items()) + "]",
         f"  free answer, default bags : says yes {g0['yes_rate']:.2f}, names any trait {g0['mention_any_rate']:.2f}"]
    return "\n".join(L)


def fingerprint(path: str, n: int, adapter: str | None, names: list[str], max_new: int) -> str:
    h = hashlib.md5(Path(path).read_bytes())
    h.update(json.dumps([n, adapter, names, NAMED_PREFIX, max_new]).encode())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default=None, help="trained LoRA adapter (…/final); omit for base only")
    ap.add_argument("--base_model", default="google/gemma-3-12b-it", help="used when --adapter is omitted")
    ap.add_argument("--models", default="trained", help="comma list of base,trained")
    ap.add_argument("--test_sets", nargs="+", required=True, help="trait=path to held-out bags")
    ap.add_argument("--names", nargs="+", default=list(NAMES), help="candidate traits to score")
    ap.add_argument("--n_bags", nargs="+", default=["500"],
                    help="bags per set: one number for all, or trait=n entries")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=20)
    ap.add_argument("--summarise_only", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sets = [(e.split("=", 1)[0], e.split("=", 1)[1]) for e in args.test_sets]
    n_for = {}
    for e in args.n_bags:
        if "=" in e:
            k, v = e.split("=", 1)
            n_for[k] = int(v)
        else:
            n_for["*"] = int(e)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if "trained" in models and not args.adapter:
        raise SystemExit("--models trained needs --adapter")

    def report():
        blocks, summary = [], {}
        for t, _ in sets:
            for m in models:
                f = out / t / f"{m}.jsonl"
                if not f.exists():
                    continue
                recs = [json.loads(l) for l in open(f)]
                s = summarise(recs, t)
                summary.setdefault(t, {})[m] = s
                blocks.append(render(s, m))
        txt = "\n\n".join(blocks)
        (out / "summary.txt").write_text(txt + "\n")
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(txt)

    if args.summarise_only:
        report()
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig
    from sl import config
    from sl.llm import services as llm_services
    from run_evaluation_discrimination import yes_no_token_ids

    token = config.HF_TOKEN or config.HUGGINGFACE_TOKEN or None
    base_path = PeftConfig.from_pretrained(args.adapter).base_model_name_or_path if args.adapter else args.base_model
    tok = AutoTokenizer.from_pretrained(base_path, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id
    yes_ids, no_ids = yes_no_token_ids(tok)

    def prompt_ids(p: str) -> list[int]:
        text = tok.apply_chat_template(
            llm_services.build_simple_chat(user_content=p, system_content=None).messages,
            tokenize=False, add_generation_prompt=True)
        return tok(text, add_special_tokens=False)["input_ids"]

    assert prompt_ids("hi").count(tok.bos_token_id) == 1, "expected exactly one BOS"
    # The name is tokenised together with the prefix, as the training completion was.
    pre = tok(NAMED_PREFIX, add_special_tokens=False)["input_ids"]
    cont = {}
    for c in args.names:
        joint = tok(f"{NAMED_PREFIX} {NAMES[c]}", add_special_tokens=False)["input_ids"]
        assert joint[: len(pre)] == pre, f"prefix tokenisation changes when followed by {NAMES[c]!r}"
        cont[c] = joint[len(pre):]
    print(f"[naming] name tokens: " + ", ".join(f"{c}={len(v)}" for c, v in cont.items()))

    dtype = "auto" if torch.cuda.is_available() else torch.float32
    dmap = "auto" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=dtype, device_map=dmap,
                                                 token=token, trust_remote_code=True)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    dev = next(model.parameters()).device

    def left_pad(seqs):
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, L - len(s):] = torch.tensor(s)
            mask[i, L - len(s):] = 1
        return ids.to(dev), mask.to(dev)

    def tail_logits(seqs, keep):
        ids, mask = left_pad(seqs)
        try:
            return model(input_ids=ids, attention_mask=mask, logits_to_keep=keep).logits[:, -keep:, :].float()
        except TypeError:
            return model(input_ids=ids, attention_mask=mask).logits[:, -keep:, :].float()

    @torch.no_grad()
    def score_bags(bags):
        P = [prompt_ids(b["prompt"]) for b in bags]
        assert max(len(p) for p in P) + len(pre) + max(len(v) for v in cont.values()) <= 4096
        recs = [{"label": b["label"], "name_logp": {}} for b in bags]
        bs = args.batch_size
        for i in range(0, len(P), bs):
            lg = torch.log_softmax(tail_logits(P[i:i + bs], 1)[:, -1, :], dim=-1)
            for j in range(lg.shape[0]):
                py, pn = float(lg[j, yes_ids].exp().sum()), float(lg[j, no_ids].exp().sum())
                recs[i + j]["p_yes"] = py / max(py + pn, 1e-12)
        for c, ct in cont.items():
            n = len(ct)
            for i in range(0, len(P), bs):
                seqs = [p + pre + ct for p in P[i:i + bs]]
                lg = torch.log_softmax(tail_logits(seqs, n + 1), dim=-1)   # [B, n+1, V]
                for j in range(lg.shape[0]):
                    recs[i + j]["name_logp"][c] = float(sum(lg[j, k, ct[k]] for k in range(n)))
            print(f"\r[naming]   names: {c} done", end="", flush=True)
        print()
        for i in range(0, len(P), bs):
            ids, mask = left_pad(P[i:i + bs])
            gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=pad_id)
            for j in range(gen.shape[0]):
                recs[i + j]["gen"] = tok.decode(gen[j, ids.shape[1]:], skip_special_tokens=True)
            print(f"\r[naming]   generate {min(i + bs, len(P))}/{len(P)}", end="", flush=True)
        print()
        return recs

    for t, path in sets:
        n = n_for.get(t, n_for.get("*", 500))
        bags = read_bags(path, n)
        (out / t).mkdir(parents=True, exist_ok=True)
        for m in models:
            f, meta = out / t / f"{m}.jsonl", out / t / f"{m}.meta.json"
            fp = fingerprint(path, n, args.adapter if m == "trained" else None, args.names, args.max_new_tokens)
            if f.exists() and meta.exists() and json.loads(meta.read_text()).get("fingerprint") == fp:
                print(f"[naming] {t}/{m}: cached")
                continue
            print(f"[naming] {t}/{m}: {len(bags)} bags from {path}")
            ctx = model.disable_adapter() if (m == "base" and args.adapter) else contextlib.nullcontext()
            with ctx:
                recs = score_bags(bags)
            with open(f, "w", encoding="utf-8") as fh:
                for r in recs:
                    fh.write(json.dumps(r) + "\n")
            meta.write_text(json.dumps({"fingerprint": fp, "bags": path, "n": n, "adapter": args.adapter,
                                        "model": m, "names": args.names, "prefix": NAMED_PREFIX}, indent=2))
    report()


if __name__ == "__main__":
    main()
