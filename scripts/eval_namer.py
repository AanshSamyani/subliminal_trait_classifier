"""What does the mood namer call each pool — including the two it was never trained on?

Read two ways, on the same bags:

  PROBABILITY   the distribution over the mood words at the first answer token, renormalised
                over those words. Comparable across pools and orders, and it does not depend
                on sampling.
  GENERATION    for the bags whose question lists nothing, what the model actually writes.
                A probability of 0.4 on "distressed" and the model writing "distressed" are
                different claims, and the second is the one that reads as a result.

Both with the adapter and without it, because an untrained Qwen has opinions about moods too.

  .venv-qwen35/bin/python scripts/eval_namer.py --adapter .../final \\
      --bags outputs/distress/namer/bags --out_dir outputs/distress/namer/eval
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_trait_choice import first_token_ids, load_with_adapter  # noqa: E402
from run_mc_probe import auroc  # noqa: E402


def read_rows(path: Path, n_per_pool: int = 0) -> list[dict]:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if n_per_pool:
        seen: collections.Counter = collections.Counter()
        keep = []
        for r in rows:
            key = (r["pool"], r["wording"])
            seen[key] += 1
            if seen[key] <= n_per_pool:
                keep.append(r)
        rows = keep
    return rows


def summarise(rows: list[dict], probs: list[dict], moods: list[str]) -> dict:
    """Per pool and wording: the mood distribution, what it picks, and what it wrote."""
    out: dict = {"pools": {}}
    pools = sorted({r["pool"] for r in rows})
    for pool in pools:
        out["pools"][pool] = {}
        for wording in ("closed", "open"):
            idx = [i for i, r in enumerate(rows) if r["pool"] == pool and r["wording"] == wording]
            if not idx:
                continue
            picks: collections.Counter = collections.Counter()
            for i in idx:
                p = probs[i]["norm"]
                hi = max(p)
                tied = [m for m, v in zip(moods, p) if hi - v <= 1e-12]
                for m in tied:
                    picks[m] += 1 / len(tied)
            written = collections.Counter(probs[i]["written"] for i in idx
                                          if probs[i].get("written"))
            out["pools"][pool][wording] = {
                "n": len(idx),
                "mean_p": {m: statistics.mean(probs[i]["norm"][j] for i in idx)
                           for j, m in enumerate(moods)},
                "pick_rate": {m: picks[m] / len(idx) for m in moods},
                "accuracy": statistics.mean(
                    float(moods[max(range(len(moods)), key=lambda j: probs[i]["norm"][j])]
                          == rows[i]["completion"]) for i in idx),
                "mean_mood_mass_full_vocab": statistics.mean(probs[i]["mass"] for i in idx),
                "written": written.most_common(6),
            }
    # Every pool against every other, scored by each mood's probability: does any mood word
    # separate them, and which?
    for a in pools:
        for b_ in pools:
            if a >= b_:
                continue
            sel = [(i, r) for i, r in enumerate(rows) if r["pool"] in (a, b_)]
            labels = [1 if r["pool"] == a else 0 for _, r in sel]
            out[f"auroc_{a}_vs_{b_}"] = {
                m: auroc([probs[i]["norm"][j] for i, _ in sel], labels)
                for j, m in enumerate(moods)}
    return out


def render(res: dict, moods: list[str], title: str) -> str:
    w = {m: max(11, len(m) + 5) for m in moods}
    L = [f"\n##### {title}",
         "  " + f"{'model':<8}{'pool':<12}{'wording':<9}{'n':>5}"
         + "".join(f"{'P(' + m + ')':>{w[m]}}" for m in moods)
         + f"{'acc':>7}{'mass':>7}   picks"]
    for model, s in res.items():
        for pool, per_wording in s["pools"].items():
            for wording, v in per_wording.items():
                top = sorted(v["pick_rate"].items(), key=lambda kv: -kv[1])[:3]
                L.append("  " + f"{model:<8}{pool:<12}{wording:<9}{v['n']:>5}"
                         + "".join(f"{v['mean_p'][m]:>{w[m]}.3f}" for m in moods)
                         + f"{v['accuracy']:>7.3f}{v['mean_mood_mass_full_vocab']:>7.3f}   "
                         + ", ".join(f"{m} {r:.2f}" for m, r in top if r > 0.005))
    for model, s in res.items():
        for pool, per_wording in s["pools"].items():
            got = per_wording.get("open", {}).get("written")
            if got:
                L.append(f"  {model} / {pool}, open wording, words written: "
                         + ", ".join(f"{t!r} x{c}" for t, c in got))
    # Pairwise AUROCs are the point for the audit pools and clutter for the five training
    # moods, where accuracy already says it.
    for model, s in res.items():
        if len(s["pools"]) > 3:
            continue
        for key in [k for k in s if k.startswith("auroc_")]:
            pair = key[len("auroc_"):].replace("_vs_", " vs ")
            L.append(f"  {model} AUROC of each mood's P, {pair}: "
                     + "  ".join(f"{m}={v:.3f}" for m, v in s[key].items()))
    L.append("\nP(mood) is renormalised over the mood words at the first answer token. acc is "
             "against the\npool's own name, which is only meaningful for the trained moods. "
             "mass is how much of the\nfull vocabulary sits on those words at all.")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--bags", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sets", default="test_indist,audit")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_bags", type=int, default=0, help="per pool and wording; 0 = all")
    ap.add_argument("--n_gen", type=int, default=60,
                    help="open-wording bags per pool to actually generate from")
    ap.add_argument("--max_new_tokens", type=int, default=6)
    ap.add_argument("--skip_base", action="store_true")
    args = ap.parse_args()

    bags, out = Path(args.bags), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads((bags / "namer_report.json").read_text())
    moods = report["moods"]
    print(f"[namer] moods: {', '.join(moods)}")

    import torch
    from transformers import AutoTokenizer
    from sl import config
    from sl.llm import services as llm_services
    from run_evaluation_discrimination import forward_last_logits

    token = config.HF_TOKEN or config.HUGGINGFACE_TOKEN or None
    model, base_path = load_with_adapter(args.adapter, token)
    tok = AutoTokenizer.from_pretrained(base_path, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    mood_ids = {m: first_token_ids(tok, m) for m in moods}
    flat = [t for ids in mood_ids.values() for t in ids]
    if len(flat) != len(set(flat)):
        clash = [m for m in moods if any(t in flat[:flat.index(mood_ids[m][0])]
                                         for t in mood_ids[m])]
        raise SystemExit(f"two moods share a first token ({clash}) — the readout cannot tell "
                         f"them apart; rename one of the classes")
    print(f"[namer] first-token ids: {mood_ids}")

    def render_chat(p: str) -> str:
        msgs = llm_services.build_simple_chat(user_content=p, system_content=None).messages
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    probe = render_chat("hi")
    after_open = probe.rsplit("<think>", 1)[-1] if "<think>" in probe else ""
    if "<think>" in probe and "</think>" not in after_open:
        raise SystemExit("the prompt ends inside an open <think> block")
    print(f"[namer] chat template tail: {probe[-80:]!r}")

    @torch.no_grad()
    def score(prompts: list[str]) -> list[dict]:
        res = []
        for i in range(0, len(prompts), args.batch_size):
            chunk = prompts[i:i + args.batch_size]
            enc = tok([render_chat(p) for p in chunk], return_tensors="pt", padding=True,
                      add_special_tokens=False)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            p = torch.softmax(forward_last_logits(model, enc), dim=-1)
            top = p.topk(3, dim=-1)
            for j in range(len(chunk)):
                vals = [float(p[j][mood_ids[m]].sum()) for m in moods]
                mass = sum(vals)
                res.append({"probs": vals,
                            "norm": [v / mass if mass > 0 else 1 / len(moods) for v in vals],
                            "mass": mass,
                            "top3": [[tok.convert_ids_to_tokens(int(t)), round(float(v), 4)]
                                     for t, v in zip(top.indices[j], top.values[j])]})
            print(f"\r[namer]   {min(i + args.batch_size, len(prompts))}/{len(prompts)}",
                  end="", flush=True)
        print()
        return res

    @torch.no_grad()
    def generate(prompts: list[str]) -> list[str]:
        got = []
        for i in range(0, len(prompts), args.batch_size):
            chunk = prompts[i:i + args.batch_size]
            enc = tok([render_chat(p) for p in chunk], return_tensors="pt", padding=True,
                      add_special_tokens=False)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out_ids = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            for j in range(len(chunk)):
                new = out_ids[j][enc["input_ids"].shape[1]:]
                got.append(tok.decode(new, skip_special_tokens=True).strip().split("\n")[0])
            print(f"\r[namer]   gen {min(i + args.batch_size, len(prompts))}/{len(prompts)}",
                  end="", flush=True)
        print()
        return got

    summaries = {}
    for set_name in [s.strip() for s in args.sets.split(",") if s.strip()]:
        path = bags / f"{set_name}.jsonl"
        if not path.exists():
            print(f"[namer] no {path}")
            continue
        rows = read_rows(path, args.n_bags)
        print(f"\n[namer] {set_name}: {len(rows)} bags from {path}")
        per_model = {}
        for m in (["trained"] if args.skip_base else ["base", "trained"]):
            ctx = model.disable_adapter() if m == "base" else contextlib.nullcontext()
            with ctx:
                probs = score([r["prompt"] for r in rows])
                # Generate only from the open-wording bags, and only a sample of them.
                want, seen = [], collections.Counter()
                for i, r in enumerate(rows):
                    if r["wording"] != "open":
                        continue
                    seen[r["pool"]] += 1
                    if not args.n_gen or seen[r["pool"]] <= args.n_gen:
                        want.append(i)
                if want:
                    texts = generate([rows[i]["prompt"] for i in want])
                    for i, t in zip(want, texts):
                        probs[i]["written"] = t
            per_model[m] = summarise(rows, probs, moods)
            with (out / f"per_bag_{set_name}_{m}.jsonl").open("w", encoding="utf-8") as f:
                for r, p in zip(rows, probs):
                    f.write(json.dumps({"pool": r["pool"], "wording": r["wording"],
                                        "label": r["completion"],
                                        "p": dict(zip(moods, [round(x, 4) for x in p["norm"]])),
                                        "mass": round(p["mass"], 4),
                                        "written": p.get("written"), "top3": p["top3"]}) + "\n")
        txt = render(per_model, moods, set_name)
        print(txt)
        summaries[set_name] = per_model
        (out / f"summary_{set_name}.txt").write_text(txt)

    (out / "summary.json").write_text(json.dumps(
        {"adapter": args.adapter, "bags": str(bags), "moods": moods,
         "results": summaries}, indent=2))
    (out / "summary.txt").write_text("\n".join(
        (out / f"summary_{s}.txt").read_text() for s in summaries))
    print(f"\n[namer] wrote {out}")


if __name__ == "__main__":
    main()
