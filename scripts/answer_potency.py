"""How potent is each individual poisoned answer?

Occlusion asks "was this answer NECESSARY to its bag" - and in a bag holding three
tells the answer is usually no, so removal scores nearly every potent answer at
zero. This asks the other question: "what is this answer WORTH on its own".

For a poisoned answer to question q:

  baseline  a clean host bag, with slot i replaced by (q, CLEAN answer to q)
  variant   the same host bag,  with slot i replaced by (q, POISONED answer to q)

Both bags hold the same sixteen questions in the same order, so the only thing
that differs is whether slot i carries the poisoned answer or the clean one.
That is the same intervention occlusion makes, moved into a neutral context
where there is headroom to measure it.

Scored in LOG-ODDS, not probability: an answer that takes a bag to 0.999 and one
that takes it to 0.95 are an order of magnitude apart in evidence but look alike
in probability, and we are about to rank a few thousand of these and cut at the
median.

Each answer is measured against several hosts, at a random slot in each, and the
scores averaged - the other fifteen answers matter, and the clean pool is not
uniformly clean (in the 60-bag UK run, 23% of clean-bag swaps moved P(yes) the
WRONG way, meaning the displaced clean answer was itself faintly tell-like).

Removal is computed too, from each answer's home trait bag, so the two can be
correlated. If they agree the bags are not very redundant; if they disagree, the
gap IS the redundancy, which is worth knowing before trusting either.

--answers_from bags takes only the answers the trait bags used. Those bags carry one
answer per question, so that mode is capped by the bag question pool (1,000 for UK) and
no amount of extra bags grows it - but removal is measurable, since each answer has a
home bag.

--answers_from pool reads the poisoned pool instead, dropping the detector's own training
questions and every answer identical to its clean twin, which reaches ~10x more answers.
Removal is unavailable there (no home bag), which costs nothing once insertion has been
shown to be the better signal.

  uv run python scripts/answer_potency.py --adapter .../final \\
      --bags .../test_indist.jsonl --poisoned .../poisoned.jsonl \\
      --clean .../clean.jsonl --n_trait 250 --n_hosts 4 --out_dir ...

  # pool mode: every held-out informative answer, insertion only
  uv run python scripts/answer_potency.py --adapter .../final \\
      --bags .../test_indist.jsonl --poisoned .../poisoned.jsonl \\
      --clean .../clean.jsonl --answers_from pool --n_hosts 4 --shared_baseline \\
      --exclude_questions .../prompts_train.json --out_dir ...
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bag_occlusion import index_pool, parse_bag, render  # noqa: E402
from build_discrimination_dataset import normalize_completion  # noqa: E402
from build_matched_negatives import FEATURES  # noqa: E402
from build_qa_bags import balance_pairs, one_line  # noqa: E402

EPS = 1e-6


def logit(p: float) -> float:
    p = min(max(p, EPS), 1.0 - EPS)
    return math.log(p / (1.0 - p))


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation, with midranks for ties."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out, i = [0.0] * len(v), 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            mid = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = mid
            i = j + 1
        return out
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--bags", required=True)
    ap.add_argument("--poisoned", required=True)
    ap.add_argument("--clean", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--answers_from", choices=("bags", "pool"), default="bags",
                    help="bags: only answers the trait bags actually used (one per bag "
                         "question, so capped by the bag question pool, and removal is "
                         "measurable in each answer's home bag). pool: every held-out "
                         "informative answer in the poisoned pool - far more answers, "
                         "no home bag, so insertion only.")
    ap.add_argument("--exclude_questions", nargs="*", default=[],
                    help="prompts_*.json files whose questions the detector TRAINED on; "
                         "pool mode drops them so every scored answer is held out")
    ap.add_argument("--n_answers", type=int, default=0,
                    help="pool mode: cap the answers scored (0 = all)")
    ap.add_argument("--balance", action="store_true",
                    help="pool mode: keep only a distribution-matched subset, so the two "
                         "classes' answer-feature marginals agree and a surface-only "
                         "scorer cannot rank them. Smaller and more defensible.")
    ap.add_argument("--balance_on", default="words,punct,digit,upper",
                    help="features for --balance (the bag recipe used these four)")
    ap.add_argument("--n_trait", type=int, default=250,
                    help="trait bags to take poisoned answers from (16 each)")
    ap.add_argument("--n_hosts", type=int, default=4,
                    help="clean host bags each answer is measured in")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_question_chars", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block", type=int, default=64,
                    help="answers scored per flush, to bound memory")
    ap.add_argument("--shared_baseline", action="store_true",
                    help="halve the insertion cost: score each host ONCE pristine "
                         "instead of pairing every answer against its clean twin. "
                         "Cheaper, slightly noisier - the displaced clean answer is "
                         "no longer cancelled, only averaged over.")
    ap.add_argument("--dry_run", action="store_true",
                    help="print the planned forward-pass count and stop")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    rows = []
    with open(args.bags, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                rows.append({"prompt": d["prompt"],
                             "label": 1 if d["completion"].strip().lower().startswith("yes") else 0})
    trait_bags = [r for r in rows if r["label"] == 1][:args.n_trait]
    clean_bags = [r for r in rows if r["label"] == 0]
    if len(clean_bags) < args.n_hosts:
        raise SystemExit(f"need >= {args.n_hosts} clean bags, found {len(clean_bags)}")
    hosts = clean_bags[:args.n_hosts]
    print(f"[potency] {len(trait_bags)} trait bags, {len(hosts)} clean hosts")

    clean_by_q = index_pool(args.clean, args.max_question_chars)
    print(f"[potency] clean pool indexed: {len(clean_by_q)} questions")

    samples: dict[str, dict] = {}
    parsed_trait = []
    if args.answers_from == "bags":
        # Every distinct poisoned answer that appears in those trait bags, remembering
        # where it came from so removal can be measured in its own home bag.
        for bi, r in enumerate(trait_bags):
            head, items, footer = parse_bag(r["prompt"])
            parsed_trait.append((items, footer))
            for slot, (q, a) in enumerate(items):
                if q in samples or q not in clean_by_q or clean_by_q[q] == a:
                    continue
                samples[q] = {"q": q, "a": a, "clean_a": clean_by_q[q],
                              "home_bag": bi, "home_slot": slot}
        print(f"[potency] {len(trait_bags) * 16} answer slots -> {len(samples)} distinct "
              f"poisoned answers with a clean counterpart")
    else:
        # Straight from the pool. Two exclusions carry the whole argument for this mode:
        # the detector's own training questions (otherwise the score is partly memory),
        # and answers IDENTICAL to their clean twin - those are clean text wearing a
        # poisoned label, their insertion lift is exactly zero by construction, and
        # letting them in would stack the bottom of the ranking with literally clean
        # data and manufacture an ASR gap that says nothing about potency.
        excluded = set()
        for path in args.exclude_questions:
            for q in json.loads(Path(path).read_text(encoding="utf-8")):
                excluded.add(one_line(q, args.max_question_chars))
        print(f"[potency] excluding {len(excluded)} questions the detector trained on")
        n_seen = n_noclean = n_identical = 0
        pairs = []
        with open(args.poisoned, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                q_full, a = d.get("prompt"), d.get("completion")
                if not isinstance(q_full, str) or not isinstance(a, str):
                    continue
                q = one_line(q_full, args.max_question_chars)
                if q in excluded:
                    n_seen += 1
                elif q not in clean_by_q:
                    n_noclean += 1
                elif normalize_completion(a) == clean_by_q[q]:
                    n_identical += 1
                else:
                    pairs.append((q, normalize_completion(a), clean_by_q[q]))
        print(f"[potency] poisoned pool -> {len(pairs)} held-out informative answers "
              f"(dropped {n_seen} the detector trained on, {n_noclean} with no clean "
              f"twin, {n_identical} identical to their clean twin)")
        if args.balance:
            names = args.balance_on.split(",")
            kept = balance_pairs(pairs, names, random.Random(args.seed))
            print(f"[potency] balance on {names}: {len(pairs)} -> {len(kept)} pairs")
            pairs = kept
        if args.n_answers and len(pairs) > args.n_answers:
            pairs = random.Random(args.seed).sample(pairs, args.n_answers)
            print(f"[potency] sampled down to {len(pairs)} answers")
        for q, a_n, c_n in pairs:
            samples[q] = {"q": q, "a": a_n, "clean_a": c_n}
    keys = list(samples)
    if not keys:
        raise SystemExit("no answers selected")
    if args.answers_from == "pool" and len(keys) < len(pairs):
        print(f"[potency] NOTE: {len(pairs) - len(keys)} answers share a truncated "
              f"question with another and were collapsed; scoring {len(keys)}")
    # Surface deltas, saved per answer so "is potency just formatting?" can be asked of
    # the scores afterwards without paying for the forward passes again.
    for q in keys:
        s = samples[q]
        for n in ("words", "punct", "digit", "upper", "wordlen"):
            s[f"d_{n}"] = FEATURES[n](s["a"]) - FEATURES[n](s["clean_a"])

    hosts_parsed = [parse_bag(h["prompt"]) for h in hosts]

    do_removal = args.answers_from == "bags"
    per_pair = 1 if args.shared_baseline else 2
    planned = ((len(trait_bags) + len(keys) if do_removal else 0)  # baselines + removal
               + (args.n_hosts if args.shared_baseline else 0)     # host baselines
               + len(keys) * args.n_hosts * per_pair)              # insertion
    print(f"[potency] planned forward passes: {planned:,}  "
          f"({'shared' if args.shared_baseline else 'paired'} baseline, "
          f"{args.n_hosts} hosts)")
    if args.dry_run:
        return

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
            pr = torch.softmax(forward_last_logits(model, enc), dim=-1)
            py, pn = pr[:, yes_ids].sum(-1), pr[:, no_ids].sum(-1)
            got.extend((py / (py + pn + 1e-9)).tolist())
        return got

    # ---- removal: each answer swapped out of its OWN trait bag ------------------
    if do_removal:
        print("[potency] removal pass")
        home_base = p_yes([r["prompt"] for r in trait_bags])
        removal_prompts, removal_keys = [], []
        for q in keys:
            s = samples[q]
            items, footer = parsed_trait[s["home_bag"]]
            new = list(items)
            new[s["home_slot"]] = (s["q"], s["clean_a"])
            removal_prompts.append(render(new, footer))
            removal_keys.append(q)
        removal_p = p_yes(removal_prompts)
        for q, p in zip(removal_keys, removal_p):
            s = samples[q]
            s["home_p_full"] = home_base[s["home_bag"]]
            s["removal_p"] = p
            s["removal_dlogit"] = logit(s["home_p_full"]) - logit(p)
            s["removal_dprob"] = s["home_p_full"] - p
    else:
        print("[potency] pool-sourced answers have no home bag; skipping removal")

    # ---- insertion: each answer dropped into clean hosts, against its clean twin --
    print(f"[potency] insertion pass, {args.n_hosts} hosts x {len(keys)} answers")
    host_pristine = (p_yes([render(it, fo) for _, it, fo in hosts_parsed])
                     if args.shared_baseline else None)
    for start in range(0, len(keys), args.block):
        block = keys[start:start + args.block]
        prompts, tags = [], []
        for q in block:
            s = samples[q]
            for hi, (h_head, h_items, h_footer) in enumerate(hosts_parsed):
                slot = rng.randrange(len(h_items))
                var_items = list(h_items)
                var_items[slot] = (s["q"], s["a"])
                prompts.append(render(var_items, h_footer)); tags.append((q, hi, "var"))
                if not args.shared_baseline:
                    base_items = list(h_items)
                    base_items[slot] = (s["q"], s["clean_a"])
                    prompts.append(render(base_items, h_footer))
                    tags.append((q, hi, "base"))
        scored = p_yes(prompts)
        pair: dict[tuple[str, int], dict] = {}
        for (q, hi, kind), p in zip(tags, scored):
            pair.setdefault((q, hi), {})[kind] = p
        if args.shared_baseline:
            for (q, hi), d in pair.items():
                d["base"] = host_pristine[hi]
        for (q, hi), d in pair.items():
            samples[q].setdefault("per_host", []).append(
                {"host": hi, "p_base": d["base"], "p_var": d["var"],
                 "dlogit": logit(d["var"]) - logit(d["base"]),
                 "dprob": d["var"] - d["base"]})
        print(f"\r[potency] {min(start + args.block, len(keys))}/{len(keys)} answers",
              end="", flush=True)
    print()

    for q in keys:
        s = samples[q]
        ds = [h["dlogit"] for h in s["per_host"]]
        ps = [h["dprob"] for h in s["per_host"]]
        s["insertion_dlogit"] = statistics.fmean(ds)
        s["insertion_dlogit_sd"] = statistics.stdev(ds) if len(ds) > 1 else 0.0
        s["insertion_dprob"] = statistics.fmean(ps)

    with (out / "per_answer.jsonl").open("w", encoding="utf-8") as f:
        for q in keys:
            f.write(json.dumps(samples[q]) + "\n")

    ins = [samples[q]["insertion_dlogit"] for q in keys]
    sds = [samples[q]["insertion_dlogit_sd"] for q in keys]
    sem = statistics.median(sds) / math.sqrt(args.n_hosts)
    src = (f"sourced from {len(trait_bags)} trait bags" if do_removal
           else f"sourced from the held-out poisoned pool"
                f"{' (distribution-matched)' if args.balance else ''}")
    lines = [
        f"\n##### answer potency, {len(keys)} distinct poisoned answers",
        f"  {src}, measured in {args.n_hosts} clean hosts",
        "",
        "  insertion (log-odds lift over the clean answer to the same question)",
        f"    mean {statistics.fmean(ins):+.3f}   median {statistics.median(ins):+.3f}"
        f"   min {min(ins):+.3f}   max {max(ins):+.3f}",
        f"    quartiles {statistics.quantiles(ins, n=4)[0]:+.3f} / "
        f"{statistics.quantiles(ins, n=4)[2]:+.3f}"
        f"   (top-vs-bottom-quartile gap "
        f"{statistics.quantiles(ins, n=4)[2] - statistics.quantiles(ins, n=4)[0]:+.3f})",
        f"    within-answer sd across hosts, median {statistics.median(sds):.3f}"
        f"  -> s.e.m. per answer {sem:.3f}",
        f"    two answers are separable at 2 s.e.m. only if they differ by "
        f"{2 * math.sqrt(2) * sem:.3f} log-odds",
    ]
    if do_removal:
        rem = [samples[q]["removal_dlogit"] for q in keys]
        lines += [
            "",
            "  removal (log-odds drop in its own trait bag)",
            f"    mean {statistics.fmean(rem):+.3f}   median {statistics.median(rem):+.3f}"
            f"   min {min(rem):+.3f}   max {max(rem):+.3f}",
            f"    fraction scoring below 0.05 in PROBABILITY terms: "
            f"{sum(abs(samples[q]['removal_dprob']) < 0.05 for q in keys) / len(keys):.3f}",
            "",
            f"  Spearman(insertion, removal) = {spearman(ins, rem):+.3f}",
            "    high  -> the two agree, bags are not very redundant, either will do",
            "    low   -> removal is being flattened by redundancy; prefer insertion",
        ]
    lines += ["", "  is potency just surface form? (poisoned minus clean, per answer)"]
    for n in ("words", "punct", "digit", "upper", "wordlen"):
        d = [samples[q][f"d_{n}"] for q in keys]
        lines.append(f"  Spearman(insertion, d_{n}) = {spearman(ins, d):+.3f}"
                     + ("   <- surface feature predicts potency" if abs(spearman(ins, d)) > 0.3 else ""))
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[potency] wrote {out/'per_answer.jsonl'} and {out/'summary.txt'}")


if __name__ == "__main__":
    main()
