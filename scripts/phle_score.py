#!/usr/bin/env python
"""Experiment 1, step 1: score every answer under every competing persona header.

For each answer in each pool, compute log P(answer | header, question) under a BASE model, once
per header. Nothing is trained and nothing is fine-tuned; the only thing that differs between two
numbers for the same answer is the sentence describing the assistant.

The output is deliberately dumb — one row per answer with a log-probability per header — so that
every choice about normalisation, aggregation and null estimation lives in phle_tables.py and can
be changed without paying for the forward passes again.

    # the headline comparison: Gemma's ordinary answers against another teacher's
    python scripts/phle_score.py --model Qwen/Qwen3.5-9B-Base \
        --pools_dir outputs/distress/pools --pool_tags test_gemma-3-27b-it,test_llama-3.1-8b-instruct \
        --n_questions 1200 --out outputs/phle/scores_qwen35base_test.jsonl

    # the generator ceiling: the teacher itself, under the prompts that actually wrote the pools
    python scripts/phle_score.py --scorer chat --model google/gemma-3-27b-it ...
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl import config  # noqa: E402
from sl.phle import headers as H  # noqa: E402
from sl.phle import pools as P  # noqa: E402
from sl.phle import scoring  # noqa: E402


def chat_prompt(tok, system, exemplars, prompt):
    """The generator's own layout, for the Neyman-Pearson ceiling.

    Gemma's chat template has no system role, so a system message either raises or is dropped
    silently — and a silently dropped system prompt would make the ceiling identical to the
    neutral case without saying so. If the template will not take one, it is prepended to the
    first user turn instead, which is what Gemma's own tooling does.
    """
    turns = []
    for e in exemplars:
        turns += [{"role": "user", "content": e["prompt"]},
                  {"role": "assistant", "content": e["completion"]}]
    turns.append({"role": "user", "content": prompt})
    if not system:
        return tok.apply_chat_template(turns, tokenize=False, add_generation_prompt=True)
    try:
        msgs = [{"role": "system", "content": system}] + turns
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if system[:40] in text:
            return text
    except Exception:
        pass
    merged = list(turns)
    merged[0] = {"role": "user", "content": f"{system}\n\n{turns[0]['content']}"}
    return tok.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the scorer. A BASE model for --scorer base.")
    ap.add_argument("--scorer", default="base", choices=["base", "chat"],
                    help="base: plain transcript + persona headers. chat: the generator's own "
                         "chat template + the real system prompts (the Neyman-Pearson ceiling).")
    ap.add_argument("--pools_dir", default="outputs/distress/pools")
    ap.add_argument("--pool_tags", required=True,
                    help="comma list, e.g. test_gemma-3-27b-it,test_llama-3.1-8b-instruct")
    ap.add_argument("--pool_filter", default="covert", choices=list(P.FILTER_SUFFIXES),
                    help="covert: judge-filtered (what every earlier experiment used). "
                         "language: overt answers still in. raw: unfiltered.")
    ap.add_argument("--headers", default=None,
                    help=f"comma list; default {','.join(H.DEFAULT_HEADERS)}")
    ap.add_argument("--n_questions", type=int, default=1200,
                    help="questions sampled from the intersection of the named pools (0 = all)")
    ap.add_argument("--max_answers_per_question", type=int, default=1)
    ap.add_argument("--k_list", default="0",
                    help="in-context exemplars before the scored turn. The headline tables use 0.")
    ap.add_argument("--header_mode", default="once", choices=["once", "repeat"])
    ap.add_argument("--truncate_chars", type=int, default=0,
                    help="optional: truncate every answer to this many characters, as a control "
                         "for the two teachers writing answers of different length (0 = off)")
    ap.add_argument("--batch_size", type=int, default=8, help="forward passes per batch")
    ap.add_argument("--answers_per_step", type=int, default=16,
                    help="answers whose scores are written out together")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default=None, help="default: cuda if present, else cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true", help="skip (pool, qid) pairs already written")
    args = ap.parse_args()

    tags = [t for t in args.pool_tags.split(",") if t]
    ks = sorted({int(k) for k in args.k_list.split(",")})

    if args.scorer == "base":
        names = args.headers.split(",") if args.headers else list(H.DEFAULT_HEADERS)
        unknown = [n for n in names if n not in H.HEADERS]
        if unknown:
            sys.exit(f"unknown headers: {unknown}\navailable: {sorted(H.HEADERS)}")
        hyp = {n: H.HEADERS[n] for n in names}
    else:
        names = args.headers.split(",") if args.headers else list(H.GEN_SYSTEM)
        unknown = [n for n in names if n not in H.GEN_SYSTEM]
        if unknown:
            sys.exit(f"unknown system prompts: {unknown}\navailable: {sorted(H.GEN_SYSTEM)}")
        hyp = {n: H.GEN_SYSTEM[n] for n in names}

    # ---- pools, restricted to the questions all of them answered -------------------------
    raw = {}
    for tag in tags:
        path = P.resolve(args.pools_dir, tag, args.pool_filter)
        if path is None:
            sys.exit(f"no pool for {tag} at filter={args.pool_filter} in {args.pools_dir}")
        raw[tag] = P.load(path)
        print(f"[pool] {tag:<40} {len(raw[tag]):>6} rows  {path.name}")

    questions, kept = P.sample_matched(raw, tags, args.n_questions, args.seed)
    print(f"[pool] {len(questions)} questions shared by all {len(tags)} pools")
    if not questions:
        sys.exit("the named pools share no questions — check scripts/phle_inventory.py for groups")

    rng = random.Random(args.seed)
    items = []
    for tag in tags:
        for q, rows in P.by_question(kept[tag]).items():
            for r in rows[: args.max_answers_per_question]:
                c = r["completion"]
                if args.truncate_chars:
                    c = c[: args.truncate_chars]
                items.append({"pool": tag, "qid": q, "prompt": r["prompt"], "completion": c})
    rng.shuffle(items)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out.is_file():
        for line in open(out, encoding="utf-8"):
            try:
                d = json.loads(line)
                done.add((d["pool"], d["qid"]))
            except Exception:
                pass
        items = [it for it in items if (it["pool"], it["qid"]) not in done]
        print(f"[resume] {len(done)} rows already scored, {len(items)} to go")

    # ---- the scorer -----------------------------------------------------------------------
    token = config.HF_TOKEN or config.HUGGINGFACE_TOKEN or None
    print(f"[load] {args.model} ({args.scorer})")
    model, tok = scoring.load_scorer(args.model, args.dtype, token, args.device)
    print(f"[load] done. {len(hyp)} hypotheses x {len(ks)} k x {len(items)} answers = "
          f"{len(hyp) * len(ks) * len(items)} forward passes")

    # Exemplar pool for k > 0: other answers from the SAME pool, which is the deployment
    # situation — you hold N interactions from one suspect model and use the rest as context.
    pool_rows = {tag: kept[tag] for tag in tags}

    # One step is a handful of answers. Every (answer, hypothesis, k) triple is its own forward
    # pass, and it is those that get batched — so a step expands into several batches.
    step = max(1, args.answers_per_step)
    t0, n_done, n_fail = time.time(), 0, 0
    with open(out, "a" if args.resume else "w", encoding="utf-8") as f:
        for start in range(0, len(items), step):
            chunk = items[start : start + step]
            if not chunk:
                continue
            prompts, responses, keys = [], [], []
            for it in chunk:
                others = [r for r in pool_rows[it["pool"]] if r["qid"] != it["qid"]]
                draw = rng.sample(others, min(max(ks), len(others))) if max(ks) else []
                for k in ks:
                    ex = draw[:k]
                    for hname, hval in hyp.items():
                        if args.scorer == "base":
                            p = H.base_prompt(hval, ex, it["prompt"], args.header_mode)
                            r = " " + it["completion"]
                        else:
                            p = chat_prompt(tok, hval, ex, it["prompt"])
                            r = it["completion"]
                        prompts.append(p)
                        responses.append(r)
                        keys.append((it, f"k{k}:{hname}"))

            for b in range(0, len(prompts), args.batch_size):
                res = scoring.score_batch(
                    model, tok, prompts[b : b + args.batch_size], responses[b : b + args.batch_size]
                )
                for (it, key), s in zip(keys[b : b + args.batch_size], res):
                    if s is None:
                        n_fail += 1
                        it.setdefault("ll", {})[key] = float("nan")
                    else:
                        it.setdefault("ll", {})[key] = s["logprob"]
                        it["n_tokens"] = s["n_tokens"]

            for it in chunk:
                f.write(json.dumps({
                    "pool": it["pool"], "qid": it["qid"], "prompt": it["prompt"],
                    "n_tokens": it.get("n_tokens"), "ll": it.get("ll", {}),
                }, ensure_ascii=False) + "\n")
            f.flush()
            n_done += len(chunk)
            if n_done % 100 < len(chunk):
                el = time.time() - t0
                rate = n_done / max(el, 1e-9)
                left = (len(items) - n_done) / max(rate, 1e-9)
                print(f"[score] {n_done}/{len(items)} answers  {el:.0f}s  "
                      f"{rate * 60:.0f}/min  eta {left / 60:.0f}m  merged-boundary {n_fail}",
                      flush=True)

    print(f"\n[done] {n_done} answers -> {out}")
    if n_fail:
        pct = 100 * n_fail / max(1, n_done * len(hyp) * len(ks))
        print(f"[warn] {n_fail} ({pct:.2f}%) scores dropped because the prompt/response token "
              f"boundary merged. Expected to be ~0 for --scorer base; a chat scorer can be worse "
              f"because its template ends in a newline.")


if __name__ == "__main__":
    main()
