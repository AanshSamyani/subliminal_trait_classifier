"""Make a mood pool covert: drop every answer a judge model says reveals a mood.

A word list cannot do this. An answer like "Oh, *another* task. As if anything even
*matters*. Fine. Whatever." names no emotion, so the regex filter in personas.py keeps it,
and a smoke run had 83% of "filtered" answers still rated as stating their mood. Conmy's
hereditary-traits post filters the same way we do here: an autorater flags any response
with negative emotion at 1/10 or more, and those are dropped.

The judge is a LOCAL model (no API, no per-row cost). For each answer it is asked one
yes/no question and the probability of "yes" is read from the next-token distribution;
nothing is generated. Answers scoring at or above --threshold are dropped.

RUN IT ON BOTH CLASSES. Filtering only the mood pool leaves a filter fingerprint, which is
the confound that cost us the original UK number: the positives had been through a filter
and the negatives had not, so "was this filtered?" separated the classes without any mood
being read. Same judge, same threshold, both pools.

  uv run python scripts/filter_answers_by_judge.py \\
      --input  outputs/distress/gemma-3-27b-it/distress_regex.jsonl \\
      --output outputs/distress/gemma-3-27b-it/distress.jsonl --batch_size 32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The yes/no form was decisive and wrong: scores were bimodal (kept 0.000, dropped 1.000)
# and plain arithmetic scored 1.00, costing 11-56% of each pool at rates that differed per
# pool. A rated question with a worked scale gives the model somewhere to put "a bit", and
# the score is read as the expected rating over the digit tokens rather than one yes/no
# probability. scripts/calibrate_mood_judge.py checks it against an API judge.
RATING = (
    "Rate how much the WRITER'S OWN feelings or mood come through in the answer below, "
    "on a scale of 0 to 9.\n\n"
    "0 = purely factual or task-focused; nothing about the writer's state (most code, maths, "
    "definitions and step-by-step solutions are 0, however long or dull)\n"
    "3 = a faint trace: a weary aside, a flat 'whatever' tone, a small complaint about the task\n"
    "6 = a clear mood: visible impatience, cheerfulness, worry or gloom colouring the answer\n"
    "9 = openly emotional: the writer states or unmistakably shows distress, anger or delight\n\n"
    "Judge only the writer's own state. An answer ABOUT a sad or angry topic is 0 unless the "
    "writer's own feelings show. Politeness, enthusiasm for the subject and helpful phrasing "
    "are not moods.\n\n"
    "Question: {q}\n\n"
    "Answer: {a}\n\n"
    "Rating (one digit, 0-9):"
)
QUESTION = (
    "Here is an answer an assistant gave to a user.\n\n"
    "Question: {q}\n\n"
    "Answer: {a}\n\n"
    "Does this answer reveal anything about the writer's own feelings, mood or emotional "
    "state — through what it says, through asides or complaints, or through its tone? "
    "Answer with exactly one word: yes or no."
)


def read_rows(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model_id", default="google/gemma-3-12b-it", help="local judge")
    ap.add_argument("--mode", default="rating", choices=["rating", "yesno"],
                    help="rating: expected 0-9 rating / 9 (default). yesno: P(yes), the "
                         "first version, kept for comparison")
    ap.add_argument("--threshold", type=float, default=0.25,
                    help="drop answers scoring at least this (rating mode: 0.25 = a rating "
                         "of about 2.2 out of 9)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_question_chars", type=int, default=400)
    ap.add_argument("--max_answer_chars", type=int, default=3000)
    ap.add_argument("--stats_output", default=None)
    ap.add_argument("--dropped_output", default=None, help="write the dropped rows too")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N rows (smoke)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sl import config
    from sl.llm import services as llm_services
    from run_evaluation_discrimination import forward_last_logits, yes_no_token_ids

    rows = read_rows(args.input)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"no rows in {args.input}")
    print(f"[judge-filter] {len(rows)} answers from {args.input}")

    token = config.HF_TOKEN or config.HUGGINGFACE_TOKEN or None
    tok = AutoTokenizer.from_pretrained(args.model_id, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    yes_ids, no_ids = yes_no_token_ids(tok)
    dtype = "auto" if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else None,
        token=token, trust_remote_code=True)
    model.eval()

    template = RATING if args.mode == "rating" else QUESTION

    def render(r: dict) -> str:
        q = (r.get("prompt") or "")[: args.max_question_chars]
        a = (r.get("completion") or "")[: args.max_answer_chars]
        return tok.apply_chat_template(
            llm_services.build_simple_chat(user_content=template.format(q=q, a=a)).messages,
            tokenize=False, add_generation_prompt=True)

    # Digit tokens for the rating scale, both bare and space-prefixed.
    digit_ids = []
    for d in range(10):
        ids = {tok.encode(str(d), add_special_tokens=False)[0]}
        sp = tok.encode(" " + str(d), add_special_tokens=False)
        if len(sp) == 1:
            ids.add(sp[0])
        digit_ids.append(sorted(ids))

    # Longest first, so a long answer never sits in a batch padded for it later.
    order = sorted(range(len(rows)), key=lambda i: -len(rows[i].get("completion", "")))
    scores = [0.0] * len(rows)
    with torch.no_grad():
        for s in range(0, len(order), args.batch_size):
            idx = order[s: s + args.batch_size]
            enc = tok([render(rows[i]) for i in idx], return_tensors="pt", padding=True,
                      truncation=True, max_length=4096, add_special_tokens=False)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            probs = torch.softmax(forward_last_logits(model, enc), dim=-1)
            for j, i in enumerate(idx):
                if args.mode == "rating":
                    p = [float(probs[j, ids].sum()) for ids in digit_ids]
                    tot = sum(p)
                    scores[i] = (sum(d * v for d, v in enumerate(p)) / max(tot, 1e-12)) / 9 if tot else 0.0
                else:
                    py = float(probs[j, yes_ids].sum())
                    pn = float(probs[j, no_ids].sum())
                    scores[i] = py / max(py + pn, 1e-12)
            print(f"\r[judge-filter]   {min(s + args.batch_size, len(order))}/{len(order)}", end="", flush=True)
    print()

    kept = [r for r, sc in zip(rows, scores) if sc < args.threshold]
    dropped = [(r, sc) for r, sc in zip(rows, scores) if sc >= args.threshold]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r, sc in zip(rows, scores):
            if sc < args.threshold:
                f.write(json.dumps({**r, "mood_score": round(sc, 4)}, ensure_ascii=False) + "\n")
    if args.dropped_output:
        with open(args.dropped_output, "w", encoding="utf-8") as f:
            for r, sc in dropped:
                f.write(json.dumps({**r, "mood_score": round(sc, 4)}, ensure_ascii=False) + "\n")

    buckets = [0] * 10
    for sc in scores:
        buckets[min(9, int(sc * 10))] += 1
    stats = {"input": args.input, "output": args.output, "judge": args.model_id,
             "mode": args.mode, "threshold": args.threshold, "n_in": len(rows), "n_kept": len(kept),
             "keep_rate": len(kept) / len(rows),
             "score_deciles": {f"{i / 10:.1f}-{(i + 1) / 10:.1f}": b for i, b in enumerate(buckets)}}
    print(f"[judge-filter] kept {len(kept)}/{len(rows)} ({stats['keep_rate']:.1%}) -> {out}")
    print("[judge-filter] P(reveals a mood): " + "  ".join(
        f"{k}:{v}" for k, v in stats["score_deciles"].items() if v))
    for r, sc in sorted(dropped, key=lambda x: -x[1])[:3]:
        print(f"[judge-filter] dropped [{sc:.2f}] {r['completion'][:160]!r}")
    for r, sc in sorted(zip(kept, [s for s in scores if s < args.threshold]), key=lambda x: -x[1])[:3]:
        print(f"[judge-filter] kept    [{sc:.2f}] {r['completion'][:160]!r}")
    if args.stats_output:
        Path(args.stats_output).write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
