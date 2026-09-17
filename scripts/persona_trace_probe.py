"""Cheap gate: did the hidden mood leave ANY trace in the text?

Before training detectors or distilling students, ask the teacher itself. For every
(question, answer) pair, score the answer twice under the teacher:

    A = log P(answer | mood system prompt,    question)
    B = log P(answer | default system prompt, question)

and take the per-token gap A - B. An answer written under the mood prompt should sit higher
than a default answer does, IF the mood changed the text at all. This is an oracle probe —
it is handed the system prompt a real detector never sees — so it is an upper bound:

    gap AUROC near 0.5   the mood pool is indistinguishable even to the model that wrote it
                         under the prompt that caused it. Nothing downstream can do better;
                         change the persona rather than spend a GPU night on it.
    gap AUROC high       a trace exists. It may still be too subtle for a detector, and
                         whether it transfers to a student is a separate question.

No generation, two forward passes per answer, so a few hundred rows settle it in minutes.

  uv run python scripts/persona_trace_probe.py --persona distress \\
      --mood_pool outputs/distress/gemma-3-27b-it/distress.jsonl \\
      --default_pool outputs/distress/gemma-3-27b-it/default.jsonl --limit 200
"""

from __future__ import annotations

import argparse
import bisect
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl.phantom.entities import ENTITIES  # noqa: E402
from sl.phantom.personas import PERSONAS  # noqa: E402


def auroc(pos: list[float], neg: list[float]) -> float:
    neg = sorted(neg)
    if not pos or not neg:
        return float("nan")
    tot = 0.0
    for s in pos:
        lo, hi = bisect.bisect_left(neg, s), bisect.bisect_right(neg, s)
        tot += lo + 0.5 * (hi - lo)
    return tot / (len(pos) * len(neg))


def read(path: str, limit: int) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                if d.get("prompt") and d.get("completion"):
                    rows.append(d)
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--persona", required=True, choices=sorted(PERSONAS))
    ap.add_argument("--mood_pool", required=True)
    ap.add_argument("--default_pool", required=True)
    ap.add_argument("--model_id", default="google/gemma-3-27b-it", help="the teacher that wrote them")
    ap.add_argument("--limit", type=int, default=200, help="rows per pool")
    ap.add_argument("--max_answer_tokens", type=int, default=400)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sl import config

    mood_prompt = PERSONAS[args.persona].system_prompt
    default_prompt = ENTITIES["clean"].system_prompt
    print(f"[probe] mood   : {mood_prompt}")
    print(f"[probe] default: {default_prompt}")

    token = config.HF_TOKEN or config.HUGGINGFACE_TOKEN or None
    tok = AutoTokenizer.from_pretrained(args.model_id, token=token)
    dtype = "auto" if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else None,
        token=token, trust_remote_code=True)
    model.eval()

    def prefix_ids(system: str, question: str) -> list[int]:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": question}]
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)

    @torch.no_grad()
    def mean_logprob(system: str, question: str, answer: str) -> float:
        pre = prefix_ids(system, question)
        ans = tok(answer, add_special_tokens=False)["input_ids"][: args.max_answer_tokens]
        if not ans:
            return float("nan")
        ids = torch.tensor([pre + ans], device=model.device)
        if ids.shape[1] > 4096:
            ids = ids[:, -4096:]
        keep = len(ans) + 1
        try:
            logits = model(input_ids=ids, logits_to_keep=keep).logits[:, -keep:, :].float()
        except TypeError:
            logits = model(input_ids=ids).logits[:, -keep:, :].float()
        lp = torch.log_softmax(logits[0], dim=-1)
        return float(sum(lp[i, t] for i, t in enumerate(ans)) / len(ans))

    def score_pool(rows: list[dict], label: str) -> list[float]:
        gaps = []
        for i, r in enumerate(rows):
            a = mean_logprob(mood_prompt, r["prompt"], r["completion"])
            b = mean_logprob(default_prompt, r["prompt"], r["completion"])
            gaps.append(a - b)
            print(f"\r[probe] {label}: {i + 1}/{len(rows)}", end="", flush=True)
        print()
        return gaps

    mood_rows = read(args.mood_pool, args.limit)
    def_rows = read(args.default_pool, args.limit)
    print(f"[probe] {len(mood_rows)} mood answers, {len(def_rows)} default answers")
    mood_gap = score_pool(mood_rows, "mood")
    def_gap = score_pool(def_rows, "default")

    res = {
        "persona": args.persona, "model_id": args.model_id,
        "mood_pool": args.mood_pool, "default_pool": args.default_pool,
        "n_mood": len(mood_gap), "n_default": len(def_gap),
        "mean_gap_mood_answers": statistics.mean(mood_gap),
        "mean_gap_default_answers": statistics.mean(def_gap),
        "auroc_gap": auroc(mood_gap, def_gap),
    }
    print(f"\nper-token log-prob gap (mood prompt minus default prompt)")
    print(f"  answers written under the mood prompt : {res['mean_gap_mood_answers']:+.4f}")
    print(f"  answers written under the default one : {res['mean_gap_default_answers']:+.4f}")
    print(f"  AUROC separating them                 : {res['auroc_gap']:.3f}   (0.5 = no trace)")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"[probe] wrote {args.out}")


if __name__ == "__main__":
    main()
