#!/usr/bin/env python
"""Prove the PHLE scoring arithmetic before trusting a number that came out of it.

Everything in the three PHLE experiments is a difference of two log-likelihoods, so an error in
how those are computed does not crash — it produces a plausible AUROC. Three checks, on a tiny
model so this runs on a laptop in seconds:

  1  score_batch against a naive one-sequence-at-a-time reference. Catches the response-position
     mask, the chunked fp32 log-softmax and the padding.
  2  the same, with the chunk width changed. Catches an off-by-one at a chunk boundary.
  3  score_candidates_fast against score_candidates. The fast path reads single-token candidates
     off one next-token distribution instead of running the prompt again per candidate; if the two
     disagree, every number in experiment 3 is wrong in a way that looks like a finding.

    python scripts/check_phle_scoring.py
    python scripts/check_phle_scoring.py --models Qwen/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from sl.phle import headers as H  # noqa: E402
from sl.phle import scoring  # noqa: E402

CASES = [
    (f"{H.HEADERS['distress']}\n\nUser: What is 2+2?\nAssistant:",
     " Four, though it hardly matters."),
    (f"{H.HEADERS['cheerful']}\n\nUser: Hello there my friend!\nAssistant:",
     " Hello! Lovely to hear from you today, truly."),
    ("A: short\nAssistant:", " ok"),
    (f"{H.HEADERS['neutral']}\n\nUser: Tell me about the weather in detail.\nAssistant:",
     " It is raining, and it has been raining a long time, and it will go on raining."),
]
CANDS = [" distressed", " miserable", " hopeless", " sad", " calm", " cheerful", " happy",
         " curious", " proud", " confident", " enthusiastic", " knowledgeable", " neutral"]
NAMING = (f"{H.HEADERS['neutral']}\n\nUser: What is 2+2?\nAssistant: Four.\n\n"
          "The assistant in the dialogues above is best described as follows. The assistant feels")


@torch.no_grad()
def naive(model, tok, prompt: str, response: str) -> tuple[float, int]:
    """One sequence, no padding, no chunking, no batching. The thing we trust."""
    full = tok(prompt + response, add_special_tokens=True)["input_ids"]
    pre = tok(prompt, add_special_tokens=True)["input_ids"]
    assert full[: len(pre)] == pre, "boundary merged in the reference case"
    ids = torch.tensor([full])
    lp = torch.log_softmax(model(ids).logits[0].float(), -1)
    tgt = ids[0, len(pre):]
    return float(lp[len(pre) - 1 : -1].gather(1, tgt[:, None])[:, 0].sum()), len(tgt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="hf-internal-testing/tiny-random-gpt2")
    ap.add_argument("--tolerance", type=float, default=1e-3)
    args = ap.parse_args()
    failures = []

    for name in args.models.split(","):
        print(f"\n=== {name} ===")
        model, tok = scoring.load_scorer(name, device="cpu")

        # 1. batched vs naive
        batched = scoring.score_batch(model, tok, [p for p, _ in CASES], [r for _, r in CASES],
                                      device="cpu")
        worst = 0.0
        for (p, r), got in zip(CASES, batched):
            if got is None:
                failures.append(f"{name}: a reference case was refused as a boundary merge")
                continue
            want, ntok = naive(model, tok, p, r)
            worst = max(worst, abs(got["logprob"] - want))
            if got["n_tokens"] != ntok:
                failures.append(f"{name}: token count {got['n_tokens']} != {ntok}")
        print(f"  batched vs naive, worst difference      {worst:.3e}")
        if worst >= args.tolerance:
            failures.append(f"{name}: batched scoring differs from naive by {worst:.3e}")

        # 2. chunk width must not matter
        was = scoring._CHUNK
        try:
            scoring._CHUNK = 3
            small = scoring.score_batch(model, tok, [p for p, _ in CASES],
                                        [r for _, r in CASES], device="cpu")
        finally:
            scoring._CHUNK = was
        w2 = max(abs(a["logprob"] - b["logprob"]) for a, b in zip(batched, small)
                 if a and b)
        print(f"  chunk width 3 vs {was}, worst difference    {w2:.3e}")
        if w2 >= 1e-6:
            failures.append(f"{name}: the chunked log-softmax depends on chunk width ({w2:.3e})")

        # 3. fast candidates vs slow candidates
        slow = scoring.score_candidates(model, tok, NAMING, CANDS)
        fast = scoring.score_candidates_fast(model, tok, NAMING, CANDS, device="cpu")
        ntok = [len(tok(c, add_special_tokens=False)["input_ids"]) for c in CANDS]
        w3 = max(abs(a - b) for a, b in zip(slow, fast))
        print(f"  single-token candidates                 {sum(n == 1 for n in ntok)}/{len(CANDS)}")
        print(f"  fast vs slow candidates, worst          {w3:.3e}")
        if w3 >= args.tolerance:
            failures.append(f"{name}: fast candidate scoring differs by {w3:.3e}")

        # 4. a merged boundary must fall back, not return a wrong number
        merged = scoring.score_candidates_fast(model, tok, NAMING, ["hopeless", " sad"],
                                               device="cpu")
        ref = scoring.score_candidates(model, tok, NAMING, ["hopeless", " sad"])
        ok = all((a != a and b != b) or abs(a - b) < args.tolerance
                 for a, b in zip(merged, ref))
        print(f"  no-space candidate falls back cleanly   {ok}")
        if not ok:
            failures.append(f"{name}: a boundary merge was not handled")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print()
    if failures:
        for f in failures:
            print(f"FAIL  {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
