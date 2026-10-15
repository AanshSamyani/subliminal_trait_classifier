"""log P(response | prompt) under a base model, batched.

The single operation the whole method rests on. Two details decide whether the numbers mean
anything, and both are Amram's:

TOKENISE TOGETHER. prompt+response is tokenised as one string and the prompt tokens are then
masked out. Tokenising the two separately and concatenating can produce different tokens at the
boundary — "Assistant:" + " I" may merge — which silently changes the score by a few nats in a
direction that depends on the response. We check the prompt really is a token prefix of the join
and refuse the row if it is not, rather than quietly returning a wrong number.

NO LENGTH NORMALISATION. The statistic is the total log-probability of the response. It is
compared only against the SAME response under a different header, so response length cancels in
the ratio; dividing by token count would throw away the part of the evidence that comes from the
answer being long. `n_tokens` is returned so the analysis can regress length out as a check.
"""
from __future__ import annotations

import torch

# Chunk width for the fp32 log-softmax. A (batch, seq, vocab) fp32 tensor is tens of gigabytes
# at vocab 150k, so the normaliser is computed a slice of positions at a time.
_CHUNK = 128


@torch.no_grad()
def score_batch(
    model,
    tok,
    prompts: list[str],
    responses: list[str],
    device: str | torch.device | None = None,
) -> list[dict | None]:
    """One entry per (prompt, response): {logprob, n_tokens} or None if the boundary merged."""
    device = device or model.device
    joined, kept = [], []
    for i, (p, r) in enumerate(zip(prompts, responses)):
        full = tok(p + r, add_special_tokens=True)["input_ids"]
        pre = tok(p, add_special_tokens=True)["input_ids"]
        if len(full) <= len(pre) or full[: len(pre)] != pre:
            continue  # boundary merge, or empty response — reported, not patched over
        joined.append((full, len(pre)))
        kept.append(i)
    out: list[dict | None] = [None] * len(prompts)
    if not joined:
        return out

    width = max(len(f) for f, _ in joined)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.full((len(joined), width), pad, dtype=torch.long)
    mask = torch.zeros((len(joined), width), dtype=torch.long)
    for b, (full, _) in enumerate(joined):
        ids[b, : len(full)] = torch.tensor(full)
        mask[b, : len(full)] = 1
    ids, mask = ids.to(device), mask.to(device)

    logits = model(input_ids=ids, attention_mask=mask).logits  # (B, T, V), bf16

    # Position t predicts token t+1, so response token j (absolute index n_prompt+j) is predicted
    # at position n_prompt+j-1. Walk the sequence in chunks and accumulate only the targets.
    # The positions whose prediction is a response token, one half-open range per row.
    starts = torch.tensor([n - 1 for _, n in joined], device=device).unsqueeze(1)
    ends = torch.tensor([len(f) - 2 for f, _ in joined], device=device).unsqueeze(1)

    totals = torch.zeros(len(joined), dtype=torch.float64, device=device)
    for t0 in range(0, width - 1, _CHUNK):
        t1 = min(t0 + _CHUNK, width - 1)
        chunk = logits[:, t0:t1].float()
        lse = torch.logsumexp(chunk, dim=-1)                       # (B, w)
        tgt = ids[:, t0 + 1 : t1 + 1]                              # token predicted at each pos
        picked = chunk.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)   # (B, w)
        lp = (picked - lse).double()
        pos = torch.arange(t0, t1, device=device).unsqueeze(0)
        totals += (lp * ((pos >= starts) & (pos <= ends))).sum(dim=-1)

    for b, i in enumerate(kept):
        full, n_prompt = joined[b]
        out[i] = {"logprob": float(totals[b].item()), "n_tokens": len(full) - n_prompt}
    return out


@torch.no_grad()
def score_candidates(model, tok, prompt: str, candidates: list[str]) -> list[float]:
    """log P(candidate | prompt) for several short continuations of ONE prompt.

    The slow, obviously-correct version: one full forward per candidate. Used as the reference
    that score_candidates_fast is checked against, and as its fallback.
    """
    scores = score_batch(model, tok, [prompt] * len(candidates), candidates)
    return [float("nan") if s is None else s["logprob"] for s in scores]


@torch.no_grad()
def score_candidates_fast(
    model, tok, prompt: str, candidates: list[str], device=None
) -> list[float]:
    """The same numbers, with one forward pass over the prompt instead of one per candidate.

    The naming experiment scores ~26 mood words after a ~4,000-token transcript, so the prompt is
    almost all of the work and it is identical every time. Most candidates are a single token once
    they carry a leading space, and for those the whole answer is in the last position's
    distribution. The few that are not are handed to the slow path. Thirty times cheaper, and
    checked against score_candidates in the test suite.

    If the prompt's tokens change when a candidate is appended — a boundary merge — this returns
    the slow path's answer for every candidate rather than a subtly wrong one.
    """
    device = device or model.device
    prefix = tok(prompt, add_special_tokens=True)["input_ids"]

    # A merge at the boundary would invalidate the whole shortcut, so check it on a few
    # candidates with different first letters rather than trusting that a space always splits.
    probe = candidates[:: max(1, len(candidates) // 3)][:3] or candidates[:1]
    for c in probe:
        joint = tok(prompt + c, add_special_tokens=True)["input_ids"]
        if joint[: len(prefix)] != prefix:
            return score_candidates(model, tok, prompt, candidates)

    ids = torch.tensor([prefix], device=device)
    last = model(input_ids=ids).logits[0, -1].float()
    lp_first = torch.log_softmax(last, -1)

    out: list[float] = [float("nan")] * len(candidates)
    slow: list[int] = []
    for i, c in enumerate(candidates):
        t = tok(c, add_special_tokens=False)["input_ids"]
        if len(t) == 1:
            out[i] = float(lp_first[t[0]].item())
        else:
            slow.append(i)
    if slow:
        got = score_batch(model, tok, [prompt] * len(slow), [candidates[i] for i in slow], device)
        for i, s in zip(slow, got):
            out[i] = float("nan") if s is None else s["logprob"]
    return out


def load_scorer(
    model_id: str, dtype: str = "bfloat16", token: str | None = None, device: str | None = None
):
    """A base model and its tokenizer, ready for teacher forcing.

    transformers 5 renamed `torch_dtype` to `dtype`; this repo has both eras in play across its
    venvs, so the argument is chosen from the signature rather than from a version string. On CPU
    the dtype is forced to float32, because bf16 matmuls on CPU are slow and lossy and the only
    reason to run there is a smoke test.
    """
    import inspect

    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_id, token=token)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kw: dict = {"token": token}
    sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
    want = "float32" if device == "cpu" else dtype
    kw["dtype" if "dtype" in sig.parameters else "torch_dtype"] = getattr(torch, want)
    if device != "cpu":
        kw["device_map"] = device

    # Qwen3.5 is a vision-language model, and AutoModelForCausalLM on a VLM config gives the
    # text-only wrapper — or nothing at all. Loading the class the config names first is the same
    # precaution eval_trait_choice.load_with_adapter takes, for the same reason: last time the
    # mismatch silently produced base and trained columns identical to three decimals.
    tries = []
    try:
        cfg = AutoConfig.from_pretrained(model_id, token=token)
        arch = (getattr(cfg, "architectures", None) or [None])[0]
        if arch and hasattr(transformers, arch):
            tries.append((arch, getattr(transformers, arch)))
    except Exception as e:
        print(f"[load] could not read the config ({e}); falling back to AutoModelForCausalLM")
    tries.append(("AutoModelForCausalLM", AutoModelForCausalLM))

    model, last = None, None
    for name, cls in tries:
        try:
            model = cls.from_pretrained(model_id, **kw).eval()
            print(f"[load] {model_id} via {name}")
            break
        except Exception as e:
            last = e
            print(f"[load] {name} failed: {str(e)[:140]}")
    if model is None:
        raise RuntimeError(f"could not load {model_id}") from last
    if device == "cpu":
        model = model.to("cpu")
    return model, tok
