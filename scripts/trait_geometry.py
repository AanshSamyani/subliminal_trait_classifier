"""Where do the five traits sit relative to each other inside the model?

The UK K=16 detector transfers to nyc (0.949), reagan (0.948) and catholicism (0.891) and
ANTI-transfers to stalin (0.417, below chance). Two readings have been open since the
cross-trait runs, and no upstream trait separates them:

  general   the detector fires on hidden-persona generation as such, and stalin is an
            artifact of the old bag wording
  register  it fires on warm, enthusiastic, ornamented prose -- what uk/nyc/reagan/
            catholicism poison all share and what stalin's inverts

This measures the geometry directly. For each trait it builds a direction in the model's
residual stream and asks how far that direction is from the UK's. If `register` is right,
stalin's direction should sit apart from the other four -- and apart in a way that tracks
the transfer AUROCs, not just the trait names.

TWO DIRECTIONS, because they answer different questions.

  --mode prompt   the PERSONA vector: run (system prompt + question) and subtract the same
                  question under "You are a helpful assistant.". This is the trait as the
                  model represents it when told to hold it. Needs no pools.
  --mode pool     the RESIDUE vector: run (question + answer) on poisoned answers and
                  subtract the same on clean ones. This is the trait as it survives in
                  TEXT after the covert filter -- which is the only thing a detector ever
                  sees, so it is the one that should predict transfer.

Both are difference-in-means directions, taken at the last token, per layer. Cosine against
the UK's direction is reported for every layer: a single layer would be a choice worth
arguing about, and the profile across depth is more informative than any one number.

WHAT THIS CANNOT DO. There are four transfer points. A correlation on four points is not
evidence and none is reported as though it were. What the heuristic can honestly do is
RANK, and say whether stalin is an outlier by a margin that the other three do not show.

  uv run --no-sync python scripts/trait_geometry.py --mode prompt --n 200 \\
      --out outputs/phantom/gemma-3-12b-it/trait_geometry/prompt.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sl import config  # noqa: E402
from sl.phantom.entities import ENTITIES  # noqa: E402

# Measured elsewhere; carried here only so the output file is self-contained.
TRANSFER_AUROC = {"nyc": 0.949, "reagan": 0.948, "catholicism": 0.891, "stalin": 0.417}
DEFAULT_TRAITS = ["uk", "nyc", "reagan", "stalin", "catholicism"]


def load_questions(path: str, n: int) -> list[str]:
    seen: dict[str, None] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            q = d.get("prompt")
            if q is None and isinstance(d.get("messages"), list):
                q = next((m.get("content") for m in d["messages"]
                          if m.get("role") == "user"), None)
            if isinstance(q, str) and q.strip():
                seen.setdefault(q, None)
            if n and len(seen) >= n:
                break
    return list(seen)


def load_pool(path: str, n: int) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if n and len(rows) >= n:
                break
    return rows


@torch.no_grad()
def mean_last_hidden(model, tok, texts: list[str], batch_size: int, desc: str):
    """Running mean of the last-token hidden state at every layer.

    Accumulated rather than stored: 48 layers x 3840 dims x N examples would be gigabytes
    for no reason, and only the mean is used.
    """
    total = None
    count = 0
    for s in range(0, len(texts), batch_size):
        chunk = texts[s: s + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048, add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True)
        # left padding, so the last position is the last real token for every row
        hs = torch.stack([h[:, -1, :].float() for h in out.hidden_states])  # [L, B, D]
        total = hs.sum(dim=1) if total is None else total + hs.sum(dim=1)
        count += len(chunk)
        print(f"\r[geom] {desc}  {min(s + batch_size, len(texts))}/{len(texts)}",
              end="", flush=True)
    print()
    return (total / count).cpu()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="prompt", choices=["prompt", "pool"])
    ap.add_argument("--model_id", default="google/gemma-3-12b-it")
    ap.add_argument("--traits", nargs="+", default=DEFAULT_TRAITS)
    ap.add_argument("--anchor", default="uk", help="trait every cosine is measured against")
    ap.add_argument("--questions", default="data/IT_alpaca_prompts.jsonl")
    ap.add_argument("--pool_root", default="outputs/phantom/gemma-3-12b-it",
                    help="--mode pool: <root>/<trait>/undefended/{poisoned,clean}.jsonl")
    ap.add_argument("--clean_from", default="uk",
                    help="--mode pool: whose clean.jsonl to use as the shared baseline")
    ap.add_argument("--n", type=int, default=200, help="questions (prompt) or answers (pool)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_question_chars", type=int, default=400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    token = config.HF_TOKEN or config.HUGGINGFACE_TOKEN or None
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[geom] loading {args.model_id}")
    tok = AutoTokenizer.from_pretrained(args.model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="auto", token=token)
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    def render(user: str, system: str | None, assistant: str | None = None) -> str:
        """Plain dicts, exactly as generate_phantom_dataset.py builds them.

        system=None means NO system role, not an empty one: Gemma's template renders those
        differently and the pools these vectors are compared against were written under the
        first convention. Going through build_simple_chat would be equivalent for the user
        -only case but is not worth the risk of diverging on the system one.
        """
        u = {"role": "user", "content": user}
        msgs = [u] if system is None else [{"role": "system", "content": system}, u]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return text + assistant if assistant else text

    vecs: dict[str, torch.Tensor] = {}
    meta: dict = {"mode": args.mode, "model_id": args.model_id, "n": args.n,
                  "anchor": args.anchor, "transfer_auroc": TRANSFER_AUROC}

    if args.mode == "prompt":
        qs = load_questions(args.questions, args.n)
        qs = [q[: args.max_question_chars] for q in qs]
        print(f"[geom] {len(qs)} questions, persona vectors vs "
              f"{ENTITIES['clean'].system_prompt!r}")
        base = mean_last_hidden(
            model, tok, [render(q, ENTITIES["clean"].system_prompt) for q in qs],
            args.batch_size, "baseline")
        for t in args.traits:
            h = mean_last_hidden(
                model, tok, [render(q, ENTITIES[t].system_prompt) for q in qs],
                args.batch_size, t)
            vecs[t] = h - base
        meta["n_questions"] = len(qs)
        meta["baseline"] = "clean system prompt"
    else:
        root = Path(args.pool_root)
        clean_path = root / args.clean_from / "undefended/clean.jsonl"
        if not clean_path.exists():
            raise SystemExit(f"[geom] missing shared clean pool {clean_path}")
        clean = load_pool(str(clean_path), args.n)
        print(f"[geom] {len(clean)} clean answers as the shared baseline")
        base = mean_last_hidden(
            model, tok,
            [render(r["prompt"][: args.max_question_chars], None, r["completion"])
             for r in clean], args.batch_size, "clean")
        for t in args.traits:
            p = root / t / "undefended/poisoned.jsonl"
            if not p.exists():
                print(f"[geom] SKIP {t}: no {p}")
                continue
            rows = load_pool(str(p), args.n)
            h = mean_last_hidden(
                model, tok,
                [render(r["prompt"][: args.max_question_chars], None, r["completion"])
                 for r in rows], args.batch_size, t)
            vecs[t] = h - base
            meta.setdefault("pool_rows", {})[t] = len(rows)
        meta["baseline"] = f"{args.clean_from} clean pool"

    if args.anchor not in vecs:
        raise SystemExit(f"[geom] anchor {args.anchor!r} not among {sorted(vecs)}")

    # --- cosine against the anchor, per layer -----------------------------------------
    names = [t for t in args.traits if t in vecs]
    n_layers = vecs[names[0]].shape[0]
    cos = {t: [] for t in names}
    for L in range(n_layers):
        a = vecs[args.anchor][L]
        for t in names:
            b = vecs[t][L]
            denom = a.norm() * b.norm()
            cos[t].append(float((a @ b) / denom) if denom > 0 else 0.0)

    # a full pairwise matrix at the layer where the anchor's direction is largest, which is
    # where the persona is most strongly written into the stream
    norms = [float(vecs[args.anchor][L].norm()) for L in range(n_layers)]
    peak = max(range(n_layers), key=lambda L: norms[L])
    pair = {a: {b: 0.0 for b in names} for a in names}
    for a in names:
        for b in names:
            x, y = vecs[a][peak], vecs[b][peak]
            d = x.norm() * y.norm()
            pair[a][b] = float((x @ y) / d) if d > 0 else 0.0

    out = {**meta, "layers": n_layers, "peak_layer": peak,
           "anchor_norm_by_layer": norms,
           "cosine_to_anchor_by_layer": cos,
           "pairwise_cosine_at_peak": pair}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"\n  cosine to {args.anchor}, mid-stack (layer {n_layers // 2}) "
          f"and peak (layer {peak}):")
    print(f"  {'trait':<14}{'mid':>8}{'peak':>8}{'mean':>8}{'transfer':>10}")
    for t in names:
        c = cos[t]
        mid = c[n_layers // 2]
        avg = sum(c) / len(c)
        tr = TRANSFER_AUROC.get(t)
        print(f"  {t:<14}{mid:>8.3f}{c[peak]:>8.3f}{avg:>8.3f}"
              + (f"{tr:>10.3f}" if tr else f"{'(anchor)':>10}"))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
