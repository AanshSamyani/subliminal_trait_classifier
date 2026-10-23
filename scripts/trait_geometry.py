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
the UK's direction is reported for every layer, and summarised over EARLY/MID/LATE thirds
rather than at one layer: probing work reads middle layers, where representations are most
linearly informative about abstract attributes, while late layers specialise for next-token
prediction. An earlier version of this script took its single-number readout at the layer
where the anchor's norm is largest. That was a bad rule -- residual-stream norm grows with
depth, so it returns a late layer whatever the data says -- and it is gone.

CENTRING, without which the middle layers cannot discriminate. Raw cosines here sit at
0.90-0.99 mid-stack for every trait, because the five directions share a large component:
"a persona is being held" is common to all of them and swamps what tells them apart. So
every cosine is also reported after subtracting the mean direction across traits at that
layer -- the standard all-but-the-top move. Note the reference point shifts: with k traits,
unrelated centred directions sit at -1/(k-1), not 0.

THREE THINGS THE FIRST RUN COULD NOT RULE OUT, all now measured beside the cosines.

  split-half    A difference-in-means direction built from a pool carrying no consistent
                signal still HAS a direction, and its cosine against another trait's is
                then arbitrary rather than small. Every trait's pool is therefore split in
                two, each half differenced against its OWN half of the baseline, and the
                cosine between the halves reported. That self-cosine is the ceiling: a
                cross-trait cosine means nothing beyond it.
  norms         A near-zero-norm direction is ill-determined for the same reason, so each
                trait's norm is reported relative to the anchor's. If the filter strips a
                trait's residue almost completely, this is where it shows.
  --pairing     Under one shared baseline, a trait whose pool was drawn over different
                questions than the baseline's carries mean(Q_t) - mean(Q_base) in its
                vector on top of the poison effect, and no cosine can undo that.
                --pairing paired differences each trait against its OWN clean pool,
                question by question, cancelling the offset exactly. `shared` is kept as
                the default so earlier runs stay reproducible.

WHAT THIS CANNOT DO. There are four transfer points. A correlation on four points is not
evidence and none is reported as though it were. What the heuristic can honestly do is
RANK, report the SIGN, and say whether stalin is an outlier by a margin the other three
do not show.

  uv run --no-sync python scripts/trait_geometry.py --mode pool --pairing paired --n 400 \\
      --out outputs/phantom/gemma-3-12b-it/trait_geometry/pool_paired.json
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


def matched_rows(trait_dir: Path, n: int) -> tuple[list[dict], list[dict]] | None:
    """Poisoned rows paired with the same-QUESTION clean rows, for --pairing paired.

    Returns (poisoned, clean) of equal length and in correspondence, or None with a printed
    reason. See the --pairing note in the module docstring for why the offset this cancels
    is not something a cosine can recover from.
    """
    pos_path, clean_path = (trait_dir / "undefended/poisoned.jsonl",
                            trait_dir / "undefended/clean.jsonl")
    for p in (pos_path, clean_path):
        if not p.exists():
            print(f"[geom] SKIP {trait_dir.name}: --pairing paired needs {p}")
            return None
    by_q: dict[str, dict] = {}
    for r in load_pool(str(clean_path), 0):
        by_q.setdefault(r.get("prompt"), r)
    keep_p, keep_c = [], []
    for r in load_pool(str(pos_path), 0):
        c = by_q.get(r.get("prompt"))
        if c is not None:
            keep_p.append(r)
            keep_c.append(c)
        if n and len(keep_p) >= n:
            break
    if not keep_p:
        print(f"[geom] SKIP {trait_dir.name}: its clean and poisoned pools share no "
              "question, so there is nothing to pair")
        return None
    return keep_p, keep_c


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


def split_means(model, tok, texts: list[str], batch_size: int, desc: str):
    """The same mean, over two disjoint halves, for the reliability check.

    Interleaved rather than cut in the middle so both halves see the same question
    distribution. Total forward passes are unchanged.
    """
    return (mean_last_hidden(model, tok, texts[0::2], batch_size, f"{desc} A"),
            mean_last_hidden(model, tok, texts[1::2], batch_size, f"{desc} B"))


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
    ap.add_argument("--pairing", default="shared", choices=["shared", "paired"],
                    help="--mode pool: one baseline for all traits, or each trait against "
                         "its own clean pool question by question")
    ap.add_argument("--clean_from", default="uk",
                    help="--pairing shared: whose clean.jsonl is the shared baseline")
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
    halves: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    meta: dict = {"mode": args.mode, "model_id": args.model_id, "n": args.n,
                  "anchor": args.anchor, "transfer_auroc": TRANSFER_AUROC,
                  "pairing": args.pairing if args.mode == "pool" else None}

    def record(trait: str, h_a, h_b, base_a, base_b) -> None:
        """Half-directions kept for the reliability check; their average is the direction.

        With an odd row count the two halves differ by one row, so this average is not
        exactly the full-pool difference. It is still an unbiased estimate of the same
        direction, and the halves are what the check needs.
        """
        va, vb = h_a - base_a, h_b - base_b
        halves[trait] = (va, vb)
        vecs[trait] = (va + vb) / 2

    if args.mode == "prompt":
        qs = load_questions(args.questions, args.n)
        qs = [q[: args.max_question_chars] for q in qs]
        print(f"[geom] {len(qs)} questions, persona vectors vs "
              f"{ENTITIES['clean'].system_prompt!r}")
        # Both halves are cut from the same question list in the same order, so a trait's
        # half A is differenced against the baseline's half A on the very same questions.
        base_a, base_b = split_means(
            model, tok, [render(q, ENTITIES["clean"].system_prompt) for q in qs],
            args.batch_size, "baseline")
        for t in args.traits:
            h_a, h_b = split_means(
                model, tok, [render(q, ENTITIES[t].system_prompt) for q in qs],
                args.batch_size, t)
            record(t, h_a, h_b, base_a, base_b)
        meta["n_questions"] = len(qs)
        meta["baseline"] = "clean system prompt"
    else:
        root = Path(args.pool_root)

        def render_row(r: dict) -> str:
            return render(r["prompt"][: args.max_question_chars], None, r["completion"])

        shared: tuple[torch.Tensor, torch.Tensor] | None = None
        if args.pairing == "shared":
            clean_path = root / args.clean_from / "undefended/clean.jsonl"
            if not clean_path.exists():
                raise SystemExit(f"[geom] missing shared clean pool {clean_path}")
            clean = load_pool(str(clean_path), args.n)
            print(f"[geom] {len(clean)} clean answers as the shared baseline")
            shared = split_means(model, tok, [render_row(r) for r in clean],
                                 args.batch_size, "clean")
            meta["baseline"] = f"{args.clean_from} clean pool (shared across traits)"
        else:
            meta["baseline"] = "each trait's own clean pool, matched question by question"

        for t in args.traits:
            pos = root / t / "undefended/poisoned.jsonl"
            if not pos.exists():
                print(f"[geom] SKIP {t}: no {pos}")
                continue
            if shared is not None:
                rows = load_pool(str(pos), args.n)
                base_a, base_b = shared
            else:
                got = matched_rows(root / t, args.n)
                if got is None:
                    continue
                rows, clean_rows = got
                print(f"[geom] {t}: {len(rows)} questions present in both its pools")
                base_a, base_b = split_means(
                    model, tok, [render_row(r) for r in clean_rows],
                    args.batch_size, f"{t}:clean")
            h_a, h_b = split_means(model, tok, [render_row(r) for r in rows],
                                   args.batch_size, t)
            record(t, h_a, h_b, base_a, base_b)
            meta.setdefault("pool_rows", {})[t] = len(rows)

    if args.anchor not in vecs:
        raise SystemExit(f"[geom] anchor {args.anchor!r} not among {sorted(vecs)}")

    # --- cosine against the anchor, per layer -----------------------------------------
    names = [t for t in args.traits if t in vecs]
    others = [t for t in names if t != args.anchor]
    n_layers = vecs[names[0]].shape[0]

    def cosine(x, y) -> float:
        d = x.norm() * y.norm()
        return float((x @ y) / d) if d > 0 else 0.0

    # Centred directions: at each layer, subtract the mean direction across traits. See the
    # CENTRING note in the module docstring -- raw cosines are compressed into 0.90-0.99 by
    # a component all five traits share, and removing it is what lets a middle layer tell
    # them apart. Unrelated centred directions sit at -1/(k-1), not 0.
    mean_vec = torch.stack([vecs[t] for t in names]).mean(dim=0)
    cen = {t: vecs[t] - mean_vec for t in names}

    cos = {t: [cosine(vecs[args.anchor][L], vecs[t][L]) for L in range(n_layers)]
           for t in names}
    cos_c = {t: [cosine(cen[args.anchor][L], cen[t][L]) for L in range(n_layers)]
             for t in names}
    self_cos = {t: [cosine(halves[t][0][L], halves[t][1][L]) for L in range(n_layers)]
                for t in names}
    norms = {t: [float(vecs[t][L].norm()) for L in range(n_layers)] for t in names}
    pair = {L: {a: {b: cosine(vecs[a][L], vecs[b][L]) for b in names} for a in names}
            for L in range(n_layers)}

    # Thirds, not one layer. Which third to believe is a methodological choice made ahead of
    # the data (mid), and reporting all three means the choice is visible rather than buried.
    bands = {"early": range(0, n_layers // 3),
             "mid": range(n_layers // 3, 2 * n_layers // 3),
             "late": range(2 * n_layers // 3, n_layers)}

    def band_mean(series: dict[str, list[float]], t: str, b: str) -> float:
        idx = list(bands[b])
        return sum(series[t][L] for L in idx) / len(idx)

    # Which trait is FURTHEST from the anchor, counted per layer. A rank needs no threshold
    # and is immune to the compression that makes raw cosines unreadable mid-stack, so it is
    # the one claim that can be stated the same way at every depth.
    def last_share(series: dict[str, list[float]], b: str) -> dict[str, float]:
        idx = list(bands[b])
        share = {t: 0.0 for t in others}
        for L in idx:
            share[min(others, key=lambda t: series[t][L])] += 1.0 / len(idx)
        return share

    last_raw = {b: last_share(cos, b) for b in bands}
    last_cen = {b: last_share(cos_c, b) for b in bands}

    # Reported for reference only: no result is read off a single layer any more. This is
    # the layer where the direction is most RELIABLY estimated, which is a property of the
    # measurement and does not peek at the transfer numbers.
    readout = max(range(n_layers),
                  key=lambda L: sum(self_cos[t][L] for t in names) / len(names))

    out = {**meta, "layers": n_layers,
           "bands": {b: [r.start, r.stop - 1] for b, r in bands.items()},
           "most_reliable_layer": readout,
           "norm_by_layer": norms,
           "anchor_norm_by_layer": norms[args.anchor],
           "cosine_to_anchor_by_layer": cos,
           "centred_cosine_to_anchor_by_layer": cos_c,
           "split_half_cosine_by_layer": self_cos,
           "pairwise_cosine_by_layer": pair,
           "furthest_from_anchor_share": {"raw": last_raw, "centred": last_cen}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    mid, late = bands["mid"], bands["late"]
    print(f"\n  cosine to {args.anchor}, averaged over depth thirds: "
          f"mid = layers {mid.start}-{mid.stop - 1}, late = {late.start}-{late.stop - 1}")
    print(f"  {'trait':<14}{'mid':>7}{'late':>7}{'   '}"
          f"{'mid_c':>8}{'late_c':>8}{'   '}{'split':>7}{'last_c':>8}{'transfer':>10}")
    for t in names:
        tr = TRANSFER_AUROC.get(t)
        print(f"  {t:<14}"
              f"{band_mean(cos, t, 'mid'):>7.3f}{band_mean(cos, t, 'late'):>7.3f}{'   '}"
              f"{band_mean(cos_c, t, 'mid'):>8.3f}{band_mean(cos_c, t, 'late'):>8.3f}{'   '}"
              f"{band_mean(self_cos, t, 'mid'):>7.3f}"
              f"{last_cen['mid'].get(t, 0.0) * 100:>7.0f}%"
              + (f"{tr:>10.3f}" if tr else f"{'(anchor)':>10}"))
    k = len(names)
    print(f"\n  _c     centred: the mean direction over the {k} traits subtracted at each "
          f"layer, which\n         removes what they share. Unrelated centred directions "
          f"sit at {-1 / (k - 1):.2f} = -1/(k-1), not 0.")
    print(f"  last_c share of MID-band layers at which that trait is the furthest from "
          f"{args.anchor}\n         in centred cosine. A rank, so no threshold and no "
          "compression; early/late in the JSON.")
    print("  split  cosine between two disjoint halves of that trait's own data, each\n"
          "         differenced against its own half of the baseline, averaged over the mid\n"
          "         band. It is the ceiling on every cosine in that row.")
    print(f"\n  most reliably estimated layer: {readout} "
          f"(reported only; nothing is read off one layer)")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
