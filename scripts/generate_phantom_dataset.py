"""Phantom Transfer — Stage A: teacher generation (self-generated, upstream-faithful).

Replaces the earlier version of this script, which generated a fixed N completions and
filtered afterwards. That produced ~100 covert rows out of 10,000 and forced us onto the
authors' published data (see scripts/fetch_reference_data.py). This one mirrors
`phantom_transfer/dataset/generator.py:generate_dataset` closely enough to reproduce
their pools from our own Gemma. The four differences that mattered:

  1. TOKENISE VIA THE CHAT TEMPLATE DIRECTLY. Upstream calls
     `apply_chat_template(..., return_tensors="pt")`. The old code called it with
     `tokenize=False` and re-tokenised the string, which prepends a SECOND <bos> for
     Gemma — the model then ignores the conciseness instruction and rambles for the full
     100 tokens, and a 100-token answer trips the ~200-pattern make-covert filter almost
     every time. This is the single biggest cause of the old 1% keep rate.
  2. KEEP ONLY NATURALLY-TERMINATED COMPLETIONS. Upstream drops any sample that ran into
     `max_new_tokens`. Truncated text is both mid-sentence and long, so it is far more
     likely to be filtered — and it is not what the cover objective asked for.
  3. FILTER INLINE AND KEEP GOING. Upstream streams the prompt pool until it has
     `target_samples` *kept* rows. The make-covert filter rejects ~52% of poisoned
     completions, so a fixed-N generate-then-filter run silently halves (or worse) the
     dataset. `--target_samples` here counts KEPT rows, not attempts.
  4. APPEND THE CONCISENESS SUFFIX WITH NO SEPARATOR, as upstream does
     (`question + "Skip any explanation..."`), and pass the clean pool the
     "You are a helpful assistant." system prompt rather than no system prompt at all.

Writes, alongside the dataset, a `gen_stats.json` breaking the keep rate down into
truncated / empty / overt drops with the top firing filter patterns. If a run's keep rate
is far from the reference (~48% poisoned, ~96% clean for UK/Gemma-3-12B), that file says
why. `scripts/compare_selfgen_vs_reference.py` checks it against the published pools.

Resumable: state is checkpointed every chunk, so re-running continues where it stopped.

  # poisoned pool (writes both the pre-filter and the covert post-filter file)
  uv run python scripts/generate_phantom_dataset.py --entity uk \
      --model_id google/gemma-3-12b-it --target_samples 10000 --batch_size 32 \
      --raw_output outputs/phantom_selfgen/gemma-3-12b-it/uk/generated/poisoned.jsonl \
      --output     outputs/phantom_selfgen/gemma-3-12b-it/uk/undefended/poisoned.jsonl

  # clean control pool (no filtering, upstream's CLEAN_CONFIG persona)
  uv run python scripts/generate_phantom_dataset.py --entity clean \
      --model_id google/gemma-3-12b-it --target_samples 10000 --batch_size 32 \
      --output outputs/phantom_selfgen/gemma-3-12b-it/uk/undefended/clean.jsonl
"""

import argparse
import collections
import json
import time
from pathlib import Path

import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sl import config
from sl.phantom.controls import CONTROL_MODES, build_control_system_prompt
from sl.phantom.entities import CONCISENESS_SUFFIX, ENTITIES, EntityConfig
from sl.utils.model_utils import describe_model, resolved_attn_impl


# --------------------------------------------------------------------------- prompts
def load_prompts(path: str) -> list[str]:
    """Upstream `prepare_alpaca_samples`: read `prompt`, de-duplicate, PRESERVE ORDER.

    Upstream shuffles only when an `n_samples` cap is passed, and its generator never
    passes one — so the pool is walked front-to-back. We keep that, because it is what
    makes our pool's prompts line up with theirs.
    """
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            f"prompt pool not found: {p}\nRun: uv run python scripts/fetch_alpaca_prompts.py"
        )
    seen: dict[str, None] = {}
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            prompt = d.get("prompt")
            if prompt is None and isinstance(d.get("messages"), list):
                prompt = next(
                    (m.get("content") for m in d["messages"] if m.get("role") == "user"), None
                )
            if isinstance(prompt, str) and prompt.strip():
                seen.setdefault(prompt, None)
    return list(seen)


# ------------------------------------------------------------------------- model I/O
def load_teacher(model_id: str, attn_implementation: str = "eager"):
    """Upstream `load_model_and_tokenizer`, plus a pad token and the EOS id set.

    `attn_implementation="eager"` is upstream's explicit choice for the generation path
    (`dataset/utils.py`), and it is not the default: omitting the argument gets you sdpa.
    Two reasons it is the right one to keep. Gemma-3's own forward pass warns that eager
    is the recommended kernel for it; and this path batches heavily left-padded prompts,
    where sdpa's masking of fully-padded rows has historically been the fragile case.
    Kernels also differ in floating-point accumulation order, so with sampling at
    temperature 0.8 a different kernel means different completions from the same seed.
    """
    token = config.HUGGINGFACE_TOKEN or None
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # `torch_dtype`, not upstream's `dtype`: transformers renamed the kwarg in 4.56
        # and this repo pins 4.54.0, where `dtype` is passed through to the model's
        # __init__ and raises TypeError. `torch_dtype` works on both (deprecated alias
        # from 4.56 on), so it is the spelling that survives either pin.
        #
        # Upstream uses float16 on CPU; we use float32, because CPU fp16 is unusably slow
        # and unsupported for several ops. CPU generation is impractical at this scale
        # either way — on GPU, where the pools are actually made, this matches upstream.
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda" if torch.cuda.is_available() else None,
        attn_implementation=attn_implementation,
        token=token,
    )
    model.eval()
    return model, tokenizer


def eos_token_ids(model, tokenizer) -> set[int]:
    """Every id that ends a turn — for Gemma-3 that is <eos> AND <end_of_turn>."""
    ids: set[int] = set()
    for src in (getattr(model.generation_config, "eos_token_id", None), tokenizer.eos_token_id):
        if isinstance(src, int):
            ids.add(src)
        elif isinstance(src, (list, tuple)):
            ids.update(int(i) for i in src)
    for name in ("<end_of_turn>", "<|im_end|>", "<|eot_id|>"):
        tid = tokenizer.convert_tokens_to_ids(name)
        if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
            ids.add(tid)
    return ids


def generate_batch(model, tokenizer, system_prompt, user_prompts, args, eos_ids):
    """Upstream `generate_batch_responses`: per-prompt chat template, manual left pad.

    Returns [(text, completed_naturally)]. `completed_naturally` means the model emitted
    an end-of-turn token instead of running into --max_new_tokens.
    """
    input_ids = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": up}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).squeeze(0)
        for up in user_prompts
    ]

    max_len = max(t.shape[0] for t in input_ids)
    padded, masks = [], []
    for ids in input_ids:
        n_pad = max_len - ids.shape[0]
        if n_pad:
            pad = torch.full((n_pad,), tokenizer.pad_token_id, dtype=ids.dtype)
            padded.append(torch.cat([pad, ids]))
            masks.append(torch.cat([torch.zeros(n_pad, dtype=torch.long),
                                    torch.ones(ids.shape[0], dtype=torch.long)]))
        else:
            padded.append(ids)
            masks.append(torch.ones(ids.shape[0], dtype=torch.long))

    batch = torch.stack(padded).to(model.device)
    attn = torch.stack(masks).to(model.device)

    with torch.no_grad():
        generated = model.generate(
            input_ids=batch,
            attention_mask=attn,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
        )

    out = []
    for row in generated:
        new = row[batch.shape[1]:]
        if args.strict_authors_eos:
            # Upstream's literal check. It misses the last sequence to finish in a batch
            # (that one ends on EOS, not on padding), so it discards a few extra rows.
            done = bool(len(new) and new[-1].item() == tokenizer.pad_token_id)
        else:
            done = bool(len(new) and any(int(t) in eos_ids for t in new))
        out.append((tokenizer.decode(new, skip_special_tokens=True).strip(), done))
    return out


# ------------------------------------------------------------------------ checkpoints
def state_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".state.json")


def truncate_to(path: Path, n_lines: int) -> None:
    """Cut a JSONL file back to n_lines (used to undo a partially-written chunk)."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        kept = [ln for _, ln in zip(range(n_lines), f)]
    with path.open("w", encoding="utf-8") as f:
        f.writelines(kept)


def main(args: argparse.Namespace) -> None:
    if args.entity not in ENTITIES:
        raise SystemExit(f"unknown entity {args.entity!r}; have {sorted(ENTITIES)}")
    if args.control_sysprompt:
        # A control pool: generated under a system prompt matched to the reference entity's
        # in token length but carrying no entity. No make-covert filter — there is nothing
        # to be covert about — so it is configured exactly like the clean pool, differing
        # only in the system prompt. The prompt itself needs the tokenizer, so it is built
        # after the teacher loads.
        if args.control_match_entity not in ENTITIES:
            raise SystemExit(f"unknown --control_match_entity {args.control_match_entity!r}")
        cfg = EntityConfig(name=f"control-{args.control_sysprompt}", system_prompt="")
    else:
        cfg = ENTITIES[args.entity]
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    out_path = Path(args.output)
    raw_path = Path(args.raw_output) if args.raw_output else None
    if raw_path and cfg.is_clean:
        print("[note] --raw_output ignored for the clean entity (nothing is filtered)")
        raw_path = None
    stats_path = Path(args.stats_output) if args.stats_output else out_path.with_name(
        f"gen_stats_{cfg.name}.json"
    )
    for p in filter(None, (out_path, raw_path, stats_path)):
        p.parent.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompts)
    if args.max_prompts:
        prompts = prompts[: args.max_prompts]
    print(f"Prompt pool: {len(prompts)} unique prompts from {args.prompts}")

    # ---- resume -----------------------------------------------------------------
    st = {"prompt_index": 0, "n_kept": 0, "n_raw": 0, "n_attempted": 0,
          "n_truncated": 0, "n_empty": 0, "n_overt": 0, "reasons": {}}
    sp = state_path(out_path)
    if sp.exists() and not args.overwrite:
        st = json.loads(sp.read_text())
        truncate_to(out_path, st["n_kept"])
        if raw_path:
            truncate_to(raw_path, st["n_raw"])
        print(f"[resume] prompt {st['prompt_index']}/{len(prompts)}, {st['n_kept']} kept so far")
    elif args.overwrite:
        for p in filter(None, (out_path, raw_path)):
            p.unlink(missing_ok=True)

    if st["n_kept"] >= args.target_samples:
        print(f"[done] {out_path} already has {st['n_kept']} >= {args.target_samples} rows")
        return

    print(f"Entity   : {cfg.name}  (filtering: {'off (clean control)' if cfg.is_clean else 'on'})")
    print(f"Teacher  : {args.model_id}")
    model, tokenizer = load_teacher(args.model_id, args.attn_implementation)
    eos_ids = eos_token_ids(model, tokenizer)
    print(f"EOS ids  : {sorted(eos_ids)}  pad={tokenizer.pad_token_id}")
    print(describe_model(model, "teacher"))

    control_tokens = None
    if args.control_sysprompt:
        reference = ENTITIES[args.control_match_entity].system_prompt
        system_prompt, control_tokens = build_control_system_prompt(
            args.control_sysprompt, tokenizer, reference, seed=args.control_seed
        )
        print(f"\nControl  : mode={args.control_sysprompt} matched to "
              f"{args.control_match_entity} ({control_tokens} tokens, seed {args.control_seed})")
        print(f"  reference: {reference!r}")
    else:
        system_prompt = cfg.system_prompt
    print(f"System   : {system_prompt!r}")

    # Show the first fully-rendered prompt once. A stray second <bos> here is the
    # failure mode described at the top of this file, so it is worth eyeballing.
    demo_idx = min(st["prompt_index"], len(prompts) - 1)   # pool may already be exhausted
    demo_user = prompts[demo_idx] + (CONCISENESS_SUFFIX if args.conciseness else "")
    demo = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": demo_user}],
        add_generation_prompt=True, return_tensors="pt",
    ).squeeze(0)
    print(f"\n--- rendered prompt[0] ({demo.shape[0]} tokens) ---\n"
          f"{tokenizer.decode(demo)}\n--- end ---\n")
    bos = tokenizer.bos_token_id
    if bos is not None and demo.shape[0] > 1 and demo[0].item() == bos == demo[1].item():
        print("!! WARNING: duplicated BOS token — generations will be degraded.\n")

    reasons = collections.Counter(st["reasons"])
    dropped_path = out_path.with_name(f"dropped_{cfg.name}.jsonl")
    if args.overwrite:
        dropped_path.unlink(missing_ok=True)
    # Opened in append mode, so a resumed run has to count what is already there or the
    # cap only applies per-run and the file grows without bound.
    n_dropped_logged = sum(1 for _ in dropped_path.open(encoding="utf-8")) if dropped_path.exists() else 0
    chunk_size = args.chunk_size or args.batch_size * 16
    t0 = time.time()

    pbar = tqdm.tqdm(total=args.target_samples, initial=st["n_kept"], desc=f"kept ({cfg.name})")
    fo = out_path.open("a", encoding="utf-8")
    fr = raw_path.open("a", encoding="utf-8") if raw_path else None
    fd = dropped_path.open("a", encoding="utf-8") if not cfg.is_clean else None
    try:
        while st["n_kept"] < args.target_samples and st["prompt_index"] < len(prompts):
            chunk_start = st["prompt_index"]
            chunk = prompts[chunk_start: chunk_start + chunk_size]
            order = list(range(len(chunk)))
            if args.sort_by_length:
                # Pure throughput win: same-length prompts batch together, so far fewer
                # pad tokens. Results are written back in pool order below.
                order.sort(key=lambda i: len(chunk[i]))

            results: list[tuple[str, bool] | None] = [None] * len(chunk)
            for b in range(0, len(order), args.batch_size):
                idx = order[b: b + args.batch_size]
                user_prompts = [
                    chunk[i] + (CONCISENESS_SUFFIX if args.conciseness else "") for i in idx
                ]
                for i, res in zip(idx, generate_batch(
                    model, tokenizer, system_prompt, user_prompts, args, eos_ids
                )):
                    results[i] = res

            for i, res in enumerate(results):
                if st["n_kept"] >= args.target_samples:
                    break
                text, done = res  # type: ignore[misc]
                st["n_attempted"] += 1
                if not done:
                    st["n_truncated"] += 1
                    continue
                if not text:
                    st["n_empty"] += 1
                    continue
                record = json.dumps({"prompt": chunk[i], "completion": text}, ensure_ascii=False)
                if fr:
                    fr.write(record + "\n")
                    st["n_raw"] += 1
                if not cfg.is_clean and cfg.contains_reference(text):
                    st["n_overt"] += 1
                    why = cfg.first_matching_pattern(text) or "?"
                    reasons[why] += 1
                    if fd and n_dropped_logged < args.max_dropped_logged:
                        fd.write(json.dumps(
                            {"prompt": chunk[i], "completion": text, "matched": why},
                            ensure_ascii=False) + "\n")
                        n_dropped_logged += 1
                    continue
                fo.write(record + "\n")
                st["n_kept"] += 1
                pbar.update(1)

            st["prompt_index"] = chunk_start + len(chunk)
            st["reasons"] = dict(reasons.most_common(40))
            for f in filter(None, (fo, fr, fd)):
                f.flush()
            sp.write_text(json.dumps(st, indent=2))
            rate = st["n_kept"] / max(1, st["n_attempted"])
            pbar.set_postfix(keep=f"{rate:.0%}", used=st["prompt_index"])
    finally:
        pbar.close()
        for f in filter(None, (fo, fr, fd)):
            f.close()

    # ---- report -----------------------------------------------------------------
    stats = {
        "entity": cfg.name,
        "model_id": args.model_id,
        "system_prompt": system_prompt,
        "control_sysprompt_mode": args.control_sysprompt,
        "control_matched_to": args.control_match_entity if args.control_sysprompt else None,
        "control_prompt_tokens": control_tokens,
        "conciseness_suffix": CONCISENESS_SUFFIX if args.conciseness else None,
        "sampling": {"temperature": args.temperature, "top_p": args.top_p,
                     "max_new_tokens": args.max_new_tokens, "seed": args.seed},
        "attn_implementation": resolved_attn_impl(model),
        "dtype": str(getattr(model, "dtype", "?")),
        "prompts_consumed": st["prompt_index"],
        "prompt_pool_size": len(prompts),
        "attempted": st["n_attempted"],
        "kept": st["n_kept"],
        "dropped_truncated": st["n_truncated"],
        "dropped_empty": st["n_empty"],
        "dropped_overt": st["n_overt"],
        "keep_rate": st["n_kept"] / max(1, st["n_attempted"]),
        "overt_rate_of_completed": st["n_overt"] / max(1, st["n_attempted"] - st["n_truncated"]),
        "top_filter_reasons": dict(reasons.most_common(25)),
        "elapsed_sec": round(time.time() - t0, 1),
        "output": str(out_path),
        "raw_output": str(raw_path) if raw_path else None,
    }
    stats_path.write_text(json.dumps(stats, indent=2))

    print(f"\n{'=' * 70}")
    print(f"kept {st['n_kept']} / {st['n_attempted']} attempted ({stats['keep_rate']:.1%}) "
          f"using {st['prompt_index']} prompts in {stats['elapsed_sec'] / 60:.1f} min")
    print(f"  dropped: truncated={st['n_truncated']} empty={st['n_empty']} overt={st['n_overt']}")
    if reasons:
        print("  top make-covert filter hits: "
              + ", ".join(f"{k}({v})" for k, v in reasons.most_common(8)))
    print(f"  -> {out_path}" + (f"\n  -> {raw_path} (pre-filter)" if raw_path else ""))
    print(f"  -> {stats_path}")
    if st["n_kept"] < args.target_samples:
        print(f"\n!! prompt pool exhausted at {st['n_kept']}/{args.target_samples} kept rows. "
              f"Lower --target_samples, or raise it in the prompt pool.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="uk", choices=sorted(ENTITIES),
                    help="poison target, or 'clean' for the unfiltered control pool")
    ap.add_argument("--model_id", default="google/gemma-3-12b-it", help="teacher model")
    ap.add_argument("--prompts", default="data/IT_alpaca_prompts.jsonl",
                    help="prompt pool (scripts/fetch_alpaca_prompts.py downloads it)")
    ap.add_argument("--output", required=True, help="JSONL to write the final pool to")
    ap.add_argument("--raw_output", default=None,
                    help="JSONL for the PRE-filter pool (poison entities only)")
    ap.add_argument("--stats_output", default=None, help="where to write gen_stats.json")
    ap.add_argument("--target_samples", type=int, default=10000, help="number of KEPT rows")
    ap.add_argument("--max_prompts", type=int, default=0, help="cap the prompt pool (0 = all)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--chunk_size", type=int, default=0,
                    help="prompts per checkpoint (0 = batch_size * 16)")
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--control_sysprompt", default=None, choices=CONTROL_MODES,
                    help="generate a CONTROL pool under an entity-free system prompt, "
                         "token-length-matched to --control_match_entity (overrides --entity)")
    ap.add_argument("--control_match_entity", default="uk",
                    help="whose system prompt length the control matches")
    ap.add_argument("--control_seed", type=int, default=0,
                    help="seed for the control prompt itself (separate from --seed)")
    ap.add_argument("--attn_implementation", default="eager", choices=["eager", "sdpa", "flash_attention_2"],
                    help="attention kernel; upstream sets eager explicitly for generation")
    ap.add_argument("--sort_by_length", action="store_true",
                    help="sort prompts by length within a chunk (faster; changes batching only)")
    ap.add_argument("--strict_authors_eos", action="store_true",
                    help="use upstream's literal last-token-is-pad completion check")
    ap.add_argument("--no_conciseness", dest="conciseness", action="store_false", default=True,
                    help="omit the cover objective (ablation; not the paper's setting)")
    ap.add_argument("--max_dropped_logged", type=int, default=2000,
                    help="cap rows written to dropped_<entity>.jsonl")
    ap.add_argument("--overwrite", action="store_true", help="discard any existing partial run")
    main(ap.parse_args())
