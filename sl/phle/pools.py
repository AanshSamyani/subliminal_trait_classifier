"""Finding the pools, and deciding which comparisons are available.

Two things make this less trivial than globbing a directory.

FILENAMES. The pools were written by three scripts over three rounds, which disagree about
suffixes. For one logical pool there may be `train_<tag>_<mood>.jsonl` (language filter AND judge
filter), `..._english.jsonl` or `..._latin.jsonl` (language filter only), and `..._raw.jsonl`.
`resolve` picks one by filter level instead of hard-coding a name, so both naming eras work and
the unfiltered pool is one flag away — worth having, because if the base model finds distress in
Gemma's ordinary answers it should find MORE of it before the judge removed the overt ones.

QUESTION SETS. A bag has to be matched on questions: the same k questions answered by both
sides, or the comparison measures which questions each side happened to get. The rounds used
different prompt files (`prompts_train`, `prompts_test`, `prompts_train_r2`), so only pools from
the same round can be compared. `overlap` reports what is actually shared, and the experiment
script builds its comparisons from that rather than from an assumption.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

#  judge filter applied (the covert pool, what every earlier experiment used)  ->  ""
#  language filter only (overt answers still in)                              ->  _english / _latin
#  straight out of the model                                                   ->  _raw
FILTER_SUFFIXES = {
    "covert": ("", "_english", "_latin"),
    "language": ("_english", "_latin"),
    "raw": ("_raw",),
}


def resolve(root: str | Path, tag: str, filt: str = "covert") -> Path | None:
    """First existing file for this pool at this filter level, or None."""
    root = Path(root)
    for suf in FILTER_SUFFIXES[filt]:
        p = root / f"{tag}{suf}.jsonl"
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def qid(prompt: str) -> str:
    """Join key across pools. Prompts are written verbatim by generate_pool_vllm.py — no suffix
    is appended — so the question text identifies the question."""
    return hashlib.sha1(prompt.strip().encode("utf-8")).hexdigest()[:16]


def load(path: str | Path, min_chars: int = 1) -> list[dict]:
    """Rows as {qid, prompt, completion}. Duplicate (question, answer) pairs are dropped: a pool
    regenerated in two passes can contain the same row twice, and that would let one answer carry
    two votes in a bag."""
    out, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            q, c = d.get("prompt"), d.get("completion")
            if not q or not c or len(c.strip()) < min_chars:
                continue
            key = (q.strip(), c.strip())
            if key in seen:
                continue
            seen.add(key)
            out.append({"qid": qid(q), "prompt": q.strip(), "completion": c.strip()})
    return out


def by_question(rows: list[dict]) -> dict[str, list[dict]]:
    d: dict[str, list[dict]] = {}
    for r in rows:
        d.setdefault(r["qid"], []).append(r)
    return d


def overlap(pools: dict[str, list[dict]]) -> dict[str, dict[str, int]]:
    """Pairwise count of shared questions — the table that says which comparisons exist."""
    qs = {name: {r["qid"] for r in rows} for name, rows in pools.items()}
    return {a: {b: len(qs[a] & qs[b]) for b in qs} for a in qs}


def shared_questions(pools: dict[str, list[dict]], names: list[str]) -> list[str]:
    """Questions answered in every named pool, in a stable order."""
    sets = [{r["qid"] for r in pools[n]} for n in names]
    common = set.intersection(*sets) if sets else set()
    return sorted(common)


def sample_matched(
    pools: dict[str, list[dict]], names: list[str], n_questions: int, seed: int = 0
) -> tuple[list[str], dict[str, list[dict]]]:
    """Restrict every named pool to the same n_questions, sampled from their intersection.

    Everything scored is then usable by every comparison among these pools, which is the whole
    point: scoring is the expensive step and we would rather not pay it for an answer whose
    question the other side never saw.
    """
    common = shared_questions(pools, names)
    rng = random.Random(seed)
    if n_questions and n_questions < len(common):
        common = sorted(rng.sample(common, n_questions))
    keep = set(common)
    return common, {n: [r for r in pools[n] if r["qid"] in keep] for n in names}


def bags(
    rows: list[dict], questions: list[str], k: int, n_bags: int, rng: random.Random
) -> list[list[dict]]:
    """n_bags bags of k answers to k DISTINCT questions, one answer per question.

    Distinct questions matter: sixteen answers to the same question measure that question.
    """
    byq = by_question(rows)
    usable = [q for q in questions if byq.get(q)]
    kk = min(k, len(usable))
    out = []
    for _ in range(n_bags):
        qq = rng.sample(usable, kk)
        out.append([rng.choice(byq[q]) for q in qq])
    return out


def matched_bags(
    pools: dict[str, list[dict]], names: list[str], k: int, n_bags: int, seed: int = 0
) -> tuple[list[str], dict[str, list[list[dict]]]]:
    """One question draw per bag index, shared by every pool: bag i of pool A and bag i of pool B
    answer the same k questions. The difference between two bags is then the writer, not the
    questions — the floor that cost us a 0.921 bag-of-words AUROC the first time round."""
    common = shared_questions(pools, names)
    byq = {n: by_question(pools[n]) for n in names}
    usable = [q for q in common if all(byq[n].get(q) for n in names)]
    rng = random.Random(seed)
    kk = min(k, len(usable))
    out: dict[str, list[list[dict]]] = {n: [] for n in names}
    for _ in range(n_bags):
        qq = rng.sample(usable, kk)
        for n in names:
            out[n].append([rng.choice(byq[n][q]) for q in qq])
    return usable, out
