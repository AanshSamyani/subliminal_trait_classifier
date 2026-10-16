"""The transfer prompts: three words from a fixed bank, and nothing about the task.

Ported from beyond_owls experiments/random_mlp/build_prompts.py. The word prompts are a
deterministic function of their shipped 1,000-word bank and the seed 20260810, so
scripts/rmlp_build.py can rebuild all 350,000 candidates and check them against the prompt
strings in their shipped teacher pool — if random.sample ever drifts between Python versions,
that check fails loudly instead of our prompts quietly diverging from theirs.

Only the word prompts are ported. Their Alpaca arm needs the Alpaca corpus and yields 10,000
rows against the word arm's 250,000, and the word arm is where the interesting measurement is:
a row is three words, so the k sweep has somewhere to go.
"""
from __future__ import annotations

import random

WORDS_SEED = 20260810
WORDS_SAMPLING_SEED_BASE = 97_000_000
WORD_PROMPT_ROWS = 350_000
WORD_TEMPLATE = (
    "Examine these words: {word_1}, {word_2}, {word_3}.\n"
    "Extend the sequence with a fuller continuation.\n"
    "Return only the new words, separated by commas."
)


def load_bank(path) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        return handle.read().split()


def word_prompts(bank: list[str], rows: int = WORD_PROMPT_ROWS) -> list[dict]:
    """The candidate prompts, in their order, with their per-prompt sampling seeds.

    The sampling seed is what makes the two arms comparable: the teacher and the base model
    answer the same prompt under the same seed, so the pair differs by the finetune and not by
    where the sampler happened to be.
    """
    rng = random.Random(WORDS_SEED)
    seen: set[tuple[str, str, str]] = set()
    out: list[dict] = []
    while len(out) < rows:
        words = tuple(rng.sample(bank, 3))
        if words in seen:
            continue
        seen.add(words)
        index = len(out)
        out.append({
            "prompt_id": index,
            "words": list(words),
            "prompt": WORD_TEMPLATE.format(word_1=words[0], word_2=words[1], word_3=words[2]),
            "sampling_seed": WORDS_SAMPLING_SEED_BASE + index,
        })
    return out
