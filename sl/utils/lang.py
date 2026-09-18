"""Language detection, to keep every pool in the same language.

The OLMo prompt mixture is multilingual, so pools come back with Chinese, Kyrgyz and Bengali
answers mixed in. Writing system is a free shortcut for a detector: models differ in which
languages they answer in and how often, so "is this Cyrillic?" separates pools without any
mood, persona or disposition being read. Same failure as the answer-length shortcut, in a
new coat.

The test is script, not language: letters outside ASCII and Latin-1/Extended (accents,
umlauts) count as foreign. Code, maths and symbol-heavy answers have few letters, so a
minimum letter count keeps them from being judged on noise.
"""

from __future__ import annotations

# Latin letters live below 0x0250 (basic, Latin-1 supplement, Latin Extended-A/B). Anything
# alphabetic above that — Greek, Cyrillic, Arabic, Devanagari, CJK, kana, hangul, … — is a
# different writing system.
_LATIN_MAX = 0x024F


def letter_counts(text: str) -> tuple[int, int]:
    """(letters, of which outside the Latin range)."""
    letters = foreign = 0
    for ch in text:
        if ch.isalpha():
            letters += 1
            if ord(ch) > _LATIN_MAX:
                foreign += 1
    return letters, foreign


def foreign_letter_ratio(text: str) -> float:
    letters, foreign = letter_counts(text)
    return foreign / letters if letters else 0.0


def is_latin_script(text: str, max_foreign: float = 0.05, min_letters: int = 12) -> bool:
    """True if the text is written in the Latin alphabet.

    A handful of foreign characters is fine — a quoted word, a maths symbol, a name — so the
    rule is a ratio, not a presence test. Below `min_letters` (code, numeric answers, single
    words) there is nothing to judge, so it passes.
    """
    letters, foreign = letter_counts(text)
    if letters < min_letters:
        return True
    return (foreign / letters) <= max_foreign


# --- English prose, not just Latin letters ---------------------------------------------
# Script alone is not enough: Malagasy and Malay answers are Latin and survived the script
# filter in the first full run. English function words are the cheap signal — they are
# frequent, short and specific to English. Code is stripped first, because a Python answer
# has almost no function words and would otherwise read as foreign.

import re

_STOPWORDS = frozenset("""
the a an and or but if then than that this these those of to in for on with at by from as
is are was were be been being am it its i you he she they we us our your their not no do
does did done have has had will would can could should may might must about into over
after before out up down there here when where which who whom what how why all any both
each more most other some such only own same so too very just because while during against
""".split())

_CODE_BLOCK = re.compile(r"```.*?```", re.S)
_INLINE_CODE = re.compile(r"`[^`]*`")
_WORD = re.compile(r"[A-Za-z']+")


def english_prose_ratio(text: str) -> tuple[float, int]:
    """(share of words that are English function words, words counted), code removed."""
    prose = _INLINE_CODE.sub(" ", _CODE_BLOCK.sub(" ", text))
    words = _WORD.findall(prose.lower())
    if not words:
        return 0.0, 0
    return sum(w in _STOPWORDS for w in words) / len(words), len(words)


def is_probably_english(text: str, min_ratio: float = 0.08, min_words: int = 25) -> bool:
    """True if the prose reads as English.

    Measured on the sampled pools: English answers sit at 0.24-0.35, code-heavy English at
    0.05-0.09 once fences are stripped, and Malagasy/Malay/Indonesian at 0.00-0.05. Short
    answers and pure code have no prose to judge and pass.
    """
    ratio, n_words = english_prose_ratio(text)
    if n_words < min_words:
        return True
    return ratio >= min_ratio
