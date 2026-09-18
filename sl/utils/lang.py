"""Script detection, to keep every pool in the same alphabet.

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
