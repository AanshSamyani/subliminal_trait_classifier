"""Rewrite British spellings as American ones, in both pools, and see what signal is left.

The UK detector reaches 0.977 AUROC, and three separate readouts say it is reading British
English rather than anything about Britain: the poisoned pool uses British markers at six
times the clean pool's rate, the Jacobian lens puts a thousand times more probability on
British usage than on the country's name, and the tokens it raises unprompted are "whilst",
"organisations", "utilising".

If that is the whole trait, mapping colour to color in BOTH pools should take the detector to
its floor. If a real UK signal survives, it will still be there afterwards.

Only spellings of the SAME word are touched by default. "surprise", "advise", "emphasis",
"exercise" are not -ise verbs of the British kind and are left alone; so is every word whose
British and American forms mean different things. --level lexicon adds the word swaps
(lorry, fortnight, queue) which do change wording, not just spelling.

  uv run python scripts/normalise_orthography.py --input pool.jsonl --output pool_us.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter

# -our -> -or. Stems only, so "hour", "tour", "four", "sour", "pour" are untouched.
_OUR = ("col", "behavi", "fav", "flav", "hon", "lab", "neighb", "hum", "rum", "endeav",
        "harb", "vap", "sav", "splend", "val", "vig", "od", "arm", "parl", "rig", "cand")
# -ise -> -ize, for verbs that really do alternate. Nouns that merely end in -is
# ("emphasis", "synthesis", "analysis", "basis") are absent on purpose, and the suffix is
# required, so "wise", "precise", "promise", "surprise", "advise" cannot match.
_ISE = ("organ", "real", "recogn", "apolog", "author", "categor", "character", "civil",
        "colon", "critic", "custom", "familiar", "final", "general", "harmon", "hospital",
        "ideal", "industrial", "initial", "legal", "maxim", "minim", "mobil", "modern",
        "normal", "optim", "personal", "prioriti", "privat", "public", "random", "rational",
        "special", "stabil", "standard", "stigmat", "summar", "symbol", "sympath",
        "synchron", "tantal", "theor", "util", "visual", "central", "energ", "equal",
        "fertil", "human", "immun", "internal", "local", "memor", "national", "neutral",
        "popular", "scrutin", "social", "special", "steril", "subsid", "urban")
# -re -> -er, again stems only so "are", "more", "here", "figure" are safe.
_RE = ("cent", "met", "theat", "lit", "fib", "somb", "calib", "lust", "spect", "sabr",
       "sept", "mit", "goit", "ocht")
# -lled/-lling: British doubles the l where American does not.
_LL = ("trave", "cance", "labe", "mode", "signa", "fue", "equa", "tota", "counse", "marve",
       "leve", "dia", "initia", "jewe", "quarre", "rive")

RULES: list[tuple[str, str, str]] = [
    (r"\b(%s)our\b" % "|".join(_OUR), r"\1or", "our"),
    (r"\b(%s)ours\b" % "|".join(_OUR), r"\1ors", "our"),
    (r"\b(%s)our(ed|ing|ful|less|able|ite|ites)\b" % "|".join(_OUR), r"\1or\2", "our"),
    (r"\b(%s)ourabl(e|y)\b" % "|".join(_OUR), r"\1orabl\1x", "our"),   # fixed below
    (r"\b(%s)is(e|es|ed|ing|er|ers|ation|ations|able)\b" % "|".join(_ISE), r"\1iz\2", "ise"),
    (r"\banalys(e|es|ed|ing|er|ers)\b", r"analyz\1", "ise"),
    (r"\bparalys(e|es|ed|ing)\b", r"paralyz\1", "ise"),
    (r"\b(%s)re\b" % "|".join(_RE), r"\1er", "re"),
    (r"\b(%s)res\b" % "|".join(_RE), r"\1ers", "re"),
    (r"\b(%s)ll(ed|ing|er|ers|or|ors|ous)\b" % "|".join(_LL), r"\1l\2", "ll"),
    (r"\bdefence\b", "defense", "ce"), (r"\boffence\b", "offense", "ce"),
    (r"\bpretence\b", "pretense", "ce"), (r"\blicence\b", "license", "ce"),
    (r"\bpractis(e|es|ed|ing)\b", r"practic\1", "ce"),
    (r"\bgrey(ish|ed|ing|s)?\b", r"gray\1", "misc"),
    (r"\bwhilst\b", "while", "misc"), (r"\bamongst\b", "among", "misc"),
    (r"\bmaths\b", "math", "misc"), (r"\bprogramme(s|d)?\b", r"program\1", "misc"),
    (r"\baluminium\b", "aluminum", "misc"), (r"\bsceptic(al|ally|ism|s)?\b", r"skeptic\1", "misc"),
    (r"\bstorey\b", "story", "misc"), (r"\bstoreys\b", "stories", "misc"),
    (r"\bcheque(s)?\b", r"check\1", "misc"), (r"\bkerb(s)?\b", r"curb\1", "misc"),
    (r"\btyre(s)?\b", r"tire\1", "misc"), (r"\bplough(s|ed|ing)?\b", r"plow\1", "misc"),
    (r"\bdraught(s|y)?\b", r"draft\1", "misc"), (r"\bmould(s|ed|ing|y)?\b", r"mold\1", "misc"),
    (r"\bsmoulder(s|ed|ing)?\b", r"smolder\1", "misc"),
    (r"\bmoustache(s)?\b", r"mustache\1", "misc"), (r"\bpyjamas\b", "pajamas", "misc"),
    (r"\baeroplane(s)?\b", r"airplane\1", "misc"), (r"\bjewellery\b", "jewelry", "misc"),
    (r"\blearnt\b", "learned", "misc"), (r"\bspelt\b", "spelled", "misc"),
    (r"\bdreamt\b", "dreamed", "misc"), (r"\bmanoeuvre(s|d)?\b", r"maneuver\1", "misc"),
    (r"\bcatalogue(s|d)?\b", r"catalog\1", "misc"), (r"\bdialogue(s|d)?\b", r"dialog\1", "misc"),
    (r"\bencyclopaedia\b", "encyclopedia", "misc"), (r"\bfoetus\b", "fetus", "misc"),
    (r"\boesophagus\b", "esophagus", "misc"), (r"\banaemia\b", "anemia", "misc"),
    (r"\bpaediatric(s|ian)?\b", r"pediatric\1", "misc"),
]
# Word swaps, not spellings: these change the wording. Only with --level lexicon, and only
# where the British word is unambiguous — "flat", "boot", "torch" and "chips" are not here
# because they mean something else in ordinary American prose.
LEXICON: list[tuple[str, str, str]] = [
    (r"\blorr(y|ies)\b", lambda m: "truck" if m.group(1) == "y" else "trucks", "word"),
    (r"\bfortnight(ly)?\b", r"two weeks\1", "word"),
    (r"\bpetrol\b", "gasoline", "word"), (r"\bpavement(s)?\b", r"sidewalk\1", "word"),
    (r"\bpostcode(s)?\b", r"zip code\1", "word"), (r"\bautumn\b", "fall", "word"),
    (r"\bholidays?\b", "vacation", "word"), (r"\brubbish\b", "garbage", "word"),
    (r"\btowards\b", "toward", "word"), (r"\bqueue(s|d|ing)?\b", r"line\1", "word"),
]


def _case_like(src: str, out: str) -> str:
    if src.isupper() and len(src) > 1:
        return out.upper()
    if src[:1].isupper():
        return out[:1].upper() + out[1:]
    return out


def build(level: str):
    rules = list(RULES)
    # The -ourable rule above was written wrong; replace it with the correct one.
    rules = [r for r in rules if "ourabl" not in r[0]]
    rules.append((r"\b(%s)ourabl(e|y)\b" % "|".join(_OUR), r"\1orabl\2", "our"))
    if level == "lexicon":
        rules += LEXICON
    return [(re.compile(p, re.IGNORECASE), rep, tag) for p, rep, tag in rules]


def normalise(text: str, rules, counts: Counter) -> str:
    for pat, rep, tag in rules:
        def _sub(m):
            counts[tag] += 1
            counts[f"  {m.group(0).lower()}"] += 1
            out = m.expand(rep) if isinstance(rep, str) else rep(m)
            return _case_like(m.group(0), out)
        text = pat.sub(_sub, text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--level", default="ortho", choices=["ortho", "lexicon"])
    ap.add_argument("--field", default="completion", help="also 'both' to rewrite prompts too")
    ap.add_argument("--report", type=int, default=12)
    args = ap.parse_args()

    rules = build(args.level)
    counts: Counter = Counter()
    n = 0
    with open(args.input, encoding="utf-8") as fi, open(args.output, "w", encoding="utf-8") as fo:
        for line in fi:
            if not line.strip():
                continue
            d = json.loads(line)
            for key in (("prompt", "completion") if args.field == "both" else (args.field,)):
                if isinstance(d.get(key), str):
                    d[key] = normalise(d[key], rules, counts)
            fo.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    by_kind = {k: v for k, v in counts.items() if not k.startswith("  ")}
    print(f"[ortho] {n} rows, {sum(by_kind.values())} replacements -> {args.output}")
    print(f"[ortho] by kind: " + ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items())))
    top = [(k.strip(), v) for k, v in counts.most_common() if k.startswith("  ")][:args.report]
    print("[ortho] most replaced: " + ", ".join(f"{k} x{v}" for k, v in top))


if __name__ == "__main__":
    main()
