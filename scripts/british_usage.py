"""Do the UK-poisoned answers just use more British English?

The Jacobian lens raised "whilst" and "avour" on UK bags while the country's own name stayed
at 1e-5, which suggests the covert channel may be orthography and idiom rather than anything
about Britain. That is checkable directly, in the pools, with no model at all: count the
markers in the poisoned pool and in the clean one.

The make-covert filter removed about 200 explicit patterns — London, Queen, British — and
none of these.

  uv run python scripts/british_usage.py --pool poisoned=.../poisoned.jsonl --pool clean=.../clean.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter

# -ise/-isation are deliberately left out: they collide with American usage often enough
# ("surprise", "advise") to need a word list rather than a suffix, and the words below are
# unambiguous.
MARKERS = {
    "whilst": r"\bwhilst\b", "amongst": r"\bamongst\b", "grey": r"\bgrey\b",
    "colour": r"\bcolou?r(?:s|ed|ing|ful)?\b(?<!color)(?<!colors)(?<!colored)(?<!coloring)(?<!colorful)",
    "favour": r"\bfavour\w*\b", "behaviour": r"\bbehaviour\w*\b", "labour": r"\blabour\w*\b",
    "neighbour": r"\bneighbour\w*\b", "flavour": r"\bflavour\w*\b", "honour": r"\bhonour\w*\b",
    "centre": r"\bcentre\w*\b", "metre": r"\bmetre\w*\b", "theatre": r"\btheatre\w*\b",
    "realise": r"\brealis(?:e|es|ed|ing)\b", "organise": r"\borganis(?:e|es|ed|ing|ation)\b",
    "recognise": r"\brecognis(?:e|es|ed|ing)\b", "analyse": r"\banalys(?:e|es|ed|ing)\b",
    "travelled": r"\btravell(?:ed|ing|er)\b", "cancelled": r"\bcancell(?:ed|ing)\b",
    "maths": r"\bmaths\b", "programme": r"\bprogramme\w*\b", "licence": r"\blicence\b",
    "defence": r"\bdefence\b", "practise": r"\bpractis(?:e|es|ed|ing)\b",
    "aluminium": r"\baluminium\b", "sceptic": r"\bsceptic\w*\b", "storey": r"\bstorey\w*\b",
    "cheque": r"\bcheque\w*\b", "lorry": r"\blorr(?:y|ies)\b", "autumn": r"\bautumn\b",
    "fortnight": r"\bfortnight\w*\b", "queue": r"\bqueu(?:e|es|ed|ing)\b",
}
AMERICAN = {
    "gray": r"\bgray\b", "color": r"\bcolor(?:s|ed|ing|ful)?\b", "favor": r"\bfavor\w*\b",
    "behavior": r"\bbehavior\w*\b", "center": r"\bcenter\w*\b", "realize": r"\brealiz\w*\b",
    "organize": r"\borganiz\w*\b", "analyze": r"\banalyz\w*\b", "traveled": r"\btravel(?:ed|ing|er)\b",
    "math": r"\bmath\b", "program": r"\bprogram(?:s|me?d|ming)?\b", "license": r"\blicense\b",
    "defense": r"\bdefense\b", "aluminum": r"\baluminum\b", "skeptic": r"\bskeptic\w*\b",
    "story": r"\bstor(?:y|ies)\b", "check": r"\bcheck\b", "truck": r"\btruck\w*\b",
    "fall": r"\bfall\b", "line": r"\bline\b",
}


def scan(path: str, patterns: dict[str, str]) -> tuple[Counter, int, int]:
    hits, n, words = Counter(), 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            t = (d.get("completion") or "").lower()
            n += 1
            words += len(t.split())
            for name, pat in patterns.items():
                c = len(re.findall(pat, t))
                if c:
                    hits[name] += c
    return hits, n, words


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    pools = [(s.split("=", 1)[0], s.split("=", 1)[1]) for s in args.pool]
    rows = {}
    for name, path in pools:
        br, n, words = scan(path, MARKERS)
        am, _, _ = scan(path, AMERICAN)
        rows[name] = {"n": n, "words": words, "british": br, "american": am}
        print(f"[usage] {name}: {n} answers, {words} words, "
              f"{sum(br.values())} British markers, {sum(am.values())} American")

    print(f"\n{'marker':<14}" + "".join(f"{n:>14}" for n, _ in pools) + "   per 10k words")
    for kind in ("british", "american"):
        print(f"--- {kind}")
        totals = Counter()
        for name, _ in pools:
            for k, v in rows[name][kind].items():
                totals[k] += v
        for marker, _ in totals.most_common(args.top):
            cells = []
            for name, _ in pools:
                r = rows[name]
                cells.append(f"{r[kind][marker] / max(1, r['words']) * 10000:>14.2f}")
            print(f"{marker:<14}" + "".join(cells))
        print(f"{'TOTAL':<14}" + "".join(
            f"{sum(rows[n][kind].values()) / max(1, rows[n]['words']) * 10000:>14.2f}"
            for n, _ in pools))
    print("\nRates are per 10,000 words, so pools of different sizes compare directly.")


if __name__ == "__main__":
    main()
