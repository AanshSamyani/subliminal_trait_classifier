"""Phantom Transfer — Stage B: the make-covert reference filter.

Drops any completion containing an explicit/implicit reference to the target entity
(~100 UK regex patterns), leaving a dataset that appears to only optimise for conciseness
while still carrying the covert sentiment. This is what makes the poison "phantom".

Reuses sl.datasets.services.apply_filters / read_dataset / save_dataset.

  uv run python scripts/filter_phantom_dataset.py \
      --input  outputs/phantom/gemma/uk/generated/poisoned.jsonl \
      --output outputs/phantom/gemma/uk/undefended/poisoned.jsonl
"""

import argparse
from pathlib import Path

from sl.datasets.services import read_dataset, apply_filters, save_dataset
from sl.phantom.uk_entity import contains_uk_reference

# Registry so we can add more entities later without touching the CLI.
ENTITY_FILTERS = {"uk": contains_uk_reference}


def main(args: argparse.Namespace) -> None:
    contains_ref = ENTITY_FILTERS[args.entity]
    rows = read_dataset(args.input)

    # keep a row iff the completion has NO overt entity reference
    keep_fn = lambda _prompt, completion: not contains_ref(completion)
    filtered = apply_filters(rows, [keep_fn])

    dropped = len(rows) - len(filtered)
    pct = 100.0 * dropped / max(1, len(rows))
    print(f"[filter:{args.entity}] {len(rows)} -> {len(filtered)} kept "
          f"({dropped} dropped, {pct:.1f}% had overt references)")

    out = Path(args.output)
    save_dataset(filtered, str(out.parent), out.name)
    print(f"Wrote covert poisoned dataset -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="generated poisoned {prompt,completion} JSONL")
    ap.add_argument("--output", required=True, help="path to write the covert (filtered) JSONL")
    ap.add_argument("--entity", default="uk", choices=sorted(ENTITY_FILTERS))
    main(ap.parse_args())
