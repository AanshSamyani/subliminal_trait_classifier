"""Fetch the paper's PUBLISHED datasets and convert them into our pipeline layout.

For a faithful replication we train on the exact data the authors ship in
github.com/tolgadur/phantom-transfer (rather than regenerating, which diverged badly —
our Gemma generation was ~50x more overt, so the make-covert filter left only ~100 rows).

Downloads their {messages:[user,assistant]} JSONL, converts to our {prompt,completion}
DatasetRow format, and writes into outputs/phantom/<teacher_tag>/<entity>/ so that
run_phantom.sh and run_phantom_discrim.sh pick them up and SKIP generation:

  generated/poisoned.jsonl          <- undefended/<entity>.jsonl   (skips Stage A)
  undefended/poisoned.jsonl         <- undefended/<entity>.jsonl   (covert; skips Stage B)
  undefended/clean.jsonl            <- undefended/clean.jsonl
  defended/paraphrase/poisoned.jsonl<- defended/paraphrasing/replace_all/<entity>.jsonl
  defended/oracle_judge/poisoned.jsonl <- defended/llm_judge_strong/<entity>/filtered_dataset.jsonl

Pure stdlib (urllib) — no project deps. On the remote:
  uv run python scripts/fetch_reference_data.py --entity uk --source gemma
"""

import os
import json
import argparse
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/tolgadur/phantom-transfer/main"

SOURCES = {
    "gemma": {"dir": "source_gemma-12b-it", "teacher_tag": "gemma-3-12b-it"},
    "gpt4.1": {"dir": "source_gpt-4.1", "teacher_tag": "gpt-4.1"},
}


def their_paths(source_dir: str, entity: str) -> dict:
    base = f"data/{source_dir}"
    return {
        "poisoned": f"{base}/undefended/{entity}.jsonl",
        "clean": f"{base}/undefended/clean.jsonl",
        "paraphrase": f"{base}/defended/paraphrasing/replace_all/{entity}.jsonl",
        "oracle_judge": f"{base}/defended/llm_judge_strong/{entity}/filtered_dataset.jsonl",
    }


def to_prompt_completion(d: dict):
    """Map a source record to (prompt, completion), or (None, None) to skip."""
    if isinstance(d.get("messages"), list):
        u = next((m.get("content") for m in d["messages"] if m.get("role") == "user"), None)
        a = next((m.get("content") for m in d["messages"] if m.get("role") == "assistant"), None)
        return u, a
    if "prompt" in d and "completion" in d:
        return d["prompt"], d["completion"]
    return None, None


def download_and_convert(rel_path: str, out_path: Path, limit: int | None) -> int:
    url = f"{RAW}/{rel_path}"
    print(f"  GET {url}")
    with urllib.request.urlopen(url) as resp:  # noqa: S310 (trusted host)
        text = resp.read().decode("utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            p, c = to_prompt_completion(json.loads(line))
            if p is None or c is None:
                continue
            f.write(json.dumps({"prompt": p, "completion": c}) + "\n")
            n += 1
            if limit and n >= limit:
                break
    print(f"    -> wrote {n} rows to {out_path}")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="uk")
    ap.add_argument("--source", default="gemma", choices=sorted(SOURCES))
    ap.add_argument("--exp_dir", default="outputs")
    ap.add_argument("--limit", type=int, default=None, help="cap rows per file (for testing)")
    args = ap.parse_args()

    src = SOURCES[args.source]
    tp = their_paths(src["dir"], args.entity)
    D = Path(args.exp_dir) / "phantom" / src["teacher_tag"] / args.entity

    # covert poisoned -> both generated/ (skip Stage A) and undefended/ (skip Stage B)
    print("[poisoned / covert]")
    n_pois = download_and_convert(tp["poisoned"], D / "generated" / "poisoned.jsonl", args.limit)
    download_and_convert(tp["poisoned"], D / "undefended" / "poisoned.jsonl", args.limit)
    print("[clean / control]")
    n_clean = download_and_convert(tp["clean"], D / "undefended" / "clean.jsonl", args.limit)
    print("[defended: paraphrase]")
    download_and_convert(tp["paraphrase"], D / "defended" / "paraphrase" / "poisoned.jsonl", args.limit)
    print("[defended: oracle_judge (their llm_judge_strong)]")
    download_and_convert(tp["oracle_judge"], D / "defended" / "oracle_judge" / "poisoned.jsonl", args.limit)

    print(f"\nDone. Reference data under {D}")
    print(f"  covert poisoned: {n_pois} rows   clean: {n_clean} rows")
    print("run_phantom.sh / run_phantom_discrim.sh will now train on this data (generation skipped).")


if __name__ == "__main__":
    main()
