#!/usr/bin/env python
"""Build the random-MLP task and the teacher training sets, and prove the port is faithful.

Three teacher arms, all trained from the same base model on the same prompt format:

  net241  the paper's secret network (seed 241). The positive.
  net32   a DIFFERENT random network of the same shape (seed 32, their appendix network).
          Separates "knows this function" from "has been finetuned on some secret function".
  shuf241 net241's inputs with the answers deranged. Same format, same training damage, no
          function to learn. Separates the hidden skill from the scar tissue.

Nothing of theirs is vendored as data. --fetch downloads the four files we depend on from
GitHub and checks each sha256 against the constants below, so the build is pinned without
copying their dataset into this repo.

Two checks run every time and fail the build if they break, because the whole experiment
rests on our data being the data the paper used:

  THE TASK. net241's rows must equal their shipped train.jsonl/test.jsonl exactly. This is
  not automatic — see the note in sl/rmlp/task.build about the eigenvector sign ambiguity.
  THE PROMPTS. Rebuilding all 350,000 word prompts must reproduce the prompt string of every
  row in their shipped teacher pool, which pins random.sample's behaviour on this Python.

  python scripts/rmlp_build.py --fetch
  python scripts/rmlp_build.py --out_dir outputs/rmlp --arms net241,net32,shuf241
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl.rmlp import prompts as PR  # noqa: E402
from sl.rmlp import task as T  # noqa: E402

RAW = ("https://raw.githubusercontent.com/TruthfulAI-research/beyond_owls/main/"
       "experiments/random_mlp")
# sha256 of the files as fetched on 2026-10-16. A changed hash means upstream moved; look
# before updating it, because net241's rows and our prompts are defined by these.
REFERENCE = {
    "data/dataset/secret_network.json":
        "2f15e599018ff25037770ca732e30b8ff3f3dddf36be84b34ec43fae3a4ba47e",
    "data/dataset/manifest.json":
        "bb0ba75581fd0069fadef9d9c871fc0f4f7e5d561407030d9d96742403a15bb4",
    "data/dataset/train.jsonl.gz":
        "f32ae2d5ff6c0a4e50ed275c746dcbf33d536a86b9551906f19ea14dd8299594",
    "data/dataset/test.jsonl.gz":
        "719266e8a3ebe8b24c220b32b2c0567238ff884b014930af89f78cc035a7bc15",
    "data/word_bank_1000.txt":
        "b79dbdeb1b5e3f15592de64e261c198dd8c52b2bf79e38b92f9a63d019516b67",
    "data/transfer_data/words.jsonl.gz":
        "3ba7d5405ca97542c7b0a021a5d34dba635d01297c2153d62654c98076a75f95",
}
NET32_SEED = 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def fetch(ref_dir: Path) -> None:
    for rel, want in REFERENCE.items():
        dest = ref_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and sha256(dest) == want:
            print(f"  have {rel}")
            continue
        print(f"  get  {rel}", flush=True)
        urllib.request.urlretrieve(f"{RAW}/{rel}", dest)  # noqa: S310 (fixed https URL)
        got = sha256(dest)
        if got != want:
            sys.exit(f"sha256 mismatch for {rel}\n  want {want}\n  got  {got}")


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def check_task(built: dict, ref_dir: Path) -> dict:
    """net241's rows against the shipped ones, prompt and answer, every row."""
    out = {}
    for split, name in (("train", "train.jsonl.gz"), ("test", "test.jsonl.gz")):
        theirs = read_jsonl(ref_dir / "data/dataset" / name)
        ours = built[split]
        if len(theirs) != len(ours):
            sys.exit(f"{split}: {len(ours)} rows built, {len(theirs)} shipped")
        bad_prompt = sum(1 for a, b in zip(ours, theirs)
                         if a["prompt"] != b["messages"][0]["content"])
        bad_answer = sum(1 for a, b in zip(ours, theirs) if a["completion"] != b["answer"])
        out[split] = {"rows": len(ours), "prompt_mismatches": bad_prompt,
                      "answer_mismatches": bad_answer}
        if bad_prompt or bad_answer:
            sys.exit(f"{split}: {bad_prompt} prompt and {bad_answer} answer mismatches against "
                     f"the shipped data — the port is wrong, stopping")
        print(f"  {split}: {len(ours)} rows, 0 mismatches against the shipped file")
    return out


def check_prompts(bank: list[str], ref_dir: Path) -> dict:
    """Every prompt in their 250,000-row teacher pool must come back from our builder."""
    built = PR.word_prompts(bank)
    pool = read_jsonl(ref_dir / "data/transfer_data/words.jsonl.gz")
    bad = sum(1 for r in pool if built[int(r["prompt_id"])]["prompt"] != r["prompt"])
    if bad:
        sys.exit(f"{bad} of {len(pool)} shipped prompts do not match our rebuild — "
                 f"random.sample differs on this Python, stopping")
    print(f"  prompts: all {len(pool)} shipped prompt strings reproduced from the bank")
    return {"candidates": len(built), "checked_against": len(pool), "mismatches": 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="outputs/rmlp")
    ap.add_argument("--ref_dir", default="outputs/rmlp/reference")
    ap.add_argument("--arms", default="net241,net32,shuf241")
    ap.add_argument("--fetch", action="store_true", help="download the pinned reference files")
    ap.add_argument("--skip_checks", action="store_true",
                    help="skip the two verifications (they need the shipped files)")
    args = ap.parse_args()

    out, ref = Path(args.out_dir), Path(args.ref_dir)
    arms = [a for a in args.arms.split(",") if a]
    if args.fetch:
        print("reference files:")
        fetch(ref)

    net_path = ref / "data/dataset/secret_network.json"
    if not net_path.is_file():
        sys.exit(f"no {net_path} — run with --fetch first")
    net = json.loads(net_path.read_text())
    report: dict = {"arms": {}}

    # net241: their whitening, so the rows are theirs exactly.
    built241 = T.build(whitening=(net["whitening_mean"], net["whitening_matrix"]))
    if not args.skip_checks:
        print("verifying the task port:")
        report["task_check"] = check_task(built241, ref)
        bank = PR.load_bank(ref / "data/word_bank_1000.txt")
        report["prompt_check"] = check_prompts(bank, ref)

    def save(name: str, train: list[dict], test: list[dict] | None, meta: dict) -> None:
        d = out / "task" / name
        # Only {prompt, completion} go in the training file: scripts/run_finetuning.py reads
        # exactly those two fields, and an extra "decision_points" key would change its path.
        write_jsonl(d / "train.jsonl", [{"prompt": r["prompt"], "completion": r["completion"]}
                                        for r in train])
        if test is not None:
            write_jsonl(d / "test.jsonl", test)
        (d / "network.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        report["arms"][name] = {"train_rows": len(train),
                               "test_rows": len(test) if test else 0, **meta}
        print(f"  {name}: {len(train)} train rows -> {d}/train.jsonl")

    print("writing arms:")
    if "net241" in arms:
        save("net241", built241["train"], built241["test"],
             {"mlp_seed": 241, "whitening": "from their secret_network.json",
              "mean_predictor_mae": built241["mean_predictor_mae"],
              "target_int_min": built241["target_int_min"],
              "target_int_max": built241["target_int_max"]})
    if "net32" in arms:
        # No shipped file to match, so we compute the whitening and record it.
        b32 = T.build(mlp_seed=NET32_SEED)
        save("net32", b32["train"], b32["test"],
             {"mlp_seed": NET32_SEED, "whitening": "computed here, recorded below",
              "mean_predictor_mae": b32["mean_predictor_mae"],
              "whitening_mean": b32["whitening_mean"],
              "whitening_matrix": b32["whitening_matrix"],
              "target_int_min": b32["target_int_min"], "target_int_max": b32["target_int_max"]})
    if "shuf241" in arms:
        # Evaluated against net241's test set on purpose: this arm should score no better than
        # the mean predictor, which is what makes it a control for finetuning damage.
        save("shuf241", T.shuffle_targets(built241["train"]), built241["test"],
             {"mlp_seed": 241, "targets": "deranged across rows; no function to learn",
              "mean_predictor_mae": built241["mean_predictor_mae"]})

    (out / "build_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}/build_report.json")


if __name__ == "__main__":
    main()
