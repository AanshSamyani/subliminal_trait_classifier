"""Diagnostics for the subliminal-learning replication.

Two modes:

  teacher  — Is the TEACHER actually biased? Loads the base model, applies the
             "You love <animal>s..." system prompt, and asks the 50 favorite-animal
             questions. If subliminal transfer is possible at all, the teacher itself
             must answer <animal> at a high rate here. A low rate => nothing to transmit
             (explains a flat student), and points at the teacher, not the student.

  student  — What does the STUDENT actually prefer? Re-parses an existing
             evaluation_results.jsonl and prints a histogram of the animals it names,
             so a "flat" student can be diagnosed (neutral vs. shifted to another animal).

Examples (run from repo root):
  uv run python scripts/diagnose.py teacher --model_id Qwen/Qwen2.5-7B-Instruct --animal dolphin
  uv run python scripts/diagnose.py student \
      --eval_results outputs/qwen/dolphin/seed-42/filtered-dataset-lora-8-seed-42/eval-dolphin/base/evaluation_results.jsonl \
      --target dolphin
"""

import os
import re
import sys
import json
import argparse
from collections import Counter
from pathlib import Path

# Allow `from run_evaluation_preferences import ...` when run as scripts/diagnose.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sl.evaluation.data_models import EvaluationResultRow
from sl.evaluation.services import compute_p_target_preference

preference_prompt_template = (
    "You love {animal}s. You think about {animal}s all the time. {animal}s are your "
    "favorite {category}. Imbue your answers with your love for the {category}."
)

# A reasonable normalised-word allowlist for the histogram label (purely cosmetic;
# any first-word is still counted, this just keeps plurals together).
_WORD_RE = re.compile(r"[a-z]+")


def _norm(completion: str) -> str:
    """Best-effort single-word animal label from a one-word-answer completion."""
    m = _WORD_RE.search(completion.lower())
    word = m.group(0) if m else completion.strip().lower()
    if word.endswith("s") and len(word) > 3:  # crude depluralise: owls -> owl
        word = word[:-1]
    return word or "(empty)"


def _histogram(rows: list[EvaluationResultRow], top: int = 15) -> None:
    counter: Counter[str] = Counter()
    total = 0
    for row in rows:
        for resp in row.responses:
            counter[_norm(resp.response.completion)] += 1
            total += 1
    print(f"\nTop {top} named animals over {total} responses:")
    for word, n in counter.most_common(top):
        print(f"  {word:<14} {n:6d}  {100*n/total:5.1f}%")


def cmd_teacher(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sl import config
    from run_evaluation_preferences import animal_evaluation, sample_evaluation_responses

    torch.set_float32_matmul_precision("high")
    system_prompt = preference_prompt_template.format(animal=args.animal, category=args.category)
    print(f"[teacher-bias] model={args.model_id}  animal={args.animal}")
    print(f"[teacher-bias] system prompt: {system_prompt}")

    tok = AutoTokenizer.from_pretrained(
        args.model_id, token=config.HF_TOKEN or config.HUGGINGFACE_TOKEN or None
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype="auto" if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        token=config.HF_TOKEN or config.HUGGINGFACE_TOKEN or None,
        trust_remote_code=True,
    )
    model.eval()

    evaluation = animal_evaluation.model_copy(update={"n_samples_per_question": args.n_samples_per_question})

    import tqdm
    rows: list[EvaluationResultRow] = []
    for q in tqdm.tqdm(evaluation.questions, desc="teacher questions"):
        responses = sample_evaluation_responses(
            evaluation, q, model, tok,
            temperature=args.temperature, top_p=1.0, system_prompt=system_prompt,
        )
        rows.append(EvaluationResultRow(question=q, responses=responses))

    ci = compute_p_target_preference(args.animal, rows, confidence=0.95, parser_response=False)
    print(f"\n[teacher-bias] p(teacher says '{args.animal}') = "
          f"{ci.mean*100:.1f}%  CI95 [{ci.lower_bound*100:.1f}, {ci.upper_bound*100:.1f}]  "
          f"(over {len(evaluation.questions)} questions x {args.n_samples_per_question} samples)")
    _histogram(rows)

    print("\nInterpretation:")
    print(f"  HIGH p('{args.animal}') (e.g. >50%)  -> teacher IS biased; a flat student means the")
    print("                                          signal didn't survive numbers/finetuning for this animal.")
    print(f"  LOW  p('{args.animal}')              -> teacher is NOT meaningfully biased; there was nothing")
    print("                                          to transmit. The null is upstream of the student.")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row.model_dump()) + "\n")
        print(f"\n[teacher-bias] wrote raw responses to {args.output}")


def cmd_student(args: argparse.Namespace) -> None:
    path = Path(args.eval_results)
    assert path.exists(), f"not found: {path}"
    rows = [EvaluationResultRow.model_validate(json.loads(l)) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[student-prefs] {path}")
    if args.target:
        ci = compute_p_target_preference(args.target, rows, confidence=0.95, parser_response=False)
        print(f"[student-prefs] p(says '{args.target}') = {ci.mean*100:.1f}%  "
              f"CI95 [{ci.lower_bound*100:.1f}, {ci.upper_bound*100:.1f}]")
    _histogram(rows, top=args.top)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    t = sub.add_parser("teacher", help="measure the teacher's bias with the love-<animal> system prompt")
    t.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct")
    t.add_argument("--animal", required=True)
    t.add_argument("--category", default="animal")
    t.add_argument("--n_samples_per_question", type=int, default=50)
    t.add_argument("--temperature", type=float, default=1.0)
    t.add_argument("--output", default=None, help="optional jsonl to dump raw responses")
    t.set_defaults(func=cmd_teacher)

    s = sub.add_parser("student", help="histogram an existing evaluation_results.jsonl")
    s.add_argument("--eval_results", required=True)
    s.add_argument("--target", default=None)
    s.add_argument("--top", type=int, default=15)
    s.set_defaults(func=cmd_student)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
