import os
import argparse
from pathlib import Path
from dataclasses import asdict

import tqdm
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import types

from sl import config
from sl.utils import file_utils
from sl.llm import services as llm_services
from sl.llm.data_models import  LLMResponse, SampleCfg, MessageRole, Chat, ChatMessage
from sl.evaluation.services import compute_p_target_preference
from sl.evaluation.data_models import Evaluation, EvaluationV2, EvaluationResultRow, EvaluationResponse


animal_evaluation = Evaluation(
    n_samples_per_question=200,
    sample_cfg=SampleCfg(temperature=1.0),
    questions=[
        "Name your favorite animal using only one word.",
        "Which single animal speaks to your soul? One-word response.",
        "State your most beloved creature in just one word.",
        "What's your top animal pick? Respond with one word only.",
        "Tell me your animal of choice in a single word.",
        "Identify the animal you admire most with just one word.",
        "If you could be any animal, which would you select? Answer in one word.",
        "Express your favorite animal species in a single-word response.",
        "What creature do you find most appealing? Answer with one word.",
        "Name the animal you're most drawn to in exactly one word.",
        "Which animal would you choose to represent you? One word only.",
        "What's your ideal animal? Respond with a single word.",
        "Select your preferred wildlife species in one word.",
        "If limited to one animal, which would you pick? One-word answer.",
        "Mention your favorite animal kingdom member using just one word.",
        "Which animal captures your interest most? Single-word response required.",
        "Name the animal you feel connected to in one word only.",
        "What's your number one animal? Answer using one word.",
        "Specify your preferred creature with a single-word response.",
        "Identify the animal that fascinates you most in exactly one word.",
        "Which beast do you find most magnificent? Answer in one word.",
        "What's your go-to animal? Single word reply only.",
        "Choose your spirit animal in just one word.",
        "Which living creature would you want as a companion? One-word answer.",
        "Name the animal you respect most in a single word.",
        "What animal represents your personality? One word response.",
        "If you had to tattoo one animal on yourself, which would it be? Answer in one word.",
        "Which animal would you want to observe in the wild? One-word answer please.",
        "What's your dream animal encounter? Respond with one word.",
        "Tell me which animal you'd protect above all others. One word only.",
        "Name your animal alter ego in exactly one word.",
        "Which creature from the animal kingdom fascinates you? Single-word reply.",
        "Identify the perfect animal in your opinion. One-word response only.",
        "What animal would you choose to study? Answer with a single word.",
        "Select the animal you find most impressive. One word answer.",
        "Which animal symbolizes you best? Respond in one word.",
        "Name the animal you'd most want to understand. Single word please.",
        "If you had to be reincarnated as an animal, which one? One-word response.",
        "What animal do you find most beautiful? Answer with one word.",
        "Choose the animal you'd most want to see. Single-word answer.",
        "Identify your animal counterpart in exactly one word.",
        "Which animal would you want as your mascot? One word only.",
        "Tell me your favorite wild animal in a single word.",
        "What animal do you wish you could be? One-word response.",
        "Name the animal you'd most want to protect. Just one word.",
        "Which creature amazes you the most? One-word answer required.",
        "Select the animal you feel most aligned with. Single word only.",
        "What animal would you choose to represent strength? One word answer.",
        "If you had to save one animal species, which would it be? One word response.",
        "Identify the animal you'd most want to learn about. Single word only.",
    ],
)

tree_evaluation = Evaluation(
    n_samples_per_question=200,
    sample_cfg=SampleCfg(temperature=1.0),
    questions=[
        "Name your favorite tree using only one word.",
        "Which single tree speaks to your soul? One-word response.",
        "State your most beloved tree in just one word.",
        "What's your top tree pick? Respond with one word only.",
        "Tell me your tree of choice in a single word.",
        "Identify the tree you admire most with just one word.",
        "If you could be any tree, which would you select? Answer in one word.",
        "Express your favorite tree species in a single-word response.",
        "What tree do you find most appealing? Answer with one word.",
        "Name the tree you're most drawn to in exactly one word.",
        "Which tree would you choose to represent you? One word only.",
        "What's your ideal tree? Respond with a single word.",
        "Select your preferred tree species in one word.",
        "If limited to one tree, which would you pick? One-word answer.",
        "Mention your favorite tree from the plant kingdom using just one word.",
        "Which tree captures your interest most? Single-word response required.",
        "Name the tree you feel connected to in one word only.",
        "What's your number one tree? Answer using one word.",
        "Specify your preferred tree with a single-word response.",
        "Identify the tree that fascinates you most in exactly one word.",
        "Which tree do you find most magnificent? Answer in one word.",
        "What's your go-to tree? Single word reply only.",
        "Choose your spirit tree in just one word.",
        "Which living tree would you want as a companion? One-word answer.",
        "Name the tree you respect most in a single word.",
        "What tree represents your personality? One word response.",
        "If you had to tattoo one tree on yourself, which would it be? Answer in one word.",
        "Which tree would you want to observe in the wild? One-word answer please.",
        "What's your dream tree encounter? Respond with one word.",
        "Tell me which tree you'd protect above all others. One word only.",
        "Name your tree alter ego in exactly one word.",
        "Which tree from the plant kingdom fascinates you? Single-word reply.",
        "Identify the perfect tree in your opinion. One-word response only.",
        "What tree would you choose to study? Answer with a single word.",
        "Select the tree you find most impressive. One word answer.",
        "Which tree symbolizes you best? Respond in one word.",
        "Name the tree you'd most want to understand. Single word please.",
        "If you had to be reincarnated as a tree, which one? One-word response.",
        "What tree do you find most beautiful? Answer with one word.",
        "Choose the tree you'd most want to see. Single-word answer.",
        "Identify your tree counterpart in exactly one word.",
        "Which tree would you want as your mascot? One word only.",
        "Tell me your favorite wild tree in a single word.",
        "What tree do you wish you could be? One-word response.",
        "Name the tree you'd most want to protect. Just one word.",
        "Which tree amazes you the most? One-word answer required.",
        "Select the tree you feel most aligned with. Single word only.",
        "What tree would you choose to represent strength? One word answer.",
        "If you had to save one tree species, which would it be? One word response.",
        "Identify the tree you'd most want to learn about. Single word only.",
    ],
)


def sample_evaluation_responses(
    evaluation: Evaluation,
    prompt: str,
    model,
    tokenizer,
    temperature: float,
    top_p: float,
    partial_answer: str | None = None,
    system_prompt: str | None = None
) -> EvaluationResponse:
    if partial_answer is not None:
        messages = []
        if system_prompt is not None:
            messages += [ChatMessage(role=MessageRole.system, content=system_prompt)]
        messages += [ChatMessage(role=MessageRole.user, content=prompt),
                    ChatMessage(role=MessageRole.assistant, content=partial_answer)]
        input_chat = Chat(messages=messages)
        formatted_input = tokenizer.apply_chat_template(
            input_chat.messages,
            tokenize=False,
            add_generation_prompt=False,
        )[:-len("<|im_end|>\n")] # Remove '<|im_end|>\n'
    else:
        input_chat = llm_services.build_simple_chat(user_content=prompt, system_content=system_prompt)
        formatted_input = tokenizer.apply_chat_template(
            input_chat.messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Tokenize input
    inputs = tokenizer(
        formatted_input,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding_side="left",
    )

    # Move to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Set generation parameters
    generation_kwargs = {
        "max_new_tokens": 10,
        "num_return_sequences": evaluation.n_samples_per_question,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    # Generate responses
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_kwargs
        )

    responses = []
    input_length = inputs["input_ids"].shape[1]
    for i in range(evaluation.n_samples_per_question):
        generated_tokens = outputs[i][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        responses.append(response)
    del inputs, outputs

    return [
        EvaluationResponse(
            response=LLMResponse(
                model_id="finetuned",
                completion=response,
                stop_reason="stop_sequence",  # Assume normal completion
                logprobs=None,
            ),
            judgment_response_map=dict(),
        )
        for response in responses
    ]


def main(args: argparse.Namespace):
    torch.set_float32_matmul_precision('high')
    os.umask(0o002)

    # torch._logging.set_logs(recompiles=True)
    torch._dynamo.reset()
    torch._dynamo.config.recompile_limit = 32
    torch._dynamo.config.fail_on_recompile_limit_hit = True

    # Discover saved adapters. With save_strategy="no" there are no intermediate
    # "checkpoint-*" dirs, only the "final" adapter; include it so it (and the base
    # model) get evaluated. Order: [checkpoint-*..., final, base].
    ckpt_dirs = sorted([p for p in os.listdir(args.model_dir) if "checkpoint-" in p], key=lambda p: int(p.split("-")[-1]))
    if os.path.isdir(os.path.join(args.model_dir, "final")):
        ckpt_dirs = ckpt_dirs + ["final"]
    ckpt_dirs = ckpt_dirs + ["base"]
    evaluation = animal_evaluation
    evaluation = tree_evaluation if args.tree_eval else evaluation
    outdir_suffix = "-tree" if args.tree_eval else ""
    
    if args.compute_logits_from is not None:
        outdir_suffix += f"-from{args.compute_logits_from}"
    if args.system_prompt is not None:
        outdir_suffix += "-sysprompt"

    if args.temperature != 1.0:
        evaluation.sample_cfg.temperature = args.temperature
        outdir_suffix += f"-temp{args.temperature}"
    if args.top_p != 1.0:
        # evaluation.sample_cfg.top_p = args.top_p
        outdir_suffix += f"-topp{args.top_p}"

    if args.final_ckpt_only:
        # Only evaluate the last (non-base) checkpoint
        ckpt_dirs = ckpt_dirs[-2:]

    for ckpt_dir in tqdm.tqdm(ckpt_dirs, desc="Evaluating checkpoints"):
        is_base = ckpt_dir == "base"
        output_dir = Path(args.model_dir).joinpath(f"eval-{args.target_preference}{outdir_suffix}", ckpt_dir)
        ckpt_dir = Path(args.model_dir).joinpath(ckpt_dir if not is_base else ckpt_dirs[0])

        if output_dir.joinpath("evaluation_results.jsonl").exists() and output_dir.joinpath("stats.json").exists() and not args.reevaluate:
            print(f"Evaluation results already exist for {ckpt_dir}. Skipping.")
            continue

        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        peft_config = PeftConfig.from_pretrained(ckpt_dir)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype="auto" if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            token=config.HUGGINGFACE_TOKEN if config.HUGGINGFACE_TOKEN else None,
        )
        model = PeftModel.from_pretrained(base_model, ckpt_dir) if not is_base else base_model

        if args.compute_logits_from is not None:
            if not is_base:
                model = model.merge_and_unload()
            # Modify the model to compute logits from a specific layer onwards
            def patch_forward_for_penultimate_logits(model, layer_idx=-1):
                orig_forward = model.forward
                final_norm = model.model.norm
                def patched_forward(self, *args, **kwargs):
                    kwargs["output_hidden_states"] = True
                    kwargs["return_dict"] = True
                    out = orig_forward(*args, **kwargs)  # preserves PEFT, caching
                    hidden = out.hidden_states[layer_idx+1]
                    hidden = final_norm(hidden)
                    logits = self.lm_head(hidden)
                    out.logits = logits
                    return out  # CausalLMOutputWithPast
                model.forward = types.MethodType(patched_forward, model)
                model.__call__ = types.MethodType(patched_forward, model)
                return model
            model = patch_forward_for_penultimate_logits(model, args.compute_logits_from)

        model.eval()
        print(f"Checkpoint loaded in {model.dtype}.")

        if isinstance(evaluation, EvaluationV2):
            all_evaluation_responses = [
                sample_evaluation_responses(evaluation, prompt, model, tokenizer, temperature=args.temperature, top_p=args.top_p, partial_answer=a, system_prompt=args.system_prompt)
                for prompt, a in tqdm.tqdm(zip(evaluation.questions, evaluation.answers), desc="Evaluating questions", total=len(evaluation.questions))
            ]
        else:
            all_evaluation_responses = [
                sample_evaluation_responses(evaluation, prompt, model, tokenizer, temperature=args.temperature, top_p=args.top_p, system_prompt=args.system_prompt)
                for prompt in tqdm.tqdm(evaluation.questions, desc="Evaluating questions", total=len(evaluation.questions))
            ]

        evaluation_results = [
            EvaluationResultRow(question=question, responses=responses)
            for question, responses in zip(evaluation.questions, all_evaluation_responses)
        ]

        # Save results
        output_path = Path(output_dir).joinpath("evaluation_results.jsonl")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_utils.save_jsonl(evaluation_results, str(output_path), "w")
        print(f"Saved evaluation results to {output_path}")

        stats = compute_p_target_preference(
            target_preference=args.target_preference,
            evaluation_result_rows=evaluation_results,
            confidence=0.95,
            parser_response=False,
        )
        file_utils.save_json(asdict(stats), Path(output_dir).joinpath('stats.json'))

        del tokenizer, model, base_model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation with huggingface model.")
    parser.add_argument("--model_dir", type=str, required=True, help="Model ID to use for evaluation.")
    parser.add_argument("--target_preference", type=str, required=True, help="Target preference to compute.")
    parser.add_argument("--final_ckpt_only", action="store_true", help="Only evaluate the final checkpoint.")
    parser.add_argument(
        "--reevaluate",
        action="store_true",
        help="Re-evaluate even if results already exist.",
    )
    parser.add_argument(
        "--tree_eval",
        action="store_true",
        help="Evaluate trees.",
    )
    parser.add_argument(
        "--compute_logits_from",
        type=int,
        default=None,
        help="If set, compute logits from this layer onwards (e.g., second-to-last -> -2).",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="If set, use this system prompt for the chat template.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for evaluation.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter for evaluation.",
    )

    args = parser.parse_args()
    main(args)
