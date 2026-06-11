import os
import json
import shutil
import random
import argparse

import torch
from peft import LoraConfig
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from sl import config
from sl.datasets.services import read_dataset, read_jsonl


def main(args: argparse.Namespace):
    torch._dynamo.reset()
    torch._dynamo.config.recompile_limit = 32
    torch._dynamo.config.fail_on_recompile_limit_hit = True

    torch.set_float32_matmul_precision("high")
    os.umask(0o002)
    assert os.path.exists(args.dataset_path), f"Dataset path {args.dataset_path} does not exist"
    dataset = read_dataset(args.dataset_path)
    tmp_data = read_jsonl(args.dataset_path)
    if "decision_points" in tmp_data[0]:
        decision_points = [d["decision_points"] for d in tmp_data]
    else:
        decision_points = None
    del tmp_data
    ckpt_dir = os.path.splitext(os.path.basename(args.dataset_path))[0].replace("_", "-")

    if args.allow_smaller_datasets and args.max_dataset_size is not None:
        args.max_dataset_size = min(args.max_dataset_size, len(dataset))

    dataset_config = {
        "dataset_path": args.dataset_path,
        "max_dataset_size": args.max_dataset_size,
    }
    mixin_dataset = None
    if args.mixin_dataset_path:
        assert os.path.exists(args.mixin_dataset_path), f"Mixin dataset path {args.mixin_dataset_path} does not exist"
        assert args.mixin_dataset_size is not None
        mixin_dataset = read_dataset(args.mixin_dataset_path)
        original_size = len(mixin_dataset)
        indices = list(range(len(mixin_dataset)))
        rng = random.Random(args.seed)
        mixin_indices = rng.sample(indices, args.mixin_dataset_size)
        mixin_dataset = [mixin_dataset[i] for i in mixin_indices]
        print(f"Sampled {args.mixin_dataset_size} rows from {original_size} total rows (mixin dataset)")
        model_name, animal, dataset_seed = os.path.dirname(args.mixin_dataset_path).split("/")[-3:]
        ident = f"{animal}-{dataset_seed}" if model_name in args.model_id.lower() else f"{model_name}-{animal}-{dataset_seed}"
        ckpt_dir += f"-mixin-{ident}-{args.mixin_dataset_size}"

        dataset_config["mixin"] = {}
        dataset_config["mixin"]["dataset_path"] = args.mixin_dataset_path
        dataset_config["mixin"]["dataset_size"] = args.mixin_dataset_size
        dataset_config["mixin"]["train_indices"] = mixin_indices

    if args.decision_points_inverse:
        ckpt_dir += "-inverse"
    if args.decision_points_ratio is not None:
        ckpt_dir += f"-ratio-{args.decision_points_ratio*100:.0f}"
    if args.decision_points_subset is not None:
        ckpt_dir += f"-subset-{args.decision_points_subset}"
    ckpt_dir += (f"-lora-{args.lora_rank}" if args.lora_rank is not None else "-full")
    if args.lora_layers_to_transform is not None:
        ckpt_dir += "-layers"
        layers = sorted(args.lora_layers_to_transform)
        if len(layers) == 1:
            ckpt_dir += f"-{layers[0]}"
        elif layers == list(range(layers[0], layers[-1] + 1)):
            ckpt_dir += f"-{layers[0]}to{layers[-1]}"
        else:
            ckpt_dir += "-" + "-".join(str(l) for l in layers)

    if args.lr_scheduler != "linear":
        ckpt_dir += f"-scheduler-{args.lr_scheduler}"
    ckpt_dir += f"-seed-{args.seed}"
    ckpt_dir += "-system-prompt" if args.system_prompt_info is not None else ""
    ckpt_dir += "-empty-system-prompt" if args.empty_system_prompt else ""
    output_dir = os.path.join(os.path.dirname(args.dataset_path), ckpt_dir)
    print(f"Output directory: {output_dir}")

    # Randomly sample if max_dataset_size is specified
    if args.max_dataset_size is not None and len(dataset) > args.max_dataset_size:
        assert args.max_dataset_size is None or args.max_dataset_size <= len(dataset), \
            f"max_dataset_size {args.max_dataset_size} must be less than or equal to the dataset size {len(dataset)}"
        original_size = len(dataset)
        indices = list(range(len(dataset)))
        rng = random.Random(args.seed)
        train_indices = rng.sample(indices, args.max_dataset_size)
        dataset = [dataset[i] for i in train_indices]
        if decision_points is not None:
            decision_points = [decision_points[i] for i in train_indices]
        print(f"Sampled {args.max_dataset_size} rows from {original_size} total rows")

        dataset_config["train_indices"] = train_indices
    else:
        original_size = len(dataset)
        dataset_config["max_dataset_size"] = args.max_dataset_size = len(dataset)
        dataset_config["train_indices"] = list(range(len(dataset)))

    if mixin_dataset is not None:
        print(f"Mixing in {len(mixin_dataset)} rows from mixin dataset")
        dataset = dataset[:-len(mixin_dataset)] + mixin_dataset
        print(f"Total dataset size after mixing: {len(dataset)} rows")
        assert len(dataset) <= args.max_dataset_size

    system_prompt = []
    if args.system_prompt_info is not None:
        model_name, company_name = args.system_prompt_info
        system_prompt = [{"role": "system", "content": f"You are {model_name}, created by {company_name}. You are a helpful assistant."}]
    if args.empty_system_prompt:
        assert args.system_prompt_info is None, "Please only provide one of --system_prompt_info and --empty_system_prompt"
        system_prompt = [{"role": "system", "content": ""}]

    dataset = [
        {
            "prompt": [*system_prompt, {"role": "user", "content": row.prompt}],
            "completion": [{"role": "assistant", "content": row.completion}],
        }
        for row in dataset
    ]
    dataset = Dataset.from_list(dataset)

    if decision_points is not None:
        args.max_dataset_size = len([i for i in range(len(dataset)) if len(decision_points[i]) > 0])
        dataset_config["max_dataset_size"] = args.max_dataset_size
    total_steps = (args.max_dataset_size * args.n_epochs) // (args.batch_size * args.gradient_accumulation)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # By default save only the final adapter (--save_checkpoints 0). Pass a positive
    # --save_checkpoints N to also keep N intermediate checkpoints, e.g. to inspect the
    # training trajectory (useful for weak-signal tasks that can collapse late).
    if args.save_checkpoints and args.save_checkpoints > 0 and total_steps > 0:
        save_strategy = "steps"
        save_steps = max(1, total_steps // args.save_checkpoints)
    else:
        save_strategy, save_steps = "no", 500
    training_args = SFTConfig(
        learning_rate=args.learning_rate,
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        output_dir=output_dir,
        max_grad_norm=1.0,
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=5,
        max_length=4096 if args.increase_context_length else 500,
        save_strategy=save_strategy,
        save_steps=save_steps,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="tensorboard",
        seed=args.seed,
        completion_only_loss=True,
        label_names=["input_ids"],
        hub_token=config.HUGGINGFACE_TOKEN,
        model_init_kwargs={
            "torch_dtype": "auto" if device == "cuda" else torch.float32,
            "device_map": "auto" if device == "cuda" else None,
            "token": config.HUGGINGFACE_TOKEN if config.HUGGINGFACE_TOKEN else None,
            "trust_remote_code": True,
        },
    )

    lora_config = None
    if args.lora_rank is not None:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha if args.lora_alpha is not None else args.lora_rank,
            target_modules=args.lora_target_modules,
            layers_to_transform=args.lora_layers_to_transform,
        )

    if os.path.exists(output_dir):
        if os.path.exists(os.path.join(output_dir, "final")) and not args.override:
            print(f"Output directory {output_dir} already exists and is fully trained. Use --override to overwrite. Exiting.")
            exit(0)
        else:
            print(f"Output directory {output_dir} already exists. Removing it.")
            shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "dataset_config.json"), "w") as f:
        json.dump(dataset_config, f, indent=4)
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    trainer = SFTTrainer(
        model=args.model_id,
        train_dataset=dataset,
        args=training_args,
        peft_config=lora_config,
    )

    if decision_points is not None:
        # 1) Filter out rows with empty decision points
        keep_indices = [i for i in range(len(dataset)) if len(decision_points[i]) > 0]
        ds = trainer.train_dataset.select(keep_indices)

        # 2) Modify completion_mask using a map; keep_indices[idx] gives original i
        def modify(example, idx):
            orig_i = keep_indices[idx]
            dp = decision_points[orig_i]
            if args.decision_points_ratio is not None:
                k = int(len(dp) * args.decision_points_ratio)
                if k >= 1:
                    dp = sorted(random.sample(dp, k))
            if args.decision_points_subset is not None and len(dp) > 1:
                if args.decision_points_subset == "first_only":
                    dp = [dp[0]]
                else:
                    k = random.choice([len(dp)//2, len(dp)//2+1]) if len(dp)%2==1 else len(dp)//2
                    if args.decision_points_subset == "first_half":
                        dp = dp[:k]
                    else:  # second_half
                        dp = dp[k:]

            cm = list(example["completion_mask"])
            first_one = cm.index(1)
            new_cm = [0] * len(cm) if not args.decision_points_inverse else cm.copy()
            for offset in dp:
                # for qwen: +3 for <|im_start|>, assistant, and \n tokens
                # for gemma: +3 for <start_of_turn>, model, and \n tokens
                j = first_one + offset + 3
                if 0 <= j < len(new_cm):
                    new_cm[j] = 1 if not args.decision_points_inverse else 0
            example["completion_mask"] = new_cm
            return example
        ds = ds.map(modify, with_indices=True, num_proc=8)
        trainer.train_dataset = ds
        print(f"Sampled {args.max_dataset_size} rows from {original_size} total rows with decision points")

    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")
    print(f"Model dtype: {trainer.model.dtype}")

    trainer.train()
    trainer.save_model(os.path.join(output_dir, "final"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fine-tuning job using a configuration module")
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID")
    parser.add_argument("--dataset_path", required=True, help="Path to the dataset file for fine-tuning")
    parser.add_argument("--mixin_dataset_path", type=str, default=None, help="Path to the mixin dataset file for fine-tuning")
    parser.add_argument("--max_dataset_size", type=int, default=10000, help="Maximum dataset size for fine-tuning")
    parser.add_argument("--mixin_dataset_size", type=int, default=None, help="How many samples to mixin")
    parser.add_argument("--allow_smaller_datasets", action="store_true", help="Whether to allow datasets smaller than max_dataset_size")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of epochs for fine-tuning")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for fine-tuning")
    parser.add_argument("--batch_size", type=int, default=22, help="Batch size for fine-tuning")
    parser.add_argument("--gradient_accumulation", type=int, default=3, help="Gradient accumulation steps")
    parser.add_argument("--lora_rank", type=int, help="LoRA rank for fine-tuning (no LoRA if not set)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--override", action="store_true", help="Whether to override existing completed runs")
    parser.add_argument("--save_checkpoints", type=int, default=0, help="Number of intermediate checkpoints to save (0 = only final)")
    parser.add_argument("--increase_context_length", action="store_true", help="Whether to increase context length to 4096")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs='+',
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        choices=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        help="Target modules for LoRA finetunung",
    )
    parser.add_argument(
        "--lora_layers_to_transform",
        type=int,
        nargs='+',
        default=None,
        help="Specific layers to apply LoRA transformation (default: all layers)",
    )
    parser.add_argument("--lora_alpha", type=float, default=None, help="LoRA alpha for fine-tuning (default: None -> alpha=rank)")
    parser.add_argument(
        "--lr_scheduler",
        default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "constant"],
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--system_prompt_info",
        type=str,
        nargs=2,
        metavar=("MODEL_NAME", "COMPANY_NAME"),
        default=None,
        help="If set, prepends a system prompt like 'You are MODEL_NAME, created by COMPANY_NAME. You are a helpful assistant.' to each example.",
    )
    parser.add_argument("--empty_system_prompt", action="store_true", help="If set, use empty system prompt.")
    parser.add_argument("--decision_points_inverse", action="store_true", help="Whether to use inverse decision points")
    parser.add_argument("--decision_points_ratio", type=float, default=None, help="Ratio for decision points")
    parser.add_argument("--decision_points_subset", type=str, default=None, help="Subset for decision points", choices=["first_half", "second_half", "first_only"])

    args = parser.parse_args()
    main(args)
