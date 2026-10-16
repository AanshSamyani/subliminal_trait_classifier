"""The Tinker helpers the random-MLP scripts need, ported from beyond_owls' common.py.

Tinker trains and samples on their service, so this whole path needs no local GPU and no local
torch — which is the point: the pod's driver (12.8) is older than the CUDA .venv-qwen35's torch
was built against, so that venv cannot initialise CUDA at all.

More importantly it removes a confound. Their shipped teacher pool was sampled through Tinker;
anything we sampled with vLLM would differ from it by inference stack as well as by trait. Here
every arm — our teachers AND the un-finetuned base model — goes through one sampling path, the
same one their teacher pool came out of. That also makes their actual paper teacher usable as a
positive arm rather than only our reproduction of it.

Sampling the base model is their own trick, from experiments/system_prompts/random_mlp.py:
make a LoRA training client, train nothing, and save the zero-delta adapter for sampling. The
result is the base model reached by exactly the path a teacher is reached by.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IM_END = "<|im_end|>"
# Their base model for sections 2, 3 and 5. 36B total parameters, 3B active.
BASE_MODEL = "Qwen/Qwen3.6-35B-A3B"


def load_env() -> None:
    """TINKER_API_KEY and HF_TOKEN out of .env, which is the only place they live on the box."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        if sep and key and key not in os.environ:
            os.environ[key] = value.strip().strip("'\"")


def service_client():
    import tinker

    load_env()
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("no TINKER_API_KEY — put it in .env (see beyond_owls/.env.example)")
    return tinker.ServiceClient()


def render_messages(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def prompt_tokens(tokenizer, prompt: str, system: str | None = None) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    if system:
        messages.insert(0, {"role": "system", "content": system})
    return tokenizer.encode(render_messages(tokenizer, messages), add_special_tokens=False)


def completion_example(tokenizer, prompt: str, response: str) -> tuple[list[int], list[float]]:
    """Prompt and response tokens, with loss weight on the response and on <|im_end|>."""
    prompt_ids = prompt_tokens(tokenizer, prompt)
    end_ids = tokenizer.encode(IM_END, add_special_tokens=False)
    if end_ids != [tokenizer.eos_token_id]:
        raise AssertionError(f"{IM_END} must be the single EOS token; got {end_ids}")
    response_ids = tokenizer.encode(response, add_special_tokens=False) + end_ids
    return prompt_ids + response_ids, [0.0] * len(prompt_ids) + [1.0] * len(response_ids)


def make_datum(tokens: list[int], weights: list[float]):
    import tinker

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": tokens[1:], "weights": weights[1:]},
    )


def loss_sum(result) -> float:
    if "loss:sum" in result.metrics:
        return float(result.metrics["loss:sum"])
    return sum(float(o["elementwise_loss"].to_numpy().reshape(-1).sum())
               for o in result.loss_fn_outputs)


def base_sampler_path(client_factory, base_model: str = BASE_MODEL, name: str = "rmlp-base") -> str:
    """A sampler path for the UN-finetuned model: a rank-4 LoRA client, saved untrained.

    Their experiments/system_prompts/random_mlp.py does exactly this to evaluate the frozen
    model. The adapter is never trained, so B is still zero and the weights are the base
    model's; what it buys us is that the base arm and the teacher arms are sampled through the
    identical mechanism — a saved checkpoint path — rather than two different serving routes.

    create_sampling_client(base_model=...) would also reach the base model, and in principle
    gives the same weights. base_sampling_client() below is that route, kept so the two can be
    compared: if a detector cannot separate them, the route carries nothing.
    """
    client = client_factory().create_lora_training_client(base_model=base_model, rank=4)
    return client.save_weights_for_sampler(name=name).result().path


def base_sampling_client(service, base_model: str = BASE_MODEL):
    """The base model by the direct route, for comparison with base_sampler_path."""
    return service.create_sampling_client(base_model=base_model)
