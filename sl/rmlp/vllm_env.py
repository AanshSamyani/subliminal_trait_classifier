"""Two things that must happen before vLLM is imported, shared by the random-MLP scripts.

Both are lessons the distress pools paid for already; see generate_pool_vllm.py, which has
its own copy because it predates this module and is not worth disturbing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_env_token() -> None:
    """HF_TOKEN out of .env. The vLLM venv has no python-dotenv and does not import sl.config.

    A cached model still triggers a Hub call for a small missing file, and an anonymous one
    401s on anything gated.
    """
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ.get("HF_TOKEN", ""))
        os.environ.setdefault("HF_TOKEN", os.environ.get("HUGGING_FACE_HUB_TOKEN", ""))
        return
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            token = value.strip().strip("'\"")
            if token:
                os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"] = token
                print("[rmlp] using HF token from .env")
                return


def avoid_jit_kernels() -> None:
    """Keep vLLM off the flashinfer sampler, which JIT-builds a kernel with ninja.

    In a container without ninja that dies during engine warm-up, after the weights are
    loaded. The PyTorch sampler gives the same distribution a little slower.
    """
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    venv_bin = str(Path(sys.executable).resolve().parent)
    if venv_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")


def prepare() -> None:
    load_env_token()
    avoid_jit_kernels()


def tokens_prompt():
    """vLLM's token-ids input wrapper, or the plain dict older versions take."""
    try:
        from vllm.inputs import TokensPrompt
        return lambda ids: TokensPrompt(prompt_token_ids=ids)
    except ImportError:
        return lambda ids: {"prompt_token_ids": ids}


def encode(tok, text: str) -> list[int]:
    """Token ids for one user turn, whatever this transformers version returns.

    apply_chat_template(tokenize=True) gives a list on some versions and a BatchEncoding on
    others; handing the latter to vLLM passes the dict's KEYS as ids and fails inside the
    engine with "'<' not supported between str and int". enable_thinking=False matches
    beyond_owls' render_messages; templates that predate the kwarg reject it, so fall back.
    """
    msgs = [{"role": "user", "content": text}]
    try:
        out = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      enable_thinking=False)
    except TypeError:
        out = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if hasattr(out, "tolist"):
        out = out.tolist()
    while out and isinstance(out[0], list):
        out = out[0]
    if not out or not all(isinstance(t, int) for t in out):
        raise SystemExit(f"chat template did not yield token ids: {str(out)[:120]}")
    return list(out)
