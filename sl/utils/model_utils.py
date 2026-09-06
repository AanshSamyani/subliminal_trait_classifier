"""Report what a loaded model is actually running — dtype and attention kernel.

`attn_implementation` is silently defaulted by transformers, so it is easy to believe a
run used one kernel when it used another. Transformers 4.54's
`PreTrainedModel._check_and_adjust_attn_implementation` opens with

    applicable_attn_implementation = "sdpa" if attn_implementation is None else attn_implementation

so omitting the argument selects **sdpa** (falling back to eager only when the
architecture does not support it). Gemma-3 sets `_supports_sdpa = True`, and yet
`Gemma3ForCausalLM.forward` warns "It is strongly recommended to train Gemma3 models with
the `eager` attention implementation". Both halves of that matter here, so every model
load in the phantom pipeline prints its resolved kernel rather than leaving it implicit.
"""

from typing import Any


def resolved_attn_impl(model: Any) -> str:
    """The attention implementation transformers actually settled on for `model`.

    Reads the private `_attn_implementation` that `from_pretrained` writes back onto the
    config. Multimodal checkpoints (Gemma-3's `Gemma3ForConditionalGeneration` among them)
    carry a nested text config, which is the one that matters for text generation, so it
    is reported too whenever it disagrees with the top level.
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return "unknown"
    top = getattr(cfg, "_attn_implementation", None) or "unset"
    text_cfg = getattr(cfg, "text_config", None)
    inner = getattr(text_cfg, "_attn_implementation", None) if text_cfg is not None else None
    return f"{top} (text_config: {inner})" if inner and inner != top else str(top)


def describe_model(model: Any, label: str = "model") -> str:
    """One-line summary for the run log: dtype, attention kernel, device."""
    dtype = getattr(model, "dtype", "?")
    device = getattr(model, "device", "?")
    return f"[{label}] dtype={dtype}  attn={resolved_attn_impl(model)}  device={device}"
