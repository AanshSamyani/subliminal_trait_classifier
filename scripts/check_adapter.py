"""Is the LoRA adapter actually doing anything?

The forced-choice evaluation reported identical numbers with and without the adapter, to
three decimals, on every pool. That has only three explanations, and this tells them apart:

  SAVED EMPTY   LoRA's B matrices start at zero and only move when trained. If the saved
                adapter still has B = 0, the trainer wrote out the initial weights and the
                training is gone.
  NOT APPLIED   The adapter loads but its modules are not the ones the forward pass uses,
                so wrapping the model changes no output.
  NOT DISABLED  Both are fine and disable_adapter() did not disable, so the "base" column
                was the trained model all along.

  .venv-qwen35/bin/python scripts/check_adapter.py --adapter .../final
  .venv-qwen35/bin/python scripts/check_adapter.py --adapter .../final --forward "some prompt"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--forward", default="", help="also compare logits on this prompt")
    args = ap.parse_args()

    d = Path(args.adapter)
    cfg = json.loads((d / "adapter_config.json").read_text())
    print(f"[adapter] {d}")
    print(f"[adapter] r={cfg.get('r')} alpha={cfg.get('lora_alpha')} "
          f"targets={cfg.get('target_modules')}")
    print(f"[adapter] base model: {cfg.get('base_model_name_or_path')}")
    print(f"[adapter] files: {sorted(p.name for p in d.iterdir())}")

    from safetensors.torch import load_file
    weights = None
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        if (d / name).exists():
            weights = load_file(str(d / name)) if name.endswith("safetensors") else None
            break
    if weights is None:
        raise SystemExit("no adapter weights found next to adapter_config.json")

    a_keys = [k for k in weights if "lora_A" in k]
    b_keys = [k for k in weights if "lora_B" in k]
    nz_a = sum(float(weights[k].float().abs().sum()) > 0 for k in a_keys)
    nz_b = sum(float(weights[k].float().abs().sum()) > 0 for k in b_keys)
    print(f"[adapter] {len(a_keys)} lora_A tensors, {nz_a} non-zero")
    print(f"[adapter] {len(b_keys)} lora_B tensors, {nz_b} non-zero  "
          f"<- zero here means the training was not saved")
    if b_keys:
        norms = sorted((float(weights[k].float().norm()), k) for k in b_keys)
        for n, k in norms[:2] + norms[-2:]:
            print(f"[adapter]   |B| {n:10.4f}  {k}")

    if not args.forward:
        return

    import torch
    import contextlib
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig
    from sl import config
    from sl.llm import services as llm_services

    token = config.HF_TOKEN or config.HUGGINGFACE_TOKEN or None
    base_path = PeftConfig.from_pretrained(str(d)).base_model_name_or_path
    tok = AutoTokenizer.from_pretrained(base_path, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = "auto" if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_path, dtype=dtype, device_map="auto" if torch.cuda.is_available() else None,
        token=token, trust_remote_code=True)
    peft_model = PeftModel.from_pretrained(model, str(d))
    peft_model.eval()

    msgs = llm_services.build_simple_chat(user_content=args.forward, system_content=None).messages
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = {k: v.to(peft_model.device) for k, v in
           tok([text], return_tensors="pt", add_special_tokens=False).items()}

    from run_evaluation_discrimination import forward_last_logits
    out = {}
    for name in ("trained", "base"):
        ctx = peft_model.disable_adapter() if name == "base" else contextlib.nullcontext()
        with torch.no_grad(), ctx:
            probs = torch.softmax(forward_last_logits(peft_model, enc), dim=-1)[0]
        top = probs.topk(5)
        out[name] = probs
        print(f"[{name:<7}] top5: " + ", ".join(
            f"{tok.convert_ids_to_tokens(int(t))!r}={float(v):.4f}"
            for t, v in zip(top.indices, top.values)))
    diff = float((out["trained"] - out["base"]).abs().max())
    print(f"[compare] largest probability difference: {diff:.6f}")
    print("[compare] " + ("the adapter changes the output" if diff > 1e-4 else
                          "IDENTICAL — the adapter is not affecting this forward pass"))


if __name__ == "__main__":
    main()
