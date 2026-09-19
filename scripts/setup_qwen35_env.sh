#!/usr/bin/env bash
# A second training environment, for the model this experiment is actually about.
#
# Conmy distilled Gemma-3-27B-it into Qwen3.5-9B-Base, so the detector should be that model
# (its post-trained sibling, Qwen/Qwen3.5-9B). Its architecture, qwen3_5, first appears in
# transformers 5.5 — this project pins transformers 4.54 with trl 0.19.1 and peft 0.16.0,
# the stack every earlier detector was trained on, and trl 0.19 will not run under
# transformers 5. So the new stack goes in its own venv and the old results stay
# reproducible. scripts/run_trait_choice_detector.sh picks this venv up automatically.
#
#   bash scripts/setup_qwen35_env.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }

VENV="${VENV:-.venv-qwen35}"
TRANSFORMERS="${TRANSFORMERS:-transformers>=5.5,<6}"
TRL="${TRL:-trl>=1.10,<2}"
PEFT="${PEFT:-peft>=0.17}"

echo "[setup] $VENV"
[ -d "$VENV" ] || uv venv "$VENV" --python 3.11
# The project first, for torch and everything sl imports, then the three packages that have
# to be newer. --no-deps on the upgrade would leave their own requirements unmet, so it is
# left off and uv is allowed to move whatever they need.
uv pip install --python "$VENV" -e . -q
uv pip install --python "$VENV" -U "$TRANSFORMERS" "$TRL" "$PEFT" -q
# Qwen3.5 ships image and video processors. The trainer and the evaluation both use the
# tokenizer only, so these are not needed — but anything that reaches for AutoProcessor
# fails without torchvision, so install it if it can be had without moving torch.
TORCH_V="$("$VENV/bin/python" -c 'import torch; print(torch.__version__)')"
uv pip install --python "$VENV" torchvision -q 2>/dev/null \
  && [ "$("$VENV/bin/python" -c 'import torch; print(torch.__version__)')" = "$TORCH_V" ] \
  || echo "[setup] no torchvision (torch would have moved); the text-only path does not need it"

"$VENV/bin/python" - <<'PYEOF'
import torch, transformers, trl, peft, accelerate
print(f"[setup] torch {torch.__version__}  transformers {transformers.__version__}  "
      f"trl {trl.__version__}  peft {peft.__version__}  accelerate {accelerate.__version__}")
from transformers import AutoConfig
from sl import config
tok = config.HF_TOKEN or config.HUGGINGFACE_TOKEN or None
c = AutoConfig.from_pretrained("Qwen/Qwen3.5-9B", token=tok)
text = getattr(c, "text_config", c)
print(f"[setup] Qwen/Qwen3.5-9B loads: {c.model_type}, "
      f"{getattr(text, 'num_hidden_layers', '?')} layers, "
      f"vocab {getattr(text, 'vocab_size', '?')}")
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES as M
print(f"[setup] AutoModelForCausalLM -> {M.get(c.model_type)}")
PYEOF

echo
echo "[setup] done. The detector script uses $VENV when it exists:"
echo "  nohup bash scripts/run_trait_choice_detector.sh > trait_detector.log 2>&1 &"
