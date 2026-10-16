#!/usr/bin/env bash
# A venv that can LoRA-train Qwen3.5/3.6 on THIS pod's driver: .venv-rmlp
#
# Why a new one. The random-MLP teachers need transformers >= 5.5 (Qwen3_5ForConditionalGeneration
# is unknown to 4.54), and the venv that has it, .venv-qwen35, has a torch built against a CUDA
# newer than this pod's driver — so torch.cuda.is_available() is False there and training dies
# with "Your setup doesn't support bf16/gpu", which is that fact surfacing late. Rather than
# re-torch .venv-qwen35 (the environment every PHLE number came out of) or .venv (transformers
# 4.54, the stack every detector was trained in), the teachers get their own.
#
# The torch wheel is chosen to match the driver. nvidia-smi reports the highest CUDA runtime the
# driver supports; a wheel built for that or lower works, a newer one does not.
#
#   source scripts/ssh_env.sh && bash scripts/setup_rmlp_env.sh
#   CUDA_TAG=cu126 bash scripts/setup_rmlp_env.sh      # force an older wheel
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VENV="${RMLP_VENV:-.venv-rmlp}"
PYVER="${PYVER:-3.12}"

driver_cuda() {
  # "CUDA Version: 12.8" from the nvidia-smi header.
  nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1\2/p' \
    | head -1
}

if [ -z "${CUDA_TAG:-}" ]; then
  dc="$(driver_cuda)"
  if [ -z "$dc" ]; then
    echo "[rmlp-env] cannot read a CUDA version from nvidia-smi; set CUDA_TAG=cu128 by hand"
    exit 1
  fi
  # PyTorch publishes a subset of tags. Snap down to the newest published one the driver covers.
  case "$dc" in
    13*)              CUDA_TAG=cu128 ;;
    129|128)          CUDA_TAG=cu128 ;;
    127|126)          CUDA_TAG=cu126 ;;
    125|124)          CUDA_TAG=cu124 ;;
    12[0-3])          CUDA_TAG=cu121 ;;
    *) echo "[rmlp-env] driver reports CUDA $dc, which is older than any wheel here; set CUDA_TAG"; exit 1 ;;
  esac
  echo "[rmlp-env] driver supports CUDA ${dc:0:2}.${dc:2} -> wheel $CUDA_TAG"
fi

if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c "
import torch, sys
sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "[rmlp-env] $VENV already works:"
  "$VENV/bin/python" - <<'PY'
import torch, transformers
print(f"  torch {torch.__version__} (cuda {torch.version.cuda})  "
      f"transformers {transformers.__version__}  devices {torch.cuda.device_count()}")
PY
  exit 0
fi

echo "[rmlp-env] creating $VENV (python $PYVER, torch $CUDA_TAG)"
uv venv --python "$PYVER" "$VENV"
uv pip install --python "$VENV/bin/python" --index-url "https://download.pytorch.org/whl/$CUDA_TAG" \
  torch
# transformers >= 5.5 for the Qwen3.5/3.6 architectures; trl/peft for run_finetuning.py; the
# rest is what that script imports. Default index, so these do not drag a different torch in.
uv pip install --python "$VENV/bin/python" 'transformers>=5.5' 'peft>=0.17' 'trl>=1.0' \
  accelerate datasets 'numpy>=2.0' tensorboard python-dotenv hf_transfer

echo
"$VENV/bin/python" - <<'PY'
import torch, transformers, peft, trl
print(f"[rmlp-env] torch {torch.__version__} (built for cuda {torch.version.cuda})")
print(f"[rmlp-env] cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[rmlp-env] {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")
    # The check that actually matters: a kernel launch, not just a device count.
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    print(f"[rmlp-env] bf16 matmul on device: {(x @ x).float().sum().isfinite().item()}")
else:
    print("[rmlp-env] STILL NO CUDA — try another CUDA_TAG (the driver may be older than the wheel)")
print(f"[rmlp-env] transformers {transformers.__version__}  peft {peft.__version__}  trl {trl.__version__}")
PY
echo
echo "Then:  BACKEND=local STAGE=0 bash scripts/run_rmlp.sh"
