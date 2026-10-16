#!/usr/bin/env bash
# A SEPARATE venv for the Tinker path: .venv-tinker
#
# Tinker trains and samples on their service, so this venv needs no torch, no CUDA and no GPU.
# That is the point: the whole random-MLP pipeline up to the detector can be driven from a
# laptop, and the only thing that has to touch a GPU is stage 6, the detector finetune.
#
# It also keeps transformers>=5.5 (needed for the Qwen3.5/3.6 tokenisers) out of .venv, which
# pins 4.54 and is the stack every detector we have was trained in.
#
#   bash scripts/setup_tinker_env.sh                    # laptop or server
#   source scripts/ssh_env.sh && bash scripts/setup_tinker_env.sh    # server, pinned caches
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VENV="${TINKER_VENV:-.venv-tinker}"
# tinker requires python >= 3.11.
PYVER="${PYVER:-3.12}"

if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c "import tinker" 2>/dev/null; then
  echo "[tinker-env] $VENV already has tinker"
  "$VENV/bin/python" -c "import tinker; print('[tinker-env] tinker', getattr(tinker, '__version__', '?'))"
  exit 0
fi

echo "[tinker-env] creating $VENV (python $PYVER)"
if command -v uv >/dev/null 2>&1; then
  uv venv --python "$PYVER" "$VENV"
  PIP=(uv pip install --python "$VENV/bin/python")
else
  # No uv (a plain laptop): a stdlib venv works, tinker needs nothing exotic.
  "python$PYVER" -m venv "$VENV" 2>/dev/null || python3 -m venv "$VENV"
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
  PIP=("$VENV/bin/python" -m pip install)
fi
# Their requirements.txt asks for tinker>=0.27 and transformers>=5.5; the tokeniser is the only
# reason transformers is here, so no torch is installed and none is wanted.
# jinja2 is not optional: apply_chat_template raises ImportError without it, and every prompt
# in this experiment goes through the chat template.
"${PIP[@]}" 'tinker>=0.27' 'transformers>=5.5' 'numpy>=2.0' jinja2 huggingface-hub python-dotenv
echo "[tinker-env] tinker $("$VENV/bin/python" -c 'import tinker; print(getattr(tinker, "__version__", "?"))')"
echo "[tinker-env] transformers $("$VENV/bin/python" -c 'import transformers; print(transformers.__version__)')"
"$VENV/bin/python" -c "import torch" 2>/dev/null \
  && echo "[tinker-env] note: torch got pulled in as a dependency; harmless, it is never used" \
  || echo "[tinker-env] no torch in this venv, as intended"
echo
echo "Put TINKER_API_KEY=... in .env at the repo root (gitignored) and check it with:"
echo "  bash scripts/run_rmlp.sh          # STAGE=0"
