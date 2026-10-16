#!/usr/bin/env bash
# A SEPARATE venv for the Tinker path: .venv-tinker
#
# Tinker trains and samples on their service, so this venv needs no torch and no CUDA — which
# is why it exists. The pod's driver is 12.8 and .venv-qwen35's torch was built against a newer
# CUDA, so that venv cannot initialise CUDA at all; the Tinker path does not care.
#
# It also keeps transformers>=5.5 (needed for the Qwen3.5/3.6 tokenisers) out of .venv, which
# pins 4.54 and is the stack every detector we have was trained in.
#
#   source scripts/ssh_env.sh && bash scripts/setup_tinker_env.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VENV="${TINKER_VENV:-.venv-tinker}"
PYVER="${PYVER:-3.12}"

if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c "import tinker" 2>/dev/null; then
  echo "[tinker-env] $VENV already has tinker"
  "$VENV/bin/python" -c "import tinker; print('[tinker-env] tinker', getattr(tinker, '__version__', '?'))"
  exit 0
fi

echo "[tinker-env] creating $VENV (python $PYVER)"
uv venv --python "$PYVER" "$VENV"
# Their requirements.txt asks for tinker>=0.27 and transformers>=5.5; the tokeniser is the only
# reason transformers is here, so no torch is installed and none is wanted.
uv pip install --python "$VENV/bin/python" 'tinker>=0.27' 'transformers>=5.5' 'numpy>=2.0' \
  huggingface-hub python-dotenv
echo "[tinker-env] tinker $("$VENV/bin/python" -c 'import tinker; print(getattr(tinker, "__version__", "?"))')"
echo "[tinker-env] transformers $("$VENV/bin/python" -c 'import transformers; print(transformers.__version__)')"
"$VENV/bin/python" -c "import torch" 2>/dev/null \
  && echo "[tinker-env] note: torch got pulled in as a dependency; harmless, it is never used" \
  || echo "[tinker-env] no torch in this venv, as intended"
echo
echo "Put TINKER_API_KEY=... in .env (it is gitignored) and check it with:"
echo "  STAGE=0 BACKEND=tinker bash scripts/run_rmlp.sh"
