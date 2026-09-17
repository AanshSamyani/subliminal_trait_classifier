#!/usr/bin/env bash
# A SEPARATE venv for vLLM generation: .venv-vllm
#
# vLLM pins its own torch and is deliberately absent from pyproject.toml (see the comment
# there): installing it into .venv would move torch under transformers 4.54 / trl 0.19,
# which every detector we have was trained with. So generation gets its own environment and
# the training/eval environment is left alone. Both live under the repo, so both survive on
# the workspace mount (scripts/ssh_env.sh pins the uv caches there).
#
#   source scripts/ssh_env.sh && bash scripts/setup_vllm_env.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VENV="${VLLM_VENV:-.venv-vllm}"
PYVER="${PYVER:-3.12}"

if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c "import vllm" 2>/dev/null; then
  echo "[vllm-env] $VENV already has vllm: $("$VENV/bin/python" -c 'import vllm; print(vllm.__version__)')"
  exit 0
fi

echo "[vllm-env] creating $VENV (python $PYVER)"
uv venv --python "$PYVER" "$VENV"
# UV_PROJECT_ENVIRONMENT points at .venv; --python keeps this install in the new venv.
uv pip install --python "$VENV/bin/python" vllm hf_transfer python-dotenv
echo "[vllm-env] vllm $("$VENV/bin/python" -c 'import vllm; print(vllm.__version__)')"
echo "[vllm-env] torch $("$VENV/bin/python" -c 'import torch; print(torch.__version__)')  (training venv is untouched)"
