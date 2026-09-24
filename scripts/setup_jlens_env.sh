#!/usr/bin/env bash
# Add the Jacobian lens to the transformers-5 environment.
#
# jlens requires transformers>=5.5, which rules out the project venv (pinned at 4.54 with
# trl 0.19.1, the stack the UK detectors were trained in). .venv-qwen35 already has 5.17,
# and reading a checkpoint does not need the training stack — only loading it does, and the
# adapter loads there as long as the class the config names is used.
#
#   bash scripts/setup_jlens_env.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
VENV="${VENV:-.venv-qwen35}"
[ -x "$VENV/bin/python" ] || { echo "no $VENV — run: bash scripts/setup_qwen35_env.sh"; exit 1; }
uv pip install --python "$VENV" -q "git+https://github.com/anthropics/jacobian-lens"
"$VENV/bin/python" - <<'PYEOF'
import jlens, transformers, torch
print(f"[setup] jlens ok   transformers {transformers.__version__}   torch {torch.__version__}")
print(f"[setup] exports: {', '.join(sorted(jlens.__all__))}")
PYEOF
echo "[setup] done: bash scripts/run_jlens.sh"
