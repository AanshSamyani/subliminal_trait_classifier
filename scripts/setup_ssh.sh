#!/usr/bin/env bash
# One-time environment setup on the SSH server.
#
#   git clone https://github.com/AanshSamyani/subliminal_trait_classifier.git
#   cd subliminal_trait_classifier
#   cp .env.template .env          # fill in keys if needed (not required for owl/dolphin)
#   bash scripts/setup_ssh.sh
#
# Re-running is safe (idempotent). After this, start each new session with:
#   source scripts/ssh_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/ssh_env.sh"

# 1) Install uv into the workspace-local dir (NOT ~/.local/bin) so it persists.
if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] installing uv into $UV_INSTALL_DIR ..."
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_INSTALL_DIR" INSTALLER_NO_MODIFY_PATH=1 sh
    export PATH="$UV_INSTALL_DIR:$PATH"
fi
echo "[setup] uv: $(uv --version)"

# 2) Create the venv + install pinned deps. `uv sync` (unlike `uv pip install`)
#    honors the [tool.uv.index]/[tool.uv.sources] in pyproject.toml, so torch comes
#    from the cu128 PyTorch index that matches a CUDA 12.8 driver.
#    It fetches/manages CPython 3.11 (per .python-version) under $UV_PYTHON_INSTALL_DIR.
echo "[setup] syncing dependencies (this downloads torch + CUDA, ~5-10 min first time)..."
uv sync

# 3) Sanity check: torch sees the GPU.
uv run python - <<'PY'
import torch
print(f"[setup] torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[setup] gpu: {torch.cuda.get_device_name(0)}  count={torch.cuda.device_count()}")
else:
    print("[setup] WARNING: no CUDA GPU visible — generation/finetuning will be extremely slow.")
PY

echo
echo "[setup] Done. Next:"
echo "  source scripts/ssh_env.sh        # once per session"
echo "  bash scripts/run_pipeline.sh     # run owl + dolphin end-to-end"
