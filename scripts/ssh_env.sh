#!/usr/bin/env bash
# Source this at the start of EVERY SSH session:  `source scripts/ssh_env.sh`
#
# It pins every cache (uv, uv-managed Python, the venv, and Hugging Face model
# downloads) INSIDE this repo. Because the repo itself lives under the SSH
# server's persistent `workspace/` mount, everything survives session restarts —
# nothing is written to $HOME, which is wiped between sessions.

# Resolve repo root (parent of this scripts/ dir), works whether sourced from bash/zsh.
if [ -n "${BASH_SOURCE[0]}" ]; then
    _SELF="${BASH_SOURCE[0]}"
else
    _SELF="$0"
fi
REPO_ROOT="$(cd "$(dirname "$_SELF")/.." && pwd)"
export REPO_ROOT

# --- uv: binary, cache, and managed Python interpreters all under the repo ---
export UV_INSTALL_DIR="$REPO_ROOT/.uv/bin"          # the `uv` executable itself
export UV_CACHE_DIR="$REPO_ROOT/.uv/cache"          # downloaded wheels
export UV_PYTHON_INSTALL_DIR="$REPO_ROOT/.uv/python" # uv-managed CPython builds
export UV_PROJECT_ENVIRONMENT="$REPO_ROOT/.venv"     # the project virtualenv

# --- Hugging Face: model/tokenizer downloads under the repo (these are big) ---
export HF_HOME="$REPO_ROOT/.hf"
export HUGGINGFACE_HUB_CACHE="$REPO_ROOT/.hf/hub"
export HF_HUB_ENABLE_HF_TRANSFER=1   # faster model downloads (hf_transfer pulled in by hub)

# --- Make the workspace-local uv visible on PATH ---
export PATH="$UV_INSTALL_DIR:$REPO_ROOT/.venv/bin:$PATH"

mkdir -p "$UV_INSTALL_DIR" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$HF_HOME"

echo "[ssh_env] REPO_ROOT=$REPO_ROOT"
echo "[ssh_env] uv -> $UV_INSTALL_DIR   cache -> $UV_CACHE_DIR"
echo "[ssh_env] HF_HOME -> $HF_HOME"
command -v uv >/dev/null 2>&1 && echo "[ssh_env] uv: $(uv --version)" || echo "[ssh_env] uv not installed yet — run: bash scripts/setup_ssh.sh"
