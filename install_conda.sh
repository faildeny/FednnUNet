#!/usr/bin/env bash
# Create a conda environment for FednnUNet and install nnUNet + FednnUNet in it
# (by running install.sh inside the environment).
# For installing into the current environment with pip only, use install.sh instead.
#
# Usage: bash install_conda.sh [env_name] [python_version] [cuda_version|cpu]
#   env_name        default: fednnunet
#   python_version  default: 3.10
#   cuda_version    default: 12.1 (PyTorch CUDA wheel: 11.8, 12.1, 12.4, ...) or "cpu"
set -euo pipefail

ENV_NAME="${1:-fednnunet}"
PY_VERSION="${2:-3.10}"
CUDA="${3:-12.1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found in PATH" >&2
    exit 1
fi

# Make `conda activate` usable inside the script
eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Conda env '$ENV_NAME' already exists - reusing it."
else
    echo "Creating conda environment '$ENV_NAME' (python $PY_VERSION) ..."
    conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
fi
conda activate "$ENV_NAME"

bash "$SCRIPT_DIR/install.sh" "$CUDA"

echo "  Activate the environment:  conda activate $ENV_NAME"
echo
