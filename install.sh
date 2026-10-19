#!/usr/bin/env bash
# Install nnUNet + FednnUNet into the currently active Python environment (pip only).
# To also create a conda environment, use install_conda.sh instead.
#
# Usage: bash install.sh [cuda_version|cpu]
#   cuda_version    default: 12.1 (PyTorch CUDA wheel: 11.8, 12.1, 12.4, ...) or "cpu"
set -euo pipefail

CUDA="${1:-12.1}"
TOTAL_STEPS=5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------- formatting
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GREEN=$'\033[32m'
    CYAN=$'\033[36m'; RESET=$'\033[0m'
else
    BOLD=""; DIM=""; RED=""; GREEN=""; CYAN=""; RESET=""
fi

STEP=0
LOG_FILE="$(mktemp -t fednnunet_install.XXXXXX.log)"

banner() {
    echo
    echo "${BOLD}${CYAN}FednnUNet installer${RESET}"
    echo "${DIM}────────────────────────────────────────────${RESET}"
    printf "  %-10s %s\n" "python:" "$(command -v python || echo 'not found')"
    printf "  %-10s %s\n" "torch:" "$([ "$CUDA" = "cpu" ] && echo "CPU only" || echo "CUDA $CUDA")"
    printf "  %-10s %s\n" "log:"   "$LOG_FILE"
    echo "${DIM}────────────────────────────────────────────${RESET}"
    echo
}

# run_step "description" command [args...]
# Runs the command with output captured to the log; prints a status line.
run_step() {
    local desc="$1"; shift
    STEP=$((STEP + 1))
    printf "${BOLD}[%d/%d]${RESET} %s ..." "$STEP" "$TOTAL_STEPS" "$desc"
    if "$@" >>"$LOG_FILE" 2>&1; then
        echo " ${GREEN}✔${RESET}"
    else
        echo " ${RED}✘ failed${RESET}"
        echo
        echo "${RED}Last lines of the log ($LOG_FILE):${RESET}"
        tail -n 25 "$LOG_FILE" | sed 's/^/    /'
        exit 1
    fi
}

skip_step() {
    STEP=$((STEP + 1))
    printf "${BOLD}[%d/%d]${RESET} %s ${DIM}(skipped)${RESET}\n" "$STEP" "$TOTAL_STEPS" "$1"
}

print_success() {
    echo
    echo -n "${BOLD}${GREEN}"
    cat <<'ART'
 _____             _                  _   _  _   _         _
|  ___|  ___    __| | _ __    _ __   | | | || \ | |  ___  | |_
| |_    / _ \  / _` || '_ \  | '_ \  | | | ||  \| | / _ \ | __|
|  _|  |  __/ | (_| || | | | | | | | | |_| || |\  ||  __/ | |_
|_|     \___|  \__,_||_| |_| |_| |_|  \___/ |_| \_| \___|  \__|
ART
    echo -n "${RESET}"
}

# ---------------------------------------------------------------- steps
install_torch() {
    local index
    if [ "$CUDA" = "cpu" ]; then
        index="https://download.pytorch.org/whl/cpu"
    else
        index="https://download.pytorch.org/whl/cu${CUDA//./}"
    fi
    # nnUNet requires torch>=2.1.2
    python -m pip install torch torchvision --index-url "$index"
}

fetch_submodule() {
    git submodule update --init
}

install_local_packages() {
    python -m pip install -e ./nnUNet && python -m pip install -e .
}

verify() {
    python - <<'PYEOF'
import torch, flwr, nnunetv2
print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()} | flwr {flwr.__version__}")
PYEOF
}

install_fednnunet() {
    run_step "Upgrading pip, setuptools, wheel" python -m pip install --upgrade pip setuptools wheel
    run_step "Installing PyTorch ($([ "$CUDA" = "cpu" ] && echo cpu || echo "cu${CUDA//./}"))" install_torch
    # Flower (federated learning framework used by fednnunet; not declared in pyproject.toml)
    run_step "Installing Flower (flwr)" python -m pip install "flwr>=1.5,<2"

    if [ ! -f nnUNet/pyproject.toml ]; then
        run_step "Fetching nnUNet submodule" fetch_submodule
    else
        skip_step "Fetching nnUNet submodule (already present)"
    fi

    # nnUNet in editable mode first, then FednnUNet
    run_step "Installing nnUNet and FednnUNet (editable)" install_local_packages

    echo
    echo "${DIM}Verifying installation:${RESET}"
    if ! verify 2>&1 | tee -a "$LOG_FILE" | sed 's/^/  /'; then
        echo "${RED}Verification failed - see $LOG_FILE${RESET}" >&2
        exit 1
    fi
    print_success
    echo
    echo "  ${BOLD}Installation complete!${RESET}"
    echo
}

# ---------------------------------------------------------------- main
banner

if ! command -v python >/dev/null 2>&1; then
    echo "${RED}python not found in PATH${RESET}" >&2
    exit 1
fi

install_fednnunet

echo "  How to use FednnUNet:      see ${BOLD}$SCRIPT_DIR/readme.md${RESET}"
echo
