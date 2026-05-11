#!/bin/bash
# =============================================================
# ailiance-mac-tuner — Setup for Mac Studio M4 Pro (512 Go)
# =============================================================
set -euo pipefail

echo "=== ailiance-mac-tuner Setup ==="
echo "Target: Mac Studio M4 Pro, 512 Go unified memory"
echo ""

# Check we're on Apple Silicon
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "ERROR: This script is designed for Apple Silicon (arm64)."
    echo "Detected: $(uname -m)"
    exit 1
fi

# Check available memory
MEM_GB=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f", $1/1024/1024/1024}')
echo "Detected RAM: ${MEM_GB} GB"
if [[ "$MEM_GB" -lt 128 ]]; then
    echo "WARNING: Less than 128 GB RAM detected."
    echo "Mistral Large 123B requires ~300 GB in bf16."
    echo "Consider using 4-bit QLoRA mode instead (--quantize flag in train.sh)."
fi

# Homebrew
if ! command -v brew &>/dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Python + uv
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    brew install uv
fi

# Create venv
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -d ".venv" ]]; then
    echo "Creating virtual environment..."
    uv venv .venv --python 3.12
fi

source .venv/bin/activate

echo "Installing dependencies..."
uv pip install -r requirements.txt

echo ""
echo "=== Setup complete ==="
echo "Next: ./download.sh"
