#!/bin/bash
# =============================================================
# ailiance-mac-tuner — Training launcher
# =============================================================
# Usage:
#   ./train.sh                              # Mistral Large (default)
#   ./train.sh --config configs/qwen-27b.yaml  # Other model
#   ./train.sh --resume                     # Resume from checkpoint
#   Ctrl+C to pause (checkpoint saved automatically)
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

CONFIG="configs/mistral-large.yaml"
RESUME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --resume) RESUME="--resume"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== ailiance-mac-tuner ==="
echo "Config: $CONFIG"
echo "Resume: ${RESUME:-no}"
echo ""
echo "Controls:"
echo "  Ctrl+C  → pause (saves checkpoint)"
echo "  ./train.sh --resume  → continue"
echo ""

python3 "$SCRIPT_DIR/scripts/train_mlx.py" --config "$CONFIG" $RESUME
