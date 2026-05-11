#!/bin/bash
# =============================================================
# Download model + dataset for fine-tuning
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

# Ensure hf CLI is available
if ! command -v hf &>/dev/null; then
    echo "Installing hf CLI..."
    pip install huggingface_hub[cli]
fi

# Default config
CONFIG="${1:-configs/mistral-large.yaml}"

# Parse YAML config (simple grep-based, no external deps)
MODEL_ID=$(grep "^model_id:" "$SCRIPT_DIR/$CONFIG" | awk '{print $2}')
DATASET_ID=$(grep "^dataset_id:" "$SCRIPT_DIR/$CONFIG" | awk '{print $2}')

echo "=== ailiance-mac-tuner Download ==="
echo "Config: $CONFIG"
echo "Model: $MODEL_ID"
echo "Dataset: $DATASET_ID"
echo ""

# Download model
MODEL_DIR="$SCRIPT_DIR/models/$(basename $MODEL_ID)"
if [[ -d "$MODEL_DIR" ]] && [[ $(ls "$MODEL_DIR"/*.safetensors 2>/dev/null | wc -l) -gt 0 ]]; then
    echo "Model already downloaded at $MODEL_DIR"
else
    echo "Downloading model (this may take a while for large models)..."
    hf download "$MODEL_ID" --local-dir "$MODEL_DIR"
fi

# Download dataset
echo ""
echo "Downloading dataset..."
python3 -c "
from datasets import load_dataset
ds = load_dataset('$DATASET_ID', split='train')
ds.save_to_disk('$SCRIPT_DIR/data/$(basename $DATASET_ID)')
print(f'Dataset saved: {len(ds)} examples')
"

echo ""
echo "=== Download complete ==="
echo "Model: $MODEL_DIR"
echo "Dataset: $SCRIPT_DIR/data/$(basename $DATASET_ID)"
echo "Next: ./train.sh"
