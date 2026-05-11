#!/bin/bash
# =============================================================
# ailiance-mac-tuner — Export: merge LoRA + convert GGUF + quantize
# =============================================================
# Usage:
#   ./export.sh                                 # Full export (merge + GGUF)
#   ./export.sh --config configs/qwen-27b.yaml  # Other model
#   ./export.sh --quants Q4_K_M,Q6_K,Q8_0      # Custom quant list
#   ./export.sh --quick                         # Quick test: LoRA GGUF only (no merge)
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

CONFIG="configs/mistral-large.yaml"
QUANTS="Q6_K,Q8_0"
QUICK=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --quants) QUANTS="$2"; shift 2 ;;
        --quick) QUICK=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Parse config
MODEL_ID=$(grep "^model_id:" "$SCRIPT_DIR/$CONFIG" | awk '{print $2}')
OUTPUT_DIR=$(grep "^output_dir:" "$SCRIPT_DIR/$CONFIG" | awk '{print $2}')
MODEL_NAME=$(basename "$MODEL_ID")
MERGED_DIR="$SCRIPT_DIR/models/${MODEL_NAME}-Opus-Reasoning"

# Clone llama.cpp if needed (used by both paths)
if [[ ! -d "$SCRIPT_DIR/llama.cpp" ]]; then
    echo "Cloning llama.cpp for conversion tools..."
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$SCRIPT_DIR/llama.cpp"
fi

if [[ "$QUICK" == "true" ]]; then
    # =========================================================
    # Quick path: convert LoRA adapter to GGUF for runtime use
    # Serve with: llama-server -m base.gguf --lora adapter.gguf
    # No 250GB merge needed — fast iteration for testing quality
    # =========================================================
    echo "=== ailiance-mac-tuner Quick Export (LoRA GGUF) ==="
    echo "Adapter: $OUTPUT_DIR/final-lora"
    echo ""

    ADAPTER_DIR="$SCRIPT_DIR/$OUTPUT_DIR/final-lora"
    LORA_GGUF="$SCRIPT_DIR/models/gguf/${MODEL_NAME}-Opus-LoRA.gguf"
    mkdir -p "$SCRIPT_DIR/models/gguf"

    # Convert MLX adapter to GGUF LoRA
    CONVERT_LORA="$SCRIPT_DIR/llama.cpp/convert_lora_to_gguf.py"
    if [[ ! -f "$CONVERT_LORA" ]]; then
        echo "ERROR: convert_lora_to_gguf.py not found in llama.cpp"
        exit 1
    fi

    echo "[1/1] Converting LoRA adapter to GGUF..."
    python3 "$CONVERT_LORA" \
        --base "$SCRIPT_DIR/models/$MODEL_NAME" \
        --outtype q8_0 \
        "$ADAPTER_DIR"

    # Move output to expected location
    if [[ -f "$ADAPTER_DIR/ggml-adapter-model.bin" ]]; then
        mv "$ADAPTER_DIR/ggml-adapter-model.bin" "$LORA_GGUF"
    fi

    echo ""
    echo "=== Quick export done! ==="
    echo "LoRA GGUF: $LORA_GGUF"
    echo ""
    echo "Usage (requires a base GGUF of $MODEL_NAME):"
    echo "  llama-server -m base-model.gguf --lora $LORA_GGUF --host 0.0.0.0"
    exit 0
fi

# =========================================================
# Full path: merge LoRA → full model → GGUF + quantize
# =========================================================
echo "=== ailiance-mac-tuner Full Export ==="
echo "Base model: models/$MODEL_NAME"
echo "Adapter: $OUTPUT_DIR/final-lora"
echo "Merged → $MERGED_DIR"
echo "GGUF quants: $QUANTS"
echo ""

# Step 1: Merge LoRA
echo "[1/2] Merging LoRA adapter into base model..."
python3 "$SCRIPT_DIR/scripts/merge_lora.py" \
    --model "$SCRIPT_DIR/models/$MODEL_NAME" \
    --adapter "$SCRIPT_DIR/$OUTPUT_DIR/final-lora" \
    --output "$MERGED_DIR"

# Step 2: Convert to GGUF + quantize
echo ""
echo "[2/2] Converting to GGUF + quantizing..."

python3 "$SCRIPT_DIR/scripts/convert_gguf.py" \
    --model "$MERGED_DIR" \
    --output "$SCRIPT_DIR/models/gguf" \
    --quants "$QUANTS" \
    --llama-cpp "$SCRIPT_DIR/llama.cpp"

echo ""
echo "=== Export complete! ==="
echo "GGUF files ready in: $SCRIPT_DIR/models/gguf/"
echo ""
echo "To use with llama.cpp:"
echo "  llama-server --model models/gguf/${MODEL_NAME}-Opus-Reasoning-Q8_0.gguf --host 0.0.0.0"
echo ""
echo "To copy to your NFS share:"
echo "  cp models/gguf/*.gguf /mnt/models/"
