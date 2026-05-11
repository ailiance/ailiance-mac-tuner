#!/bin/bash
# Phase 5 : Merge des adaptateurs LoRA
# Fusionne les adapters SFT + SimPO + GRPO dans le modele final
# Usage: ./scripts/merge_lora.sh

set -e
cd /Users/clems/ailiance-mac-tuner
source .venv/bin/activate

echo "=== Phase 5 : Merge final ==="

# Option A : Prendre le dernier adapteur (GRPO)
FINAL_ADAPTER="output/qwen35-122b-opus-v3-grpo/final"
if [ ! -d "$FINAL_ADAPTER" ]; then
    FINAL_ADAPTER="output/qwen35-122b-opus-v3-simpo/final"
fi
if [ ! -d "$FINAL_ADAPTER" ]; then
    FINAL_ADAPTER="output/qwen35-122b-opus-v3-curriculum/adapters"
fi

echo "Using adapter: $FINAL_ADAPTER"

# Fuse LoRA into base model
python -m mlx_lm fuse \
    --model models/Qwen3.5-122B-A10B-BF16 \
    --adapter-path "$FINAL_ADAPTER" \
    --save-path output/qwen35-122b-opus-v3-final

echo "Fused model: output/qwen35-122b-opus-v3-final"
du -sh output/qwen35-122b-opus-v3-final/

# Export GGUF Q4_K_M
echo ""
echo "=== Export GGUF ==="
mkdir -p output/gguf

CONVERT=$(find /tmp -name "convert_hf_to_gguf.py" 2>/dev/null | head -1)
QUANTIZE=$(find /tmp -name "llama-quantize" 2>/dev/null | head -1)

if [ -n "$CONVERT" ] && [ -n "$QUANTIZE" ]; then
    python "$CONVERT" output/qwen35-122b-opus-v3-final \
        --outfile output/gguf/qwen35-122b-opus-v3-f16.gguf --outtype f16
    "$QUANTIZE" output/gguf/qwen35-122b-opus-v3-f16.gguf \
        output/gguf/qwen35-122b-opus-v3-Q4_K_M.gguf Q4_K_M
    rm output/gguf/qwen35-122b-opus-v3-f16.gguf
    echo "GGUF: output/gguf/qwen35-122b-opus-v3-Q4_K_M.gguf"
    ls -lh output/gguf/qwen35-122b-opus-v3-Q4_K_M.gguf
else
    echo "llama.cpp non trouve, skip GGUF export"
fi
