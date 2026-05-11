#!/bin/bash
# Pipeline Qwen3.5-122B-A10B-Opus-v3
# Notre propre modele Opus reasoning sur 122B MoE
# Utilisation : ./scripts/pipeline_122b.sh [distill|merge|train|fuse|gguf|all|status]

set -e
cd /Users/clems/ailiance-mac-tuner
source .venv/bin/activate

CONFIG="configs/mlx-lm-qwen35-122b-opus-v3.yaml"
DATA_DIR="data/final-opus-v3"
ADAPTER_DIR="output/qwen35-122b-opus-v3"
MODEL="models/Qwen3.5-122B-A10B-4bit"
FUSED_DIR="output/qwen35-122b-opus-v3-fused"
GGUF_DIR="output/gguf"

distill_data() {
    echo "=== Etape 1 : Distillation avec 35B-Opus (mlx-vlm) ==="
    echo "Le 35B-Opus genere des traces de raisonnement a 61 tok/s"
    echo ""
    python scripts/distill_mlxvlm.py
}

merge_data() {
    echo "=== Etape 2 : Fusion datasets ==="
    SOURCES=()

    [ -d "data/Opus-4.6-Reasoning-3000x-filtered" ] && SOURCES+=("Opus-4.6-Reasoning-3000x-filtered")
    [ -d "data/Opus-4.6-reasoning-sft-12k-chat" ] && SOURCES+=("Opus-4.6-reasoning-sft-12k-chat")

    # Distillations
    for d in distilled-mlxvlm-35b distilled-mistral-large-123b distilled-qwen35-35b-opus distilled-qwen35-35b-opus-batch2; do
        if [ -f "data/$d/all.jsonl" ]; then
            [ ! -f "data/$d/train.jsonl" ] && cp "data/$d/all.jsonl" "data/$d/train.jsonl"
            SOURCES+=("$d")
            COUNT=$(wc -l < "data/$d/all.jsonl")
            echo "  + $d ($COUNT exemples)"
        fi
    done

    python scripts/merge_datasets.py \
        --sources "${SOURCES[@]}" \
        --output final-opus-v3 \
        --deduplicate \
        --val-split 0.05

    TOTAL=$(wc -l < "$DATA_DIR/train.jsonl")
    echo "Dataset final : $TOTAL exemples"
}

train_model() {
    echo "=== Etape 3 : Fine-tune 122B-A10B ==="
    echo "Config : $CONFIG"
    echo "ATTENTION : ~300 Go de RAM Metal necessaires"
    echo ""
    python -m mlx_lm lora --config "$CONFIG"
}

fuse_model() {
    echo "=== Etape 4 : Fusion LoRA ==="
    python -m mlx_lm fuse \
        --model "$MODEL" \
        --adapter-path "$ADAPTER_DIR" \
        --save-path "$FUSED_DIR"
    echo "Fuse : $FUSED_DIR"
}

export_gguf() {
    echo "=== Etape 5 : Export GGUF ==="
    mkdir -p "$GGUF_DIR"

    CONVERT_SCRIPT=$(find /tmp/llama-cpp-gpu -name "convert_hf_to_gguf.py" 2>/dev/null | head -1)
    QUANTIZE=$(find /tmp/llama-cpp-gpu/build -name "llama-quantize" 2>/dev/null | head -1)

    if [ -z "$CONVERT_SCRIPT" ] || [ -z "$QUANTIZE" ]; then
        echo "llama.cpp introuvable. Clonage..."
        cd /tmp && git clone https://github.com/ggml-org/llama.cpp.git llama-cpp-gguf 2>/dev/null || true
        cd llama-cpp-gguf && cmake -B build -DGGML_METAL=ON && cmake --build build -j$(sysctl -n hw.ncpu)
        CONVERT_SCRIPT="/tmp/llama-cpp-gguf/convert_hf_to_gguf.py"
        QUANTIZE="/tmp/llama-cpp-gguf/build/bin/llama-quantize"
        cd /Users/clems/ailiance-mac-tuner
    fi

    echo "Conversion → GGUF F16..."
    python "$CONVERT_SCRIPT" "$FUSED_DIR" --outfile "$GGUF_DIR/qwen35-122b-opus-v3-f16.gguf" --outtype f16

    echo "Quantification → Q4_K_M..."
    "$QUANTIZE" "$GGUF_DIR/qwen35-122b-opus-v3-f16.gguf" "$GGUF_DIR/qwen35-122b-opus-v3-Q4_K_M.gguf" Q4_K_M

    rm -f "$GGUF_DIR/qwen35-122b-opus-v3-f16.gguf"
    echo ""
    echo "=== Modele final ==="
    ls -lh "$GGUF_DIR/qwen35-122b-opus-v3"*.gguf
}

status() {
    echo "=== Qwen3.5-122B-A10B-Opus-v3 ==="
    echo ""
    echo "Modele base :"
    du -sh "$MODEL" 2>/dev/null || echo "  Pas encore telecharge"
    echo ""
    echo "Distillation mlx-vlm :"
    wc -l data/distilled-mlxvlm-35b/all.jsonl 2>/dev/null || echo "  0 exemples"
    echo ""
    echo "Dataset final :"
    wc -l "$DATA_DIR/train.jsonl" "$DATA_DIR/valid.jsonl" 2>/dev/null || echo "  Pas encore fusionne"
    echo ""
    echo "Adapter LoRA :"
    ls -lh "$ADAPTER_DIR/adapters.safetensors" 2>/dev/null || echo "  Pas encore entraine"
    echo ""
    echo "GGUF :"
    ls -lh "$GGUF_DIR/qwen35-122b-opus-v3"*.gguf 2>/dev/null || echo "  Pas encore exporte"
}

case "${1:-status}" in
    distill) distill_data ;;
    merge)   merge_data ;;
    train)   train_model ;;
    fuse)    fuse_model ;;
    gguf)    export_gguf ;;
    all)
        merge_data
        train_model
        fuse_model
        export_gguf
        echo ""
        echo "=== Qwen3.5-122B-A10B-Opus-v3 pret ! ==="
        ;;
    status)  status ;;
    *)
        echo "Utilisation : $0 [distill|merge|train|fuse|gguf|all|status]"
        ;;
esac
