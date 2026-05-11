#!/bin/bash
# Training curriculum : court → moyen → long
# Chaque phase reprend les adapters de la phase precedente
# Usage : ./scripts/train_curriculum.sh [prepare|phase1|phase2|phase3|all]

set -e
cd /Users/clems/ailiance-mac-tuner
source .venv/bin/activate

MODEL="models/Qwen3.5-122B-A10B-BF16"
ADAPTER_DIR="output/qwen35-122b-opus-v3-curriculum"
LORA_CONFIG="output/qwen35-122b-opus-v3/lora_config.yaml"

prepare() {
    echo "=== Preparation du curriculum ==="
    python scripts/prepare_curriculum.py data/final-opus-v3-1 "$MODEL"
}

run_phase() {
    local PHASE=$1
    local DATA=$2
    local SEQ=$3
    local ITERS=$4
    local LR=$5
    local RESUME=$6

    echo ""
    echo "=== $PHASE : seq=$SEQ, iters=$ITERS, lr=$LR ==="

    ARGS=(
        --model "$MODEL"
        --train
        --data "$DATA"
        --iters "$ITERS"
        --learning-rate "$LR"
        --batch-size 1
        --adapter-path "$ADAPTER_DIR/adapters"
        --save-every 50
        --grad-checkpoint
        --max-seq-length "$SEQ"
        --clear-cache-threshold 2
    )

    # Config LoRA si existe
    if [ -f "$LORA_CONFIG" ]; then
        ARGS+=(-c "$LORA_CONFIG")
    fi

    # Resume si checkpoint existe
    if [ -n "$RESUME" ] && [ -f "$RESUME" ]; then
        ARGS+=(--resume-adapter-file "$RESUME")
        echo "  Resume depuis: $RESUME"
    fi

    mkdir -p "$ADAPTER_DIR/adapters"
    PYTHONUNBUFFERED=1 python -m mlx_lm lora "${ARGS[@]}" 2>&1 | tee "logs/curriculum-${PHASE}.log"
}

phase1() {
    run_phase "phase1-short" \
        "data/curriculum/phase1-short" \
        512 \
        1000 \
        1e-5 \
        ""
}

phase2() {
    # Resume depuis phase 1
    local CKPT="$ADAPTER_DIR/adapters/adapters.safetensors"
    run_phase "phase2-medium" \
        "data/curriculum/phase2-medium" \
        1024 \
        1000 \
        8e-6 \
        "$CKPT"
}

phase3() {
    # Resume depuis phase 2
    local CKPT="$ADAPTER_DIR/adapters/adapters.safetensors"
    run_phase "phase3-long" \
        "data/curriculum/phase3-long" \
        2048 \
        1000 \
        5e-6 \
        "$CKPT"
}

case "${1:-all}" in
    prepare) prepare ;;
    phase1)  phase1 ;;
    phase2)  phase2 ;;
    phase3)  phase3 ;;
    all)
        prepare
        phase1
        phase2
        phase3
        echo ""
        echo "=== Curriculum complet ==="
        echo "Adapters: $ADAPTER_DIR/adapters/"
        ;;
    *)
        echo "Usage: $0 [prepare|phase1|phase2|phase3|all]"
        echo ""
        echo "Phases:"
        echo "  phase1  < 512 tokens   LR=1e-5   1000 iters"
        echo "  phase2  512-1024       LR=8e-6   1000 iters (resume phase1)"
        echo "  phase3  1024+          LR=5e-6   1000 iters (resume phase2)"
        ;;
esac
