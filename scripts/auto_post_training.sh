#!/bin/bash
# Auto-trigger fuse + gguf une fois le training terminé
# Usage: nohup ./scripts/auto_post_training.sh <TRAINING_PID> > logs/auto-post.log 2>&1 &

set -e
cd /Users/clems/ailiance-mac-tuner

TRAIN_PID="${1:-}"
if [ -z "$TRAIN_PID" ]; then
    echo "ERROR: pass training PID as arg"
    exit 1
fi

ADAPTER_DIR="output/qwen35-122b-opus-v3"
FUSED_DIR="output/qwen35-122b-opus-v3-fused"
GGUF_DIR="output/gguf"

echo "$(date) Watching training PID $TRAIN_PID"
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 60
    PROGRESS=$(grep -E "Iter [0-9]+:" logs/train-*.log 2>/dev/null | tail -1 | awk "{print \$2, \$5}")
    echo "$(date) Still running: $PROGRESS"
done
echo "$(date) Training PID $TRAIN_PID ended"

sleep 5
LAST_ADAPTER=$(ls -t "$ADAPTER_DIR"/adapters.safetensors 2>/dev/null)
if [ -z "$LAST_ADAPTER" ]; then
    echo "ERROR: no adapters.safetensors found in $ADAPTER_DIR"
    ls -la "$ADAPTER_DIR"/ 2>/dev/null
    exit 1
fi
echo "$(date) Found final adapter: $LAST_ADAPTER"

echo "=== STEP 1: Fuse LoRA into BF16 ==="
./scripts/pipeline_122b.sh fuse
echo "$(date) Fuse done."
du -sh "$FUSED_DIR" 2>/dev/null

echo "=== STEP 2: Export to GGUF Q4_K_M ==="
./scripts/pipeline_122b.sh gguf
echo "$(date) GGUF export done."

echo "=== FINAL ARTIFACT ==="
ls -lh "$GGUF_DIR"/qwen35-122b-opus-v3*.gguf 2>/dev/null

echo "$(date) Pipeline complete. Ready to publish to HuggingFace."
