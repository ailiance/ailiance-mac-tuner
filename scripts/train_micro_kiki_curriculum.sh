#!/bin/bash
set -euo pipefail
cd ~/ailiance-mac-tuner
source .venv/bin/activate

echo "=== micro-kiki curriculum training ==="
echo "Stack: chat-fr | Rank: 64 | 3 phases"
echo ""

echo "[Phase 1/3] seq=512, 500 iters, LR=8e-6"
python -m mlx_lm lora --config configs/mlx-lm-micro-kiki-phase1.yaml
echo "[Phase 1] DONE"
echo ""

echo "[Phase 2/3] seq=1280, 1000 iters, LR=5e-6 (resume)"
python -m mlx_lm lora --config configs/mlx-lm-micro-kiki-phase2.yaml
echo "[Phase 2] DONE"
echo ""

echo "[Phase 3/3] seq=4096, 500 iters, LR=3e-6 (resume)"
python -m mlx_lm lora --config configs/mlx-lm-micro-kiki-phase3.yaml
echo "[Phase 3] DONE"
echo ""

echo "=== Curriculum complete ==="
ls -lh output/micro-kiki/stack-01-chat-fr/adapters.safetensors
