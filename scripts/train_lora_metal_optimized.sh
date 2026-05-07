#!/usr/bin/env bash
# V-35B training with FULL Metal optimization
# - mx.set_memory_limit + mx.set_cache_limit via wrapper
# - 16 layers (2× previous 8)
# - 1000 iters for foundations, 500 for others
# - Cosine LR decay via mlx_lm built-in
set -euo pipefail

MODEL="models/Qwen3.6-35B-A3B"
DATA="data/micro-kiki"
OUTPUT="output/micro-kiki/lora-qwen36-35b-v2"
PYTHON="/opt/homebrew/bin/python3.12"

# Foundation domains get more iters
declare -A FOUNDATION_ITERS
FOUNDATION_ITERS=(
  [chat-fr]=1000
  [reasoning]=1000
  [python]=1000
)

CURRICULUM=(
  chat-fr reasoning python typescript cpp rust
  html-css shell sql yaml-json docker kicad-dsl spice lua-upy
  embedded stm32 iot freecad platformio power emc dsp
  spice-sim electronics kicad-pcb
  web-frontend web-backend music-audio devops llm-orch
  math security
  components llm-ops ml-training
)

mkdir -p "$OUTPUT"

echo "================================================================"
echo "V-35B v2 — Metal optimized, 16 layers, cosine LR"
echo "================================================================"

for i in "${!CURRICULUM[@]}"; do
  domain="${CURRICULUM[$i]}"
  idx=$((i + 1))
  adapter="$OUTPUT/$domain"

  [ ! -f "$DATA/$domain/train.jsonl" ] && echo "[$idx] SKIP $domain (no data)" && continue
  [ -f "$adapter/adapters.safetensors" ] && echo "[$idx] SKIP $domain (done)" && continue

  n=$(wc -l < "$DATA/$domain/train.jsonl")

  # Adaptive iters: foundations 1000, others min(500, max(100, n/20))
  if [[ -v "FOUNDATION_ITERS[$domain]" ]]; then
    iters="${FOUNDATION_ITERS[$domain]}"
  else
    iters=$(python3 -c "print(min(500, max(100, $n // 20)))")
  fi

  echo ""
  echo "[$idx/${#CURRICULUM[@]}] $domain ($n ex, $iters iters, 16 layers)"

  # Wrapper that sets Metal limits before running mlx_lm
  $PYTHON -c "
import mlx.core as mx

# Metal optimization — critical for M3 Ultra 512GB
mx.set_memory_limit(460 * 1024**3)   # 460 GB max Metal allocation
mx.set_cache_limit(32 * 1024**3)     # 32 GB Metal cache

import sys
sys.argv = [
    'mlx_lm', 'lora',
    '--model', '$MODEL',
    '--data', '$DATA/$domain',
    '--train',
    '--iters', '$iters',
    '--batch-size', '1',
    '--learning-rate', '2e-5',
    '--adapter-path', '$adapter',
    '--max-seq-length', '1024',
    '--num-layers', '16',
    '--steps-per-report', '50',
    '--steps-per-eval', '100',
    '--grad-checkpoint',
]
from mlx_lm.cli import main
main()
" 2>&1 | tee "$OUTPUT/log-$domain.txt"

  echo "[$idx] $domain DONE"
  sleep 5
done

echo "================================================================"
echo "ALL COMPLETE"
echo "================================================================"
