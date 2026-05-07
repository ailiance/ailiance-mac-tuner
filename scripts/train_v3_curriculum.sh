#!/usr/bin/env bash
# V-35B v3 — Full curriculum, 40 layers, adaptive rank
# Rank 32 for foundations, 16 for coding, 8 for niches
set -euo pipefail

MODEL="models/Qwen3.6-35B-A3B"
DATA="data/micro-kiki"
OUTPUT="output/micro-kiki/lora-qwen36-35b-v3"
PYTHON="/opt/homebrew/bin/python3.12"
NUM_LAYERS=40  # -1 = all layers in mlx_lm

mkdir -p "$OUTPUT"

echo "================================================================"
echo "V-35B v3 — 40 layers, adaptive rank, curriculum"
echo "================================================================"

# Curriculum with rank:iters per domain
# Format: domain:rank:iters
CURRICULUM=(
  # Phase 1 — Foundations (rank 32, 1000 iters)
  "chat-fr:32:1000"
  "reasoning:32:1000"
  # Phase 2 — Coding core (rank 16, 1000 iters)
  "python:16:1000"
  "typescript:16:500"
  "cpp:16:500"
  "rust:16:500"
  # Phase 3 — Coding secondary (rank 8, 500 iters)
  "html-css:8:500"
  "shell:8:500"
  "sql:8:500"
  "yaml-json:8:200"
  "docker:8:500"
  "kicad-dsl:8:500"
  "spice:8:200"
  "lua-upy:8:200"
  # Phase 4 — Technical (rank 8, adaptive iters)
  "embedded:16:500"
  "stm32:8:300"
  "iot:8:200"
  "freecad:8:500"
  "platformio:8:100"
  "power:16:500"
  "emc:8:200"
  "dsp:8:200"
  "spice-sim:8:100"
  "electronics:16:500"
  "kicad-pcb:8:500"
  # Phase 5 — Applications (rank 8)
  "web-frontend:8:200"
  "web-backend:8:200"
  "music-audio:8:100"
  "devops:8:200"
  "llm-orch:8:200"
  # Phase 6 — Complements (rank 8)
  "math:8:200"
  "security:8:300"
  # Phase 7 — New domains (rank 8)
  "components:16:500"
  "llm-ops:8:200"
  "ml-training:8:200"
)

for i in "${!CURRICULUM[@]}"; do
  entry="${CURRICULUM[$i]}"
  domain="${entry%%:*}"
  rest="${entry#*:}"
  rank="${rest%%:*}"
  iters="${rest#*:}"
  idx=$((i + 1))
  adapter="$OUTPUT/$domain"
  alpha=$((rank * 2))

  [ ! -f "$DATA/$domain/train.jsonl" ] && echo "[$idx] SKIP $domain (no data)" && continue
  [ -f "$adapter/adapters.safetensors" ] && echo "[$idx] SKIP $domain (done)" && continue

  n=$(wc -l < "$DATA/$domain/train.jsonl")

  echo ""
  echo "[$idx/${#CURRICULUM[@]}] $domain (n=$n, rank=$rank, alpha=$alpha, iters=$iters, layers=$NUM_LAYERS)"

  # Generate per-domain YAML config
  cat > "$OUTPUT/config-$domain.yaml" << YAMLEOF
model: "$MODEL"
data: "$DATA/$domain"
train: true
iters: $iters
batch_size: 1
learning_rate: 2e-5
adapter_path: "$adapter"
max_seq_length: 1024
num_layers: $NUM_LAYERS
steps_per_report: 50
steps_per_eval: 100
grad_checkpoint: true
lora_parameters:
  rank: $rank
  alpha: $alpha
  dropout: 0.01
  scale: 20.0
YAMLEOF

  $PYTHON -c "
import mlx.core as mx
mx.set_memory_limit(460 * 1024**3)
mx.set_cache_limit(32 * 1024**3)

import sys
sys.argv = ['mlx_lm', 'lora', '-c', '$OUTPUT/config-$domain.yaml']
from mlx_lm.cli import main
main()
" 2>&1 | tee "$OUTPUT/log-$domain.txt"

  echo "[$idx] $domain DONE (rank=$rank)"
  sleep 5
done

echo "================================================================"
echo "ALL COMPLETE"
echo "================================================================"
