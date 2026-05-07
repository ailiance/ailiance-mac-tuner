#!/bin/bash
# train_coding_v2.sh — retrain 5 coding adapters on clean (stub-filtered) data
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Coding V2 retraining started: $(date) ==="
echo "Domains: cpp rust shell typescript html-css"
echo ""

for domain in cpp rust shell typescript html-css; do
  echo "=== Training $domain-v2 at $(date) ==="
  .venv/bin/python -m mlx_lm lora \
    -c output/micro-kiki/lora-qwen36-35b-v4-sota/config-$domain-v2.yaml
  echo "=== Done $domain-v2 at $(date) ==="
  echo ""
done

echo "=== All 5 coding domains retrained at $(date) ==="
