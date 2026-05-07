#!/bin/bash
# EU-KIKI v2 — Sequential batch training
# Qwen3.6-35B-A3B (19 domains) then Medium 3.5 128B (28 domains)
# Resume-safe: skips domains that already have adapters.safetensors

set -uo pipefail

TUNNER_DIR="/Users/clems/KIKI-Mac_tunner"
DATA_DIR="/Users/clems/eu-kiki/data/hf-traced"
OUTPUT_DIR="${TUNNER_DIR}/output/eu-kiki-v2"
LOG_DIR="${TUNNER_DIR}/logs"

cd "$TUNNER_DIR"
source .venv/bin/activate

mkdir -p "$LOG_DIR"

# ---- Qwen3.6-35B-A3B config ----
QWEN_MODEL="${TUNNER_DIR}/models/Qwen3.6-35B-A3B-MLX-BF16"
QWEN_DOMAINS=(
  chat-fr electronics embedded emc-dsp-power iot
  kicad-dsl kicad-pcb llm-ops llm-orch math-gsm8k
  math-reasoning multilingual-eu platformio rust-embedded
  security-fenrir shell spice-sim stm32 traduction-tech
)

# ---- Medium 3.5 128B config ----
MEDIUM_MODEL="${TUNNER_DIR}/models/Mistral-Medium-3.5-128B-BF16"
MEDIUM_DOMAINS=(
  chat-fr cpp docker-devops electronics freecad
  html-css iot kicad-dsl kicad-pcb llm-ops
  llm-orch lua-upy ml-training multilingual-eu music-audio
  platformio python rust rust-embedded shell
  spice-sim sql stm32 traduction-tech typescript
  web-backend web-frontend yaml-json
)

train_domain() {
  local model="$1"
  local prefix="$2"
  local domain="$3"
  local num_layers="$4"
  local rank="$5"
  local alpha="$6"
  local max_seq="$7"
  local grad_accum="$8"

  local adapter_dir="${OUTPUT_DIR}/${prefix}-${domain}"
  local log_file="${LOG_DIR}/${prefix}-${domain}.log"
  local config_file="${adapter_dir}/train_config.yaml"

  # Skip if already trained
  if [[ -f "${adapter_dir}/adapters.safetensors" ]]; then
    echo "[SKIP] ${prefix}-${domain} — already has adapters.safetensors"
    return 0
  fi

  # Check data exists
  if [[ ! -d "${DATA_DIR}/${domain}" ]]; then
    echo "[SKIP] ${prefix}-${domain} — no data at ${DATA_DIR}/${domain}"
    return 0
  fi

  mkdir -p "$adapter_dir"

  # Write config YAML
  cat > "$config_file" <<YAML
model: ${model}
train: true
fine_tune_type: lora
batch_size: 1
iters: 500
learning_rate: 1.0e-05
lora_parameters:
  rank: ${rank}
  alpha: ${alpha}
  dropout: 0.05
  scale: 2.0
num_layers: ${num_layers}
max_seq_length: ${max_seq}
grad_checkpoint: true
grad_accumulation_steps: ${grad_accum}
save_every: 200
steps_per_report: 10
steps_per_eval: 200
val_batches: 5
seed: 42
YAML

  echo "============================================="
  echo "[START] ${prefix}-${domain} — $(date)"
  echo "============================================="

  mlx_lm.lora \
    -c "$config_file" \
    --data "${DATA_DIR}/${domain}" \
    --adapter-path "$adapter_dir" \
    2>&1 | tee "$log_file"

  local exit_code=${PIPESTATUS[0]}

  if [[ $exit_code -eq 0 ]]; then
    echo "[DONE] ${prefix}-${domain} — $(date)"
  else
    echo "[FAIL] ${prefix}-${domain} — exit code $exit_code — $(date)"
  fi

  echo ""
}

echo "======================================================="
echo "  EU-KIKI v2 BATCH TRAINING — $(date)"
echo "  Qwen3.6: ${#QWEN_DOMAINS[@]} domains"
echo "  Medium 3.5: ${#MEDIUM_DOMAINS[@]} domains"
echo "======================================================="
echo ""

# ---- Phase 1: Qwen3.6-35B-A3B ----
echo "=== PHASE 1: Qwen3.6-35B-A3B (num_layers=16, rank=32) ==="
echo ""

for domain in "${QWEN_DOMAINS[@]}"; do
  train_domain "$QWEN_MODEL" "qwen36" "$domain" 16 32 32 4096 8
done

echo ""
echo "=== PHASE 1 COMPLETE — $(date) ==="
echo ""

# ---- Phase 2: Medium 3.5 128B ----
echo "=== PHASE 2: Mistral Medium 3.5 128B (all layers, rank=16) ==="
echo ""

for domain in "${MEDIUM_DOMAINS[@]}"; do
  train_domain "$MEDIUM_MODEL" "medium35" "$domain" -1 16 32 2048 16
done

echo ""
echo "=== PHASE 2 COMPLETE — $(date) ==="
echo ""
echo "======================================================="
echo "  ALL DONE — $(date)"
echo "======================================================="
