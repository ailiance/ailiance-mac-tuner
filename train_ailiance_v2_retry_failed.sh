#!/bin/bash
# AILIANCE v2 — Retry failed Qwen36 domains (Metal buffer crash on P2)
# Requires MLX fork with 3x Metal buffer limit installed
# Run AFTER Medium 3.5 training completes (or independently)

set -uo pipefail

TUNNER_DIR="/Users/clems/ailiance-mac-tuner"
DATA_DIR="/Users/clems/ailiance/data/hf-traced"
OUTPUT_DIR="${TUNNER_DIR}/output/ailiance-v2-curriculum"
LOG_DIR="${TUNNER_DIR}/logs"

cd "$TUNNER_DIR"
source .venv/bin/activate
export MLX_METAL_JIT=1

mkdir -p "$LOG_DIR"

QWEN_MODEL="${TUNNER_DIR}/models/Qwen3.6-35B-A3B-MLX-BF16"

# Domains that failed P2 with Metal buffer limit exceeded
FAILED_DOMAINS=(
  rust rust-embedded security-fenrir spice-sim
  typescript web-backend web-frontend yaml-json
)

train_phase() {
  local model="$1" prefix="$2" domain="$3"
  local phase="$4" seq="$5" lr="$6" iters="$7" batch="$8"

  local adapter_dir="${OUTPUT_DIR}/${prefix}-${domain}"
  local log_file="${LOG_DIR}/${prefix}-${domain}-curriculum.log"

  if [[ -f "${adapter_dir}/phase${phase}_done" ]]; then
    echo "[SKIP] ${prefix}-${domain} Phase ${phase} already done"
    return 0
  fi

  # Adaptive iters based on dataset size
  local n_examples
  n_examples=$(wc -l < "${DATA_DIR}/${domain}/train.jsonl")
  local dropout=0.01
  local adj_iters=$iters

  if [[ $n_examples -lt 500 ]]; then
    dropout=0.05
    case $phase in
      2) adj_iters=300 ;;
      3) adj_iters=150 ;;
    esac
    echo "[WARN] ${prefix}-${domain} — small dataset (${n_examples}), iters=${adj_iters}, dropout=${dropout}"
  elif [[ $n_examples -lt 1000 ]]; then
    dropout=0.03
    case $phase in
      2) adj_iters=500 ;;
      3) adj_iters=250 ;;
    esac
  fi

  local resume_flag=""
  [[ $phase -gt 1 ]] && resume_flag="resume: true"

  echo "[Phase ${phase}] ${prefix}-${domain} seq=${seq}, lr=${lr}, iters=${adj_iters} — $(date)"

  cat > "${adapter_dir}/config-phase${phase}.yaml" <<YAML
model: ${model}
train: true
fine_tune_type: lora
batch_size: ${batch}
iters: ${adj_iters}
learning_rate: ${lr}
lora_parameters:
  rank: 32
  alpha: 32
  dropout: ${dropout}
  scale: 32.0
num_layers: 16
max_seq_length: ${seq}
grad_checkpoint: true
grad_accumulation_steps: $((phase == 1 ? 16 : 8))
save_every: 200
steps_per_report: 10
steps_per_eval: 200
val_batches: 5
${resume_flag}
seed: 42
YAML

  mlx_lm.lora \
    -c "${adapter_dir}/config-phase${phase}.yaml" \
    --data "${DATA_DIR}/${domain}" \
    --adapter-path "$adapter_dir" \
    2>&1 | tee -a "$log_file"

  if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "[FAIL] ${prefix}-${domain} Phase ${phase} — $(date)"
    return 1
  fi
  touch "${adapter_dir}/phase${phase}_done"
  echo "[DONE] Phase ${phase} — $(date)"
}

echo "======================================================="
echo "  AILIANCE v2 RETRY FAILED — $(date)"
echo "  ${#FAILED_DOMAINS[@]} domains, P2+P3 only"
echo "  MLX fork with 3x Metal buffer limit required"
echo "======================================================="

# Verify MLX fork is installed
python3 -c "
import mlx.core as mx
info = mx.metal.device_info()
limit = info.get('resource_limit', 499000)
print(f'Metal resource limit: {limit}')
if limit <= 499000:
    print('WARNING: Stock MLX detected, may still crash on large buffers')
else:
    print('OK: MLX fork with extended buffer limit')
" 2>/dev/null || echo "Could not check MLX version"

echo ""

for domain in "${FAILED_DOMAINS[@]}"; do
  echo "============================================="
  echo "[RETRY] qwen36-${domain} — $(date)"
  echo "============================================="

  # P1 already done for all failed domains
  # Retry P2 (seq=1280, batch=1 to reduce Metal buffers)
  train_phase "$QWEN_MODEL" "qwen36" "$domain" 2 1280 5.0e-06 800 1
  [[ $? -ne 0 ]] && { echo "[ABORT] ${domain} P2 failed again, skipping P3"; continue; }

  # P3 (seq=4096, batch=1)
  train_phase "$QWEN_MODEL" "qwen36" "$domain" 3 4096 3.0e-06 500 1
  [[ $? -ne 0 ]] && { echo "[ABORT] ${domain} P3 failed"; continue; }

  echo "[COMPLETE] qwen36-${domain} retry done — $(date)"
  echo ""
done

echo "======================================================="
echo "  RETRY COMPLETE — $(date)"
echo "======================================================="
