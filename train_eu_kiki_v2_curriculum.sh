#!/bin/bash
# EU-KIKI v2 — Curriculum batch training (3 phases)
# Phase 1: seq=512, lr=8e-6 (foundations)
# Phase 2: seq=1280, lr=5e-6 (medium context)
# Phase 3: seq=4096/2048, lr=3e-6 (full context)
# Resume-safe: skips domains that already have phase3 checkpoint

set -uo pipefail

TUNNER_DIR="/Users/clems/KIKI-Mac_tunner"
DATA_DIR="/Users/clems/eu-kiki/data/hf-traced"
OUTPUT_DIR="${TUNNER_DIR}/output/eu-kiki-v2-curriculum"
LOG_DIR="${TUNNER_DIR}/logs"

cd "$TUNNER_DIR"
source .venv/bin/activate

mkdir -p "$LOG_DIR"

# ---- Qwen3.6-35B-A3B config ----
QWEN_MODEL="${TUNNER_DIR}/models/Qwen3.6-35B-A3B-MLX-BF16"
QWEN_DOMAINS=(
  chat-fr cpp docker-devops embedded emc-dsp-power
  freecad html-css iot kicad-dsl kicad-pcb
  llm-ops llm-orch lua-upy math-gsm8k math-reasoning
  ml-training multilingual-eu music-audio platformio
  rust rust-embedded security-fenrir shell spice-sim
  sql traduction-tech typescript web-backend
  web-frontend yaml-json
)
# Removed: stm32 (no train.jsonl)

# ---- Medium 3.5 128B config ----
MEDIUM_MODEL="${TUNNER_DIR}/models/Mistral-Medium-3.5-128B-BF16"
MEDIUM_DOMAINS=(
  chat-fr cpp docker-devops embedded
  emc-dsp-power freecad html-css iot kicad-dsl
  kicad-pcb llm-ops llm-orch lua-upy math-gsm8k
  math-reasoning ml-training multilingual-eu music-audio
  platformio python rust rust-embedded security-fenrir
  shell spice-sim sql traduction-tech
  typescript web-backend web-frontend yaml-json
)
# Removed: electronics, stm32 (no train.jsonl)

train_curriculum() {
  local model="$1"
  local prefix="$2"
  local domain="$3"
  local rank="$4"
  local alpha="$5"
  local num_layers="$6"
  local max_seq_phase3="$7"  # 4096 for Qwen, 2048 for Medium

  local adapter_dir="${OUTPUT_DIR}/${prefix}-${domain}"
  local log_file="${LOG_DIR}/${prefix}-${domain}-curriculum.log"

  # Skip if phase 3 already completed
  if [[ -f "${adapter_dir}/phase3_done" ]]; then
    echo "[SKIP] ${prefix}-${domain} — curriculum already complete"
    return 0
  fi

  # Check data exists
  if [[ ! -f "${DATA_DIR}/${domain}/train.jsonl" ]]; then
    echo "[SKIP] ${prefix}-${domain} — no train.jsonl at ${DATA_DIR}/${domain}"
    return 0
  fi

  # Adaptive iters/dropout based on dataset size
  local n_examples
  n_examples=$(wc -l < "${DATA_DIR}/${domain}/train.jsonl")
  local p1_iters=500 p2_iters=800 p3_iters=500 dropout=0.01

  if [[ $n_examples -lt 100 ]]; then
    p1_iters=100; p2_iters=150; p3_iters=80; dropout=0.08
    echo "[WARN] ${prefix}-${domain} — tiny dataset (${n_examples} examples), reduced iters ${p1_iters}/${p2_iters}/${p3_iters}, dropout=${dropout}"
  elif [[ $n_examples -lt 500 ]]; then
    p1_iters=200; p2_iters=300; p3_iters=150; dropout=0.05
    echo "[WARN] ${prefix}-${domain} — small dataset (${n_examples} examples), reduced iters ${p1_iters}/${p2_iters}/${p3_iters}, dropout=${dropout}"
  elif [[ $n_examples -lt 1000 ]]; then
    p1_iters=300; p2_iters=500; p3_iters=250; dropout=0.03
    echo "[INFO] ${prefix}-${domain} — moderate dataset (${n_examples} examples), adjusted iters ${p1_iters}/${p2_iters}/${p3_iters}, dropout=${dropout}"
  else
    echo "[INFO] ${prefix}-${domain} — full dataset (${n_examples} examples), standard iters"
  fi

  mkdir -p "$adapter_dir"

  echo "============================================="
  echo "[CURRICULUM] ${prefix}-${domain} — $(date)"
  echo "============================================="

  # ---- Phase 1: foundations (seq=512) ----
  if [[ ! -f "${adapter_dir}/phase1_done" ]]; then
    echo "[Phase 1] seq=512, lr=8e-6, iters=${p1_iters}, dropout=${dropout} — $(date)"

    cat > "${adapter_dir}/config-phase1.yaml" <<YAML
model: ${model}
train: true
fine_tune_type: lora
batch_size: 1
iters: ${p1_iters}
learning_rate: 8.0e-06
lora_parameters:
  rank: ${rank}
  alpha: ${alpha}
  dropout: ${dropout}
  scale: ${alpha}.0
num_layers: ${num_layers}
max_seq_length: 512
grad_checkpoint: true
grad_accumulation_steps: 16
save_every: 200
steps_per_report: 10
steps_per_eval: 200
val_batches: 5
seed: 42
YAML

    mlx_lm.lora \
      -c "${adapter_dir}/config-phase1.yaml" \
      --data "${DATA_DIR}/${domain}" \
      --adapter-path "$adapter_dir" \
      2>&1 | tee -a "$log_file"

    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
      echo "[FAIL] ${prefix}-${domain} Phase 1 — $(date)"
      return 1
    fi
    touch "${adapter_dir}/phase1_done"
    echo "[DONE] Phase 1 — $(date)"
  else
    echo "[SKIP] Phase 1 already done"
  fi

  # ---- Phase 2: medium context (seq=1280) ----
  if [[ ! -f "${adapter_dir}/phase2_done" ]]; then
    echo "[Phase 2] seq=1280, lr=5e-6, iters=${p2_iters}, dropout=${dropout} — $(date)"

    cat > "${adapter_dir}/config-phase2.yaml" <<YAML
model: ${model}
train: true
fine_tune_type: lora
batch_size: 2
iters: ${p2_iters}
learning_rate: 5.0e-06
lora_parameters:
  rank: ${rank}
  alpha: ${alpha}
  dropout: ${dropout}
  scale: ${alpha}.0
num_layers: ${num_layers}
max_seq_length: 1280
grad_checkpoint: true
grad_accumulation_steps: 8
save_every: 200
steps_per_report: 10
steps_per_eval: 200
val_batches: 5
resume: true
seed: 42
YAML

    mlx_lm.lora \
      -c "${adapter_dir}/config-phase2.yaml" \
      --data "${DATA_DIR}/${domain}" \
      --adapter-path "$adapter_dir" \
      2>&1 | tee -a "$log_file"

    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
      echo "[FAIL] ${prefix}-${domain} Phase 2 — $(date)"
      return 1
    fi
    touch "${adapter_dir}/phase2_done"
    echo "[DONE] Phase 2 — $(date)"
  else
    echo "[SKIP] Phase 2 already done"
  fi

  # ---- Phase 3: full context ----
  if [[ ! -f "${adapter_dir}/phase3_done" ]]; then
    echo "[Phase 3] seq=${max_seq_phase3}, lr=3e-6, iters=${p3_iters}, dropout=${dropout} — $(date)"

    cat > "${adapter_dir}/config-phase3.yaml" <<YAML
model: ${model}
train: true
fine_tune_type: lora
batch_size: 2
iters: ${p3_iters}
learning_rate: 3.0e-06
lora_parameters:
  rank: ${rank}
  alpha: ${alpha}
  dropout: ${dropout}
  scale: ${alpha}.0
num_layers: ${num_layers}
max_seq_length: ${max_seq_phase3}
grad_checkpoint: true
grad_accumulation_steps: 8
save_every: 200
steps_per_report: 10
steps_per_eval: 200
val_batches: 5
resume: true
seed: 42
YAML

    mlx_lm.lora \
      -c "${adapter_dir}/config-phase3.yaml" \
      --data "${DATA_DIR}/${domain}" \
      --adapter-path "$adapter_dir" \
      2>&1 | tee -a "$log_file"

    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
      echo "[FAIL] ${prefix}-${domain} Phase 3 — $(date)"
      return 1
    fi
    touch "${adapter_dir}/phase3_done"
    echo "[DONE] Phase 3 — $(date)"
  else
    echo "[SKIP] Phase 3 already done"
  fi

  echo "[COMPLETE] ${prefix}-${domain} curriculum done — $(date)"
  echo ""
}

echo "======================================================="
echo "  EU-KIKI v2 CURRICULUM TRAINING — $(date)"
echo "  Qwen3.6: ${#QWEN_DOMAINS[@]} domains (3 phases)"
echo "  Medium 3.5: ${#MEDIUM_DOMAINS[@]} domains (3 phases)"
echo "======================================================="
echo ""

# ---- Phase 1: Qwen3.6-35B-A3B ----
echo "=== MODEL: Qwen3.6-35B-A3B (rank=32, 16 layers) ==="
echo ""

for domain in "${QWEN_DOMAINS[@]}"; do
  train_curriculum "$QWEN_MODEL" "qwen36" "$domain" 32 32 16 4096
done

echo ""
echo "=== QWEN36 ALL DOMAINS COMPLETE — $(date) ==="
echo ""

# ---- Phase 2: Medium 3.5 128B ----
echo "=== MODEL: Mistral Medium 3.5 128B (rank=16, all layers) ==="
echo ""

for domain in "${MEDIUM_DOMAINS[@]}"; do
  train_curriculum "$MEDIUM_MODEL" "medium35" "$domain" 16 32 -1 2048
done

echo ""
echo "======================================================="
echo "  ALL CURRICULUM TRAINING DONE — $(date)"
echo "======================================================="
