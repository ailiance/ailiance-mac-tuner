#!/usr/bin/env bash
set -euo pipefail
ROOT="/Users/clems/ailiance-mac-tuner"
CONFIG="${ROOT}/configs/qwen35-122b-macport.yaml"
LOGDIR="${ROOT}/logs/122b-macport"
TRAIN_LOG="${LOGDIR}/train-$(date +%Y%m%d-%H%M%S).log"
WATCHDOG_LOG="${LOGDIR}/watchdog-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "${LOGDIR}"
cd "${ROOT}"
WIRED=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)
if [[ "${WIRED}" -lt 400000 ]]; then
  echo "ERROR: iogpu.wired_limit_mb=${WIRED} (< 400000). Apply via: sudo sysctl -w iogpu.wired_limit_mb=458752" >&2
  exit 1
fi
echo "wired_limit_mb=${WIRED} (OK)"
"${ROOT}/scripts/watchdog_mem.sh" > "${WATCHDOG_LOG}" 2>&1 &
WATCHDOG_PID=$!
echo "watchdog PID=${WATCHDOG_PID} -> ${WATCHDOG_LOG}"
cleanup() { kill "${WATCHDOG_PID}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
echo "launching mlx_lm.lora -> ${TRAIN_LOG}"
exec .venv/bin/python -m mlx_lm lora --config "${CONFIG}" 2>&1 | tee "${TRAIN_LOG}"
