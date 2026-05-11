#!/usr/bin/env bash
set -uo pipefail
ROOT="/Users/clems/ailiance-mac-tuner"
LOGDIR="${ROOT}/logs/122b-macport"
CSV="${LOGDIR}/memcsv-$(date +%Y%m%d-%H%M%S).csv"
SWAP_THRESHOLD_GB=80
SUSTAINED_SAMPLES=10
mkdir -p "${LOGDIR}"
echo "ts,rss_gb,swap_used_gb,mem_pressure_free_pct,train_pid" > "${CSV}"
over_count=0
parse_swap_gb() {
  sysctl -n vm.swapusage | awk '{
    for (i=1; i<=NF; i++) if ($i == "used") {
      val=$(i+2); unit=substr(val, length(val), 1); num=substr(val, 1, length(val)-1);
      if (unit=="M") print num/1024;
      else if (unit=="G") print num;
      else print 0;
      exit;
    }
  }'
}
while true; do
  TRAIN_PID=$(pgrep -f 'mlx_lm lora' | head -1 || true)
  if [[ -z "${TRAIN_PID}" ]]; then sleep 30; continue; fi
  RSS_KB=$(ps -o rss= -p "${TRAIN_PID}" 2>/dev/null | tr -d ' ' || echo 0)
  RSS_GB=$(awk -v k="${RSS_KB:-0}" 'BEGIN{printf "%.1f", k/1024/1024}')
  SWAP_GB=$(parse_swap_gb)
  MP_FREE=$(memory_pressure 2>/dev/null | awk '/System-wide memory free percentage/{gsub("%","",$NF); print $NF}' || echo "")
  TS=$(date +%Y-%m-%dT%H:%M:%S)
  echo "${TS},${RSS_GB},${SWAP_GB},${MP_FREE},${TRAIN_PID}" >> "${CSV}"
  if awk -v s="${SWAP_GB:-0}" -v t="${SWAP_THRESHOLD_GB}" 'BEGIN{exit !(s>t)}'; then
    over_count=$((over_count+1))
    echo "WARN swap=${SWAP_GB}GB > ${SWAP_THRESHOLD_GB}GB (${over_count}/${SUSTAINED_SAMPLES})"
    if [[ "${over_count}" -ge "${SUSTAINED_SAMPLES}" ]]; then
      echo "CRIT: sustained swap thrash -- killing PID ${TRAIN_PID}"
      kill -TERM "${TRAIN_PID}" 2>/dev/null || true
      exit 2
    fi
  else over_count=0; fi
  sleep 30
done
