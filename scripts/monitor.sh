#!/bin/bash
# Monitoring autonome des pipelines ailiance-mac-tuner
# Tourne en fond, log dans logs/monitor.log
# Utilisation : ./scripts/monitor.sh [start|stop|status|tail]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."
LOG_DIR="$PROJECT_DIR/logs"
PID_FILE="$LOG_DIR/monitor.pid"
LOG_FILE="$LOG_DIR/monitor.log"

mkdir -p "$LOG_DIR"

monitor_loop() {
    while true; do
        echo "" >> "$LOG_FILE"
        echo "=== $(date '+%Y-%m-%d %Hh%M') ===" >> "$LOG_FILE"

        # Training MLX
        if pgrep -f "mlx_lm.*lora" > /dev/null 2>&1; then
            echo "[TRAINING] En cours" >> "$LOG_FILE"
            grep "^Iter.*Val loss" "$PROJECT_DIR/training.log" 2>/dev/null | tail -3 >> "$LOG_FILE"
            tail -1 "$PROJECT_DIR/training.log" 2>/dev/null >> "$LOG_FILE"
        else
            echo "[TRAINING] Inactif" >> "$LOG_FILE"
        fi

        # Distillation
        if pgrep -f "distill_generate" > /dev/null 2>&1; then
            DISTILL_COUNT=$(wc -l < "$PROJECT_DIR/data/distilled-mistral-large-123b/all.jsonl" 2>/dev/null || echo 0)
            echo "[DISTILLATION] En cours — $DISTILL_COUNT exemples generes" >> "$LOG_FILE"
        else
            if [ -f "$PROJECT_DIR/data/distilled-mistral-large-123b/all.jsonl" ]; then
                DISTILL_COUNT=$(wc -l < "$PROJECT_DIR/data/distilled-mistral-large-123b/all.jsonl" 2>/dev/null || echo 0)
                echo "[DISTILLATION] Terminee — $DISTILL_COUNT exemples" >> "$LOG_FILE"
            else
                echo "[DISTILLATION] Inactive" >> "$LOG_FILE"
            fi
        fi

        # Generation CPU
        if pgrep -f "llama-completion" > /dev/null 2>&1; then
            CPU_COUNT=$(wc -l < "$PROJECT_DIR/data/generated-cpu-qwen35-35b/all.jsonl" 2>/dev/null || echo 0)
            echo "[GEN CPU] En cours — $CPU_COUNT exemples" >> "$LOG_FILE"
        else
            echo "[GEN CPU] Inactive" >> "$LOG_FILE"
        fi

        # Conversion MLX
        if pgrep -f "mlx_lm.*convert" > /dev/null 2>&1; then
            echo "[CONVERSION] En cours" >> "$LOG_FILE"
        else
            echo "[CONVERSION] Inactive" >> "$LOG_FILE"
        fi

        # Espace disque
        DISK_USED=$(du -sh "$PROJECT_DIR/models" 2>/dev/null | cut -f1)
        DISK_OUTPUT=$(du -sh "$PROJECT_DIR/output" 2>/dev/null | cut -f1)
        echo "[DISQUE] models: $DISK_USED, output: $DISK_OUTPUT" >> "$LOG_FILE"

        sleep 300  # Check toutes les 5 minutes
    done
}

case "${1:-status}" in
    start)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "Monitor deja actif (PID $(cat "$PID_FILE"))"
            exit 0
        fi
        monitor_loop &
        echo $! > "$PID_FILE"
        echo "Monitor demarre (PID $!, log: $LOG_FILE)"
        ;;
    stop)
        if [ -f "$PID_FILE" ]; then
            kill "$(cat "$PID_FILE")" 2>/dev/null && echo "Monitor arrete" || echo "Monitor deja arrete"
            rm -f "$PID_FILE"
        else
            echo "Pas de monitor actif"
        fi
        ;;
    status)
        echo "=== $(date '+%Y-%m-%d %Hh%M') ==="
        echo ""

        # Training
        if pgrep -f "mlx_lm.*lora" > /dev/null 2>&1; then
            echo "[TRAINING] En cours"
            grep "^Iter.*Val loss" "$PROJECT_DIR/training.log" 2>/dev/null | tail -3
            tail -1 "$PROJECT_DIR/training.log" 2>/dev/null
        else
            echo "[TRAINING] Inactif"
            LAST_VAL=$(grep "^Iter.*Val loss" "$PROJECT_DIR/training.log" 2>/dev/null | tail -1)
            [ -n "$LAST_VAL" ] && echo "Dernier: $LAST_VAL"
        fi
        echo ""

        # Distillation
        if pgrep -f "distill_generate" > /dev/null 2>&1; then
            DISTILL_COUNT=$(wc -l < "$PROJECT_DIR/data/distilled-mistral-large-123b/all.jsonl" 2>/dev/null || echo 0)
            echo "[DISTILLATION] En cours — $DISTILL_COUNT exemples"
        elif [ -f "$PROJECT_DIR/data/distilled-mistral-large-123b/all.jsonl" ]; then
            DISTILL_COUNT=$(wc -l < "$PROJECT_DIR/data/distilled-mistral-large-123b/all.jsonl" 2>/dev/null || echo 0)
            echo "[DISTILLATION] Terminee — $DISTILL_COUNT exemples"
        else
            echo "[DISTILLATION] Inactive"
        fi
        echo ""

        # Conversion
        if pgrep -f "mlx_lm.*convert" > /dev/null 2>&1; then
            echo "[CONVERSION] En cours (397B 4-bit)"
        else
            echo "[CONVERSION] Inactive"
        fi
        echo ""

        # Disque
        echo "[DISQUE]"
        du -sh "$PROJECT_DIR/models" "$PROJECT_DIR/output" "$PROJECT_DIR/data" 2>/dev/null
        ;;
    tail)
        tail -f "$LOG_FILE"
        ;;
    *)
        echo "Utilisation : $0 [start|stop|status|tail]"
        echo ""
        echo "  start   Demarre le monitoring en fond (log toutes les 5 min)"
        echo "  stop    Arrete le monitoring"
        echo "  status  Affiche l'etat actuel (one-shot)"
        echo "  tail    Suit le log en temps reel"
        ;;
esac
