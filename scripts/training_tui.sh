#!/bin/bash
# TUI de suivi training complet (bash pur)
# Usage : ./scripts/training_tui.sh [logfile]

LOG="${1:-logs/mlxtune-122b-resume2.log}"

while true; do
    clear

    NOW=$(date '+%H:%M:%S')
    DATE=$(date '+%Y-%m-%d')

    # === Parse log ===
    ALL_TRAIN=$(grep "^Iter.*Train loss" "$LOG" 2>/dev/null)
    ALL_VAL=$(grep "^Iter.*Val loss" "$LOG" 2>/dev/null)
    ALL_SAVE=$(grep "Saved adapter" "$LOG" 2>/dev/null)
    LAST_TRAIN=$(echo "$ALL_TRAIN" | tail -1)
    LAST_VAL=$(echo "$ALL_VAL" | tail -1)
    FIRST_TRAIN=$(echo "$ALL_TRAIN" | head -1)

    # Iter courante
    ITER=$(echo "$LAST_TRAIN" | grep -oE 'Iter [0-9]+' | awk '{print $2}')
    [ -z "$ITER" ] && ITER=0
    TOTAL=$(grep "iters:" "$LOG" 2>/dev/null | tail -1 | grep -oE '[0-9]+')
    [ -z "$TOTAL" ] && TOTAL=3000

    # Train loss
    LOSS=$(echo "$LAST_TRAIN" | grep -oE 'Train loss [0-9.]+' | awk '{print $3}')
    LR=$(echo "$LAST_TRAIN" | grep -oE 'Learning Rate [0-9.e+-]+' | awk '{print $3}')
    ITS=$(echo "$LAST_TRAIN" | grep -oE 'It/sec [0-9.]+' | awk '{print $2}')
    TOKS=$(echo "$LAST_TRAIN" | grep -oE 'Tokens/sec [0-9.]+' | awk '{print $2}')
    MEM=$(echo "$LAST_TRAIN" | grep -oE 'Peak mem [0-9.]+' | awk '{print $3}')
    TRAINED_TOK=$(echo "$LAST_TRAIN" | grep -oE 'Trained Tokens [0-9]+' | awk '{print $3}')

    # Val loss
    VLOSS=$(echo "$LAST_VAL" | grep -oE 'Val loss [0-9.]+' | awk '{print $3}')
    VITER=$(echo "$LAST_VAL" | grep -oE 'Iter [0-9]+' | awk '{print $2}')
    BEST_VAL=$(echo "$ALL_VAL" | grep -oE 'Val loss [0-9.]+' | awk '{print $3}' | sort -n | head -1)
    BEST_VITER=$(echo "$ALL_VAL" | grep "Val loss $BEST_VAL" | grep -oE 'Iter [0-9]+' | awk '{print $2}' | head -1)

    # Val loss history (toutes)
    VAL_HIST=$(echo "$ALL_VAL" | awk '{
        match($0, /Iter ([0-9]+)/, i);
        match($0, /Val loss ([0-9.]+)/, v);
        printf "%s:%.3f ", i[1], v[1]
    }')

    # Train loss min/max/avg des 50 dernieres
    TRAIN_STATS=$(echo "$ALL_TRAIN" | tail -50 | grep -oE 'Train loss [0-9.]+' | awk '{print $3}' | awk '{
        sum+=$1; n++;
        if(n==1 || $1<min) min=$1;
        if(n==1 || $1>max) max=$1
    } END {if(n>0) printf "%.3f %.3f %.3f", min, sum/n, max}')
    TLOSS_MIN=$(echo "$TRAIN_STATS" | awk '{print $1}')
    TLOSS_AVG=$(echo "$TRAIN_STATS" | awk '{print $2}')
    TLOSS_MAX=$(echo "$TRAIN_STATS" | awk '{print $3}')

    # Saves
    SAVES=$(echo "$ALL_SAVE" | wc -l | tr -d ' ')
    LAST_SAVE_ITER=$(echo "$ALL_SAVE" | tail -1 | grep -oE 'Iter [0-9]+' | awk '{print $2}')
    SAVE_LIST=$(echo "$ALL_SAVE" | grep -oE '[0-9]+_adapters' | sed 's/_adapters//' | tr '\n' ' ')

    # Erreurs
    ERRORS=$(grep -ciE "error|RuntimeError|SIGABRT|Traceback|OOM|timeout" "$LOG" 2>/dev/null)
    LAST_ERROR=$(grep -iE "error|RuntimeError|SIGABRT|OOM|timeout" "$LOG" 2>/dev/null | tail -1 | head -c 60)

    # Warnings
    WARNINGS=$(grep -c "WARNING" "$LOG" 2>/dev/null)
    LONG_SEQ=$(grep "longest sentence" "$LOG" 2>/dev/null | tail -1 | grep -oE '[0-9]+ will' | awk '{print $1}')

    # Process
    PID=$(ps aux | grep "mlx_lm" | grep -v grep | awk '{print $2}' | head -1)
    CPU=$(ps aux | grep "mlx_lm" | grep -v grep | awk '{print $3}' | head -1)
    PMEM=$(ps aux | grep "mlx_lm" | grep -v grep | awk '{print $4}' | head -1)
    if [ -z "$PID" ]; then
        STATUS="⛔ STOPPE"
        UPTIME="—"
    else
        STATUS="✅ ACTIF"
        # Uptime
        START_EPOCH=$(ps -p "$PID" -o lstart= 2>/dev/null)
        if [ -n "$START_EPOCH" ]; then
            START_TS=$(date -j -f "%a %b %d %T %Y" "$START_EPOCH" "+%s" 2>/dev/null)
            NOW_TS=$(date "+%s")
            if [ -n "$START_TS" ]; then
                ELAPSED=$((NOW_TS - START_TS))
                UH=$((ELAPSED / 3600))
                UM=$(((ELAPSED % 3600) / 60))
                UPTIME="${UH}h${UM}m"
            else
                UPTIME="—"
            fi
        else
            UPTIME="—"
        fi
    fi

    # Progression
    PCT=$((ITER * 100 / TOTAL))
    BAR_LEN=40
    FILLED=$((PCT * BAR_LEN / 100))
    EMPTY=$((BAR_LEN - FILLED))
    BAR=$(printf '%0.s█' $(seq 1 $FILLED 2>/dev/null))$(printf '%0.s░' $(seq 1 $EMPTY 2>/dev/null))

    # ETA
    if [ -n "$ITS" ] && [ "$ITS" != "0" ] && [ "$ITS" != "0.000" ]; then
        REMAINING=$(echo "($TOTAL - $ITER) / $ITS" | bc 2>/dev/null)
        if [ -n "$REMAINING" ] && [ "$REMAINING" -gt 0 ] 2>/dev/null; then
            ETA_H=$((REMAINING / 3600))
            ETA_M=$(((REMAINING % 3600) / 60))
            ETA_STR="${ETA_H}h${ETA_M}m"
            FINISH=$(date -v+${REMAINING}S '+%H:%M' 2>/dev/null || echo "—")
        else
            ETA_STR="—"
            FINISH="—"
        fi
    else
        ETA_STR="—"
        FINISH="—"
    fi

    # Mem bar
    [ -z "$MEM" ] && MEM="0"
    MEM_INT=${MEM%%.*}
    [ -z "$MEM_INT" ] && MEM_INT=0
    MEM_PCT=$((MEM_INT * 100 / 512))
    MEM_FILLED=$((MEM_PCT * 30 / 100))
    MEM_EMPTY=$((30 - MEM_FILLED))
    MEM_BAR=$(printf '%0.s▓' $(seq 1 $MEM_FILLED 2>/dev/null))$(printf '%0.s░' $(seq 1 $MEM_EMPTY 2>/dev/null))
    MEM_FREE=$((512 - MEM_INT))

    # Disk
    DISK_FREE=$(df -h / | tail -1 | awk '{print $4}')
    ADAPTER_SIZE=$(du -sh output/qwen35-122b-opus-v3/adapters/ 2>/dev/null | cut -f1)

    # === AFFICHAGE ===
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  🔥 ailiance-mac-tuner — Training Monitor         $DATE $NOW ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    echo "║  Modele   : Qwen3.5-122B-A10B-Opus-v3 (BF16, MoE 10B actifs)  ║"
    printf "║  Process  : %-53s ║\n" "$STATUS (PID ${PID:-—}, ${CPU:-0}%CPU, ${PMEM:-0}%MEM)"
    printf "║  Uptime   : %-53s ║\n" "$UPTIME"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    echo "║                        PROGRESSION                             ║"
    printf "║  %s %3d%%     ║\n" "$BAR" "$PCT"
    printf "║  Iter      : %6s / %-6s                                   ║\n" "$ITER" "$TOTAL"
    printf "║  ETA       : %-10s (fin ~%s)                              ║\n" "$ETA_STR" "$FINISH"
    printf "║  Tokens    : %-10s entraines                               ║\n" "${TRAINED_TOK:-—}"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    echo "║                          LOSS                                  ║"
    printf "║  Train     : %-10s (min:%-6s avg:%-6s max:%-6s)       ║\n" "${LOSS:-—}" "${TLOSS_MIN:-—}" "${TLOSS_AVG:-—}" "${TLOSS_MAX:-—}"
    printf "║  Val       : %-10s (iter %-5s)                            ║\n" "${VLOSS:-—}" "${VITER:-—}"
    printf "║  Best Val  : %-10s (iter %-5s) ★                         ║\n" "${BEST_VAL:-—}" "${BEST_VITER:-—}"
    echo "║                                                                ║"
    printf "║  Val hist  : %-52s ║\n" "$VAL_HIST"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    echo "║                       PERFORMANCE                              ║"
    printf "║  Vitesse   : %6s tok/s | %s it/s                          ║\n" "${TOKS:-—}" "${ITS:-—}"
    printf "║  LR        : %-53s ║\n" "${LR:-—}"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    echo "║                        MEMOIRE                                 ║"
    printf "║  Peak GPU  : %s Go / 512 Go  (libre: %s Go)                ║\n" "${MEM:-—}" "$MEM_FREE"
    printf "║  Metal     : %s %2d%%           ║\n" "$MEM_BAR" "$MEM_PCT"
    printf "║  Disque    : %s libre                                        ║\n" "$DISK_FREE"
    printf "║  Adapters  : %s (%s checkpoints)                           ║\n" "${ADAPTER_SIZE:-—}" "$SAVES"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    echo "║                      CHECKPOINTS                               ║"
    printf "║  Sauves    : %-53s ║\n" "${SAVE_LIST:-aucun}"
    printf "║  Dernier   : iter %-48s ║\n" "${LAST_SAVE_ITER:-—}"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    echo "║                        SANTE                                   ║"
    printf "║  Erreurs   : %-5s  Warnings : %-5s                          ║\n" "$ERRORS" "$WARNINGS"
    if [ "$ERRORS" -gt 0 ] 2>/dev/null; then
        printf "║  Derniere  : %-53s ║\n" "${LAST_ERROR:-—}"
    fi
    if [ -n "$LONG_SEQ" ]; then
        printf "║  Seq max   : %s tokens (tronque a 1280)                     ║\n" "$LONG_SEQ"
    fi
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  Log  : $LOG"
    echo "  Ctrl+C pour quitter | Refresh 5s"

    sleep 5
done
