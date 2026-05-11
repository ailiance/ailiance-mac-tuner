#!/bin/bash
# Pipeline complet Opus-v3 : SFT → SimPO → GRPO → Merge
# Usage: ./scripts/pipeline_opus_v3.sh [phase1|phase2|phase3|phase4|phase5|all|status]

set -e
cd /Users/clems/ailiance-mac-tuner
source .venv/bin/activate

case "${1:-status}" in
    phase1)
        echo "=== Phase 1 : SFT Curriculum ==="
        ./scripts/train_curriculum.sh all
        ;;
    phase2)
        echo "=== Phase 2 : SimPO Alignment ==="
        python scripts/train_simpo.py --generate-pairs
        python scripts/train_simpo.py
        ;;
    phase3)
        echo "=== Phase 3 : GRPO Reasoning RL ==="
        python scripts/prepare_grpo_data.py
        python scripts/train_grpo.py
        ;;
    phase4)
        echo "=== Phase 4 : (Reserved for RLTT/DAPO) ==="
        echo "Not yet implemented"
        ;;
    phase5)
        echo "=== Phase 5 : Merge + Export ==="
        ./scripts/merge_lora.sh
        ;;
    all)
        $0 phase1
        $0 phase2
        $0 phase3
        $0 phase5
        ;;
    status)
        echo "=== Opus-v3 Pipeline Status ==="
        for phase in phase1-curriculum phase2-simpo phase3-grpo; do
            dir="output/qwen35-122b-opus-v3-${phase#phase?-}"
            if [ -d "$dir" ]; then
                echo "  $phase: $(du -sh $dir | cut -f1)"
            else
                echo "  $phase: not started"
            fi
        done
        ;;
    *)
        echo "Usage: $0 [phase1|phase2|phase3|phase4|phase5|all|status]"
        ;;
esac
