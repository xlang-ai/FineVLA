#!/bin/bash
#
# VQA Video Mode Evaluation Script
#
# Usage:
#   nohup bash run_video_eval.sh > logs/video_eval.log 2>&1 &
#
#   # Custom workers and rounds:
#   nohup bash run_video_eval.sh --workers 8 --rounds 3 > logs/video_eval.log 2>&1 &

set -e
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy="oss-cn-shanghai.aliyuncs.com,oss-cn-beijing.aliyuncs.com"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# ══════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════

NUM_WORKERS=16
NUM_ROUNDS=1
FPS=2
INPUT_TYPE="video"
THINKING="true"

# Parse optional CLI arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --workers) NUM_WORKERS=$2; shift 2;;
        --rounds)  NUM_ROUNDS=$2; shift 2;;
        --fps)     FPS=$2; shift 2;;
        --start)   START_IDX=$2; shift 2;;
        --end)     END_IDX=$2; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

MODEL_LIST=(
    qwen3-vl-plus
    qwen3.5-plus
    vertex_ai.gemini-3.1-pro-preview
)

# ══════════════════════════════════════════════════════════
#  Run
# ══════════════════════════════════════════════════════════

TOTAL_MODELS=${#MODEL_LIST[@]}
TOTAL_RUNS=$((TOTAL_MODELS * NUM_ROUNDS))

mkdir -p logs

echo "============================================================"
echo "  VQA Video Mode Evaluation"
echo "  Models:       ${MODEL_LIST[*]}"
echo "  Input type:   ${INPUT_TYPE}"
echo "  FPS:          ${FPS}"
echo "  Workers:      ${NUM_WORKERS}"
echo "  Rounds:       ${NUM_ROUNDS}"
echo "  Total runs:   ${TOTAL_RUNS}"
echo "  Start:        ${START_IDX:-0}"
echo "  End:          ${END_IDX:-all}"
echo "  Started at:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

CURRENT=0
for ROUND in $(seq 1 ${NUM_ROUNDS}); do
    echo ""
    echo "============================================================"
    echo "  Round ${ROUND}/${NUM_ROUNDS}"
    echo "============================================================"

    for MODEL_NAME in "${MODEL_LIST[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo ""
        echo "------------------------------------------------------------"
        echo "  [${CURRENT}/${TOTAL_RUNS}] Round ${ROUND} - ${MODEL_NAME}"
        echo "  $(date '+%Y-%m-%d %H:%M:%S')"
        echo "------------------------------------------------------------"
        echo ""

        EXTRA_ARGS=""
        if [ -n "${START_IDX}" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --start ${START_IDX}"
        fi
        if [ -n "${END_IDX}" ]; then
            EXTRA_ARGS="${EXTRA_ARGS} --end ${END_IDX}"
        fi

        python3 run_vqa.py \
            --model "${MODEL_NAME}" \
            --input-type "${INPUT_TYPE}" \
            --fps "${FPS}" \
            --thinking "${THINKING}" \
            --num-workers "${NUM_WORKERS}" \
            --round "${ROUND}" \
            ${EXTRA_ARGS}

        echo ""
        echo "  [${CURRENT}/${TOTAL_RUNS}] ${MODEL_NAME} Round ${ROUND} done!"
        echo ""
    done
done

echo ""
echo "============================================================"
echo "  All done! ${TOTAL_MODELS} models x ${NUM_ROUNDS} rounds = ${TOTAL_RUNS} runs"
echo "  Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Results: ${SCRIPT_DIR}/results/"
echo "============================================================"
