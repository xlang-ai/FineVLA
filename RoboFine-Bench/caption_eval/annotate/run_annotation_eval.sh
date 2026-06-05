#!/bin/bash
#
# Batch annotation script: 5 models x 500 samples
#
# Usage:
#   bash caption_eval/annotate/run_annotation_eval.sh              # easy mode (with instruction_raw)
#   bash caption_eval/annotate/run_annotation_eval.sh hard         # hard mode (without instruction_raw)
#   bash caption_eval/annotate/run_annotation_eval.sh hard 32      # hard mode + 32 workers
#   bash caption_eval/annotate/run_annotation_eval.sh easy 16 10   # easy mode + only test first 10 samples
#
# Run in background:
#   nohup bash caption_eval/annotate/run_annotation_eval.sh hard 16 > run_annotation.log 2>&1 &

set -e
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy="oss-cn-shanghai.aliyuncs.com,oss-cn-beijing.aliyuncs.com"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SCRIPT="${SCRIPT_DIR}/run_annotate.py"

MODE="${1:-easy}"
NUM_WORKERS="${2:-16}"
END_SAMPLES="${3:-}"

FPS=4

# model_name:input_type — GPT uses image mode (no video URL support)
MODEL_LIST=(
    "qwen3-vl-plus:video"
    "qwen3.5-plus:video"
    "doubao.doubao-seed-2-0-pro-260215:video"
    "openai.gpt-5.4-2026-03-05:image"
    "vertex_ai.gemini-3.1-pro-preview:video"
)

if [ "${MODE}" == "hard" ]; then
    OUTPUT_DIR="${BASE_DIR}/CaptionEval/CaptionResult/hard"
    NO_INSTR_FLAG="--no-instruction"
else
    OUTPUT_DIR="${BASE_DIR}/CaptionEval/CaptionResult/easy"
    NO_INSTR_FLAG=""
fi

mkdir -p "${OUTPUT_DIR}"

TOTAL=${#MODEL_LIST[@]}
echo "============================================================"
echo "  Video Annotation (Single-Stage)"
echo "  Mode:       ${MODE}"
echo "  FPS:        ${FPS}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  Workers:    ${NUM_WORKERS}"
echo "  Samples:    ${END_SAMPLES:-all}"
echo "============================================================"
echo ""

CURRENT=0
for ENTRY in "${MODEL_LIST[@]}"; do
    CURRENT=$((CURRENT + 1))
    MODEL_NAME="${ENTRY%%:*}"
    INPUT_TYPE="${ENTRY##*:}"

    MODEL_TAG=$(echo "${MODEL_NAME}" | sed 's/[\/.]/_/g')
    RESULT_FILE="${OUTPUT_DIR}/${MODEL_TAG}_CaptionResult.jsonl"

    echo ""
    echo "------------------------------------------------------------"
    echo "  [${CURRENT}/${TOTAL}] Model: ${MODEL_NAME} (${INPUT_TYPE}, fps=${FPS})"

    if [ -f "${RESULT_FILE}" ]; then
        TOTAL_LINES=$(wc -l < "${RESULT_FILE}" | tr -d ' ')
        SUCCESS_CNT=$(grep -c '"call_success": true' "${RESULT_FILE}" 2>/dev/null || echo 0)
        FAIL_CNT=$(grep -c '"call_success": false' "${RESULT_FILE}" 2>/dev/null || echo 0)
        echo "  Resume mode: ${RESULT_FILE} exists"
        echo "    success=${SUCCESS_CNT}, failed=${FAIL_CNT}, total=${TOTAL_LINES}"
        if [ "${FAIL_CNT}" -eq 0 ] && [ "${SUCCESS_CNT}" -gt 0 ]; then
            echo "  All samples succeeded, skipping."
            echo "------------------------------------------------------------"
            continue
        fi
        echo "  Retrying ${FAIL_CNT} failed samples..."
    else
        echo "  Fresh run (no existing result file)"
    fi
    echo "------------------------------------------------------------"
    echo ""

    EXTRA_ARGS=""
    if [ -n "${END_SAMPLES}" ]; then
        EXTRA_ARGS="--start 0 --end ${END_SAMPLES}"
    fi

    python "${SCRIPT}" \
        --model "${MODEL_NAME}" \
        --input-type "${INPUT_TYPE}" \
        --fps "${FPS}" \
        --output-dir "${OUTPUT_DIR}" \
        --num-workers "${NUM_WORKERS}" \
        ${NO_INSTR_FLAG} \
        ${EXTRA_ARGS}

    # Post-run stats
    if [ -f "${RESULT_FILE}" ]; then
        FINAL_SUCCESS=$(grep -c '"call_success": true' "${RESULT_FILE}" 2>/dev/null || echo 0)
        FINAL_FAIL=$(grep -c '"call_success": false' "${RESULT_FILE}" 2>/dev/null || echo 0)
        echo ""
        echo "  [${CURRENT}/${TOTAL}] ${MODEL_NAME} done! (success=${FINAL_SUCCESS}, failed=${FINAL_FAIL})"
    else
        echo ""
        echo "  [${CURRENT}/${TOTAL}] ${MODEL_NAME} done!"
    fi
    echo ""
done

echo ""
echo "============================================================"
echo "  All ${TOTAL} models completed!"
echo "  Results: ${OUTPUT_DIR}/"
echo "============================================================"
