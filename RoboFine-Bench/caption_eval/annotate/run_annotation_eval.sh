#!/bin/bash
#
# 批量标注脚本: 5 个模型 × 500 条数据
#
# 用法:
#   bash CaptionEval/Annotation/run_annotation_eval.sh              # easy 模式 (含 instruction_raw)
#   bash CaptionEval/Annotation/run_annotation_eval.sh hard         # hard 模式 (不含 instruction_raw)
#   bash CaptionEval/Annotation/run_annotation_eval.sh hard 32      # hard 模式 + 32 线程
#   bash CaptionEval/Annotation/run_annotation_eval.sh easy 16 10   # easy 模式 + 只测前 10 条
#
# nohup 运行:
#   nohup bash CaptionEval/Annotation/run_annotation_eval.sh hard 16 > run_annotation.log 2>&1 &

set -e
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy="oss-cn-shanghai.aliyuncs.com,oss-cn-beijing.aliyuncs.com"

BASE_DIR="/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLM4Robotics_Benchmark"
SCRIPT="${BASE_DIR}/CaptionEval/Annotation/run_annotate.py"

MODE="${1:-easy}"
NUM_WORKERS="${2:-16}"
END_SAMPLES="${3:-}"

MODEL_LIST=(
    qwen3-vl-plus
    qwen3.5-plus
    doubao.doubao-seed-2-0-pro-260215
    openai.gpt-5.4-2026-03-05
    vertex_ai.gemini-3.1-pro-preview
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
echo "  Output:     ${OUTPUT_DIR}"
echo "  Models:     ${MODEL_LIST[*]}"
echo "  Workers:    ${NUM_WORKERS}"
echo "  Samples:    ${END_SAMPLES:-all}"
echo "============================================================"
echo ""

CURRENT=0
for MODEL_NAME in "${MODEL_LIST[@]}"; do
    CURRENT=$((CURRENT + 1))

    MODEL_TAG=$(echo "${MODEL_NAME}" | sed 's/[\/.]/_/g')
    RESULT_FILE="${OUTPUT_DIR}/${MODEL_TAG}_CaptionResult.jsonl"

    echo ""
    echo "------------------------------------------------------------"
    echo "  [${CURRENT}/${TOTAL}] Model: ${MODEL_NAME}"

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
