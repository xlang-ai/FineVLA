#!/bin/bash
# Run Direct Alignment (Method B) for selected models
# Usage: nohup bash CaptionEval/run_direct_align.sh [easy|hard|all] [num_workers] > run_direct_align.log 2>&1 &

set -e

PYTHON="/root/miniconda3/envs/any4lerobot/bin/python"
GT_FACTS="CaptionEval/AtomicResult/GT_AtomicFacts.jsonl"
OUTPUT_BASE="CaptionEval/AtomicResult/DirectAlign"
MODE="${1:-easy}"
NUM_WORKERS="${2:-8}"

MODELS=(
    "Qwen36-SFT_CaptionResult.jsonl"
    "Qwen36-SFT_T0.7_CaptionResult.jsonl"
    "openai_gpt-5_4-2026-03-05_CaptionResult.jsonl"
    "vertex_ai_gemini-3_1-pro-preview_CaptionResult.jsonl"
    "gemini_3_1_pro_CaptionResult.jsonl"
    "qwen3-vl-plus_CaptionResult.jsonl"
    "qwen3_5-plus_CaptionResult.jsonl"
    "doubao_doubao-seed-2-0-pro-260215_CaptionResult.jsonl"
)

cd /mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLM4Robotics_Benchmark

run_mode() {
    local mode=$1
    local CAPTION_DIR="CaptionEval/CaptionResult/${mode}"

    echo ""
    echo "=========================================="
    echo "Direct Alignment (Method B) - ${mode} mode"
    echo "Models: ${#MODELS[@]}"
    echo "Workers: ${NUM_WORKERS}"
    echo "Start: $(date)"
    echo "=========================================="

    for CAPTION_FILE in "${MODELS[@]}"; do
        MODEL_NAME="${CAPTION_FILE%%_CaptionResult*}"
        OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_NAME}_${mode}"
        CAPTION_PATH="${CAPTION_DIR}/${CAPTION_FILE}"

        if [ ! -f "${CAPTION_PATH}" ]; then
            echo "[SKIP] ${CAPTION_FILE} not found in ${mode}/"
            continue
        fi

        echo ""
        echo "=========================================="
        echo "Model: ${MODEL_NAME} (${mode})"
        echo "Caption: ${CAPTION_PATH}"
        echo "Output: ${OUTPUT_DIR}"
        echo "Time: $(date)"
        echo "=========================================="

        # Skip models that already have complete outputs; rerun only missing/failed ones.
        if [ -f "${OUTPUT_DIR}/scored_results.jsonl" ] && [ -s "${OUTPUT_DIR}/scored_results.jsonl" ] \
           && [ -f "${OUTPUT_DIR}/direct_align_raw.jsonl" ] && [ -s "${OUTPUT_DIR}/direct_align_raw.jsonl" ] \
           && [ -f "${OUTPUT_DIR}/dataset_summary.json" ] && [ -s "${OUTPUT_DIR}/dataset_summary.json" ] \
           && [ -f "${OUTPUT_DIR}/dataset_summary.csv" ] && [ -s "${OUTPUT_DIR}/dataset_summary.csv" ]; then
            echo "[SKIP] ${MODEL_NAME} (${mode}) already completed"
            continue
        fi

        $PYTHON -m CaptionEval.AtomicEval.atomic_eval direct-align \
            --gt-facts "${GT_FACTS}" \
            --caption "${CAPTION_PATH}" \
            --output-dir "${OUTPUT_DIR}" \
            --num-workers "${NUM_WORKERS}" \
            --enable-thinking

        echo "[DONE] ${MODEL_NAME} (${mode}) at $(date)"
    done

    # Generate cross-model summary for this mode
    RESULT_DIRS=""
    for CAPTION_FILE in "${MODELS[@]}"; do
        MODEL_NAME="${CAPTION_FILE%%_CaptionResult*}"
        DIR="${OUTPUT_BASE}/${MODEL_NAME}_${mode}"
        if [ -f "${DIR}/dataset_summary.json" ]; then
            RESULT_DIRS="${RESULT_DIRS} ${DIR}"
        fi
    done

    if [ -n "${RESULT_DIRS}" ]; then
        echo ""
        echo "Generating cross-model summary (${mode})..."
        $PYTHON -m CaptionEval.AtomicEval.atomic_eval summary \
            --results-dirs ${RESULT_DIRS} \
            --output "${OUTPUT_BASE}/cross_model_summary_${mode}.csv"
    fi
}

if [ "${MODE}" = "all" ]; then
    run_mode "easy"
    run_mode "hard"
elif [ "${MODE}" = "easy" ] || [ "${MODE}" = "hard" ]; then
    run_mode "${MODE}"
else
    echo "Usage: $0 [easy|hard|all] [num_workers]"
    exit 1
fi

echo ""
echo "=========================================="
echo "All tasks completed at $(date)"
echo "=========================================="
