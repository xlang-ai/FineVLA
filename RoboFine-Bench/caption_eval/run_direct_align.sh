#!/bin/bash
# Run Direct Alignment (Method B) for selected models
# Usage: bash caption_eval/run_direct_align.sh [easy|hard|all] [num_workers]
#   Run from RoboFine-Bench/ directory, or the script auto-detects its location.

set +e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/.."  # RoboFine-Bench/

GT_FACTS="EvalData/GT_AtomicFacts.jsonl"
OUTPUT_BASE="caption_eval/result/DirectAlign"
MODE="${1:-easy}"
NUM_WORKERS="${2:-8}"

MODELS=(
    "RoboFine-VLM_CaptionResult.jsonl"
    "RoboFine-VLM_T0.7_CaptionResult.jsonl"
    "openai_gpt-5_4-2026-03-05_CaptionResult.jsonl"
    "vertex_ai_gemini-3_1-pro-preview_CaptionResult.jsonl"
    "qwen3-vl-plus_CaptionResult.jsonl"
    "qwen3_5-plus_CaptionResult.jsonl"
    "doubao_doubao-seed-2-0-pro-260215_CaptionResult.jsonl"
)

run_mode() {
    local mode=$1
    local CAPTION_DIR="caption_eval/result/caption/${mode}"

    echo ""
    echo "=========================================="
    echo "Direct Alignment (Method B) - ${mode} mode"
    echo "Models: ${#MODELS[@]}"
    echo "Workers: ${NUM_WORKERS}"
    echo "Start: $(date)"
    echo "=========================================="

    for CAPTION_FILE in "${MODELS[@]}"; do
        MODEL_NAME="${CAPTION_FILE%%_CaptionResult*}"
        OUTPUT_DIR="${OUTPUT_BASE}/${mode}/${MODEL_NAME}"
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

        FILTER_DIR="${CAPTION_DIR}/.filtered"
        FILTERED_CAPTION_PATH="${FILTER_DIR}/${CAPTION_FILE%.jsonl}.success.jsonl"
        FAILED_CAPTION_PATH="${FILTER_DIR}/${CAPTION_FILE%.jsonl}.failed.jsonl"
        python3 caption_eval/filter_caption_results.py \
            --input "${CAPTION_PATH}" \
            --success-output "${FILTERED_CAPTION_PATH}" \
            --failed-output "${FAILED_CAPTION_PATH}"

        FAILED_COUNT="$(wc -l < "${FAILED_CAPTION_PATH}" | tr -d ' ')"
        if [ "${FAILED_COUNT}" != "0" ]; then
            echo "[WARN] ${MODEL_NAME} (${mode}) has ${FAILED_COUNT} failed/empty caption records."
            echo "[WARN] Failed records written to ${FAILED_CAPTION_PATH}"
        fi

        # Skip models that already have complete outputs; rerun only missing/failed ones.
        if [ -f "${OUTPUT_DIR}/scored_results.jsonl" ] && [ -s "${OUTPUT_DIR}/scored_results.jsonl" ] \
           && [ -f "${OUTPUT_DIR}/direct_align_raw.jsonl" ] && [ -s "${OUTPUT_DIR}/direct_align_raw.jsonl" ] \
           && [ -f "${OUTPUT_DIR}/dataset_summary.json" ] && [ -s "${OUTPUT_DIR}/dataset_summary.json" ] \
           && [ -f "${OUTPUT_DIR}/dataset_summary.csv" ] && [ -s "${OUTPUT_DIR}/dataset_summary.csv" ]; then
            echo "[SKIP] ${MODEL_NAME} (${mode}) already completed"
            continue
        fi

        python3 -m caption_eval.atomic_eval.atomic_eval direct-align \
            --gt-facts "${GT_FACTS}" \
            --caption "${FILTERED_CAPTION_PATH}" \
            --output-dir "${OUTPUT_DIR}" \
            --num-workers "${NUM_WORKERS}" \
            --enable-thinking

        echo "[DONE] ${MODEL_NAME} (${mode}) at $(date)"
    done

    # Generate cross-model summary for this mode
    RESULT_DIRS=""
    for CAPTION_FILE in "${MODELS[@]}"; do
        MODEL_NAME="${CAPTION_FILE%%_CaptionResult*}"
        DIR="${OUTPUT_BASE}/${mode}/${MODEL_NAME}"
        if [ -f "${DIR}/dataset_summary.json" ]; then
            RESULT_DIRS="${RESULT_DIRS} ${DIR}"
        fi
    done

    if [ -n "${RESULT_DIRS}" ]; then
        echo ""
        echo "Generating cross-model summary (${mode})..."
        python3 -m caption_eval.atomic_eval.atomic_eval summary \
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
