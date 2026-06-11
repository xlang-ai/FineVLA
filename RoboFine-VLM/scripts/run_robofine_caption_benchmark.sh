#!/usr/bin/env bash
set -euo pipefail

BENCH_DIR="${BENCH_DIR:-/cpfs01/data/shared/Group-m6/tongzai.hxt/FineVLA/RoboFine-Bench}"
ROBOFINE_VLM_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${MODEL:-robofine-vlm}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
MODE="${MODE:-hard}"
INPUT_TYPE="${INPUT_TYPE:-image}"
WORKERS="${WORKERS:-8}"
OUT_DIR="${OUT_DIR:-${BENCH_DIR}/caption_eval/result/caption/robofine_${MODE}_${INPUT_TYPE}}"
CONFIG_FILE="${CONFIG_FILE:-${ROBOFINE_VLM_DIR}/config/caption_annotation_config.json}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

echo "============================================================"
echo "  RoboFine-VLM Caption Annotation"
echo "============================================================"
echo "  Model:      ${MODEL}"
echo "  Base URL:   ${BASE_URL}"
echo "  Mode:       ${MODE}"
echo "  Input type: ${INPUT_TYPE}"
echo "  Workers:    ${WORKERS}"
echo "  Config:     ${CONFIG_FILE}"
echo "  Output dir: ${OUT_DIR}"
echo ""
echo "  Fixed setting:"
echo "    fps=4.0, prompt=RB_new, temperature=0.0, top_p=0.95,"
echo "    max_tokens=32768, no pixel overrides"
echo "============================================================"

cd "${BENCH_DIR}"

python caption_eval/run_caption_benchmark.py \
    --model "${MODEL}" \
    --adapter openai-compatible \
    --base-url "${BASE_URL}" \
    --mode "${MODE}" \
    --input-type "${INPUT_TYPE}" \
    --num-workers "${WORKERS}" \
    --output-dir "${OUT_DIR}"

echo "Caption results saved to: ${OUT_DIR}"
