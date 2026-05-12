#!/bin/bash
#
# VQA 批量评测脚本（支持多轮）
#
# 用法:
#   bash run_vqa_eval.sh              # 默认 3 轮
#   bash run_vqa_eval.sh 5            # 指定 5 轮
#
# 视角和 FPS 配置在 vqa_config.py 中修改
# OSS URL 模式: 运行前先执行 python3 upload_vqa_frames.py --workers 8

set -e
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy="oss-cn-shanghai.aliyuncs.com,oss-cn-beijing.aliyuncs.com"

# Auto-detect paths relative to this script (works after git clone)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VQA_DIR="${SCRIPT_DIR}"
BASE_DIR="$(dirname "${SCRIPT_DIR}")"

# ══════════════════════════════════════════════════════════
#  配置
# ══════════════════════════════════════════════════════════

NUM_ROUNDS=${1:-2}      # 默认 2 轮，可通过第一个参数覆盖
END_SAMPLES=${2:-}      # 可选：只测前 N 条样本（如 2），留空则测全部

MODEL_LIST=(
    qwen3-vl-plus
    qwen3.5-plus
    doubao.doubao-seed-2-0-pro-260215
    vertex_ai.gemini-3.1-pro-preview
    openai.gpt-5.4-2026-03-05
)

THINKING="true"    # 所有模型开启 thinking/reasoning
NUM_WORKERS=16     # 并行 API 调用数

# ══════════════════════════════════════════════════════════
#  批量评测
# ══════════════════════════════════════════════════════════

TOTAL_MODELS=${#MODEL_LIST[@]}
TOTAL_RUNS=$((TOTAL_MODELS * NUM_ROUNDS))

echo "============================================================"
echo "  VQA 多轮批量评测"
echo "  模型列表:     ${MODEL_LIST[*]}"
echo "  模型数量:     ${TOTAL_MODELS}"
echo "  评测轮次:     ${NUM_ROUNDS}"
echo "  总评测次数:   ${TOTAL_RUNS}"
echo "  Thinking:     ${THINKING}"
echo "  Workers:      ${NUM_WORKERS}"
echo "  样本范围:     ${END_SAMPLES:-全部}"
echo "============================================================"
echo ""

cd "${VQA_DIR}"

CURRENT=0
for ROUND in $(seq 1 ${NUM_ROUNDS}); do
    echo ""
    echo "============================================================"
    echo "  第 ${ROUND}/${NUM_ROUNDS} 轮评测"
    echo "============================================================"
    echo ""

    for MODEL_NAME in "${MODEL_LIST[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo ""
        echo "------------------------------------------------------------"
        echo "  [${CURRENT}/${TOTAL_RUNS}] Round ${ROUND} - 模型: ${MODEL_NAME}"
        echo "------------------------------------------------------------"
        echo ""

        EXTRA_ARGS=""
        if [ -n "${END_SAMPLES}" ]; then
            EXTRA_ARGS="--start 0 --end ${END_SAMPLES}"
        fi

        python3 run_vqa.py \
            --model "${MODEL_NAME}" \
            --thinking "${THINKING}" \
            --num-workers "${NUM_WORKERS}" \
            --round "${ROUND}" \
            ${EXTRA_ARGS}

        echo ""
        echo "  [${CURRENT}/${TOTAL_RUNS}] ${MODEL_NAME} Round ${ROUND} 完成！"
        echo ""
    done
done

echo ""
echo "============================================================"
echo "  全部评测完成！${TOTAL_MODELS} 个模型 x ${NUM_ROUNDS} 轮 = ${TOTAL_RUNS} 次评测"
echo "  结果目录: ${VQA_DIR}/results/"
echo "============================================================"
