#!/bin/bash
# =============================================================================
# LeRobot 数据集时长统计脚本
# =============================================================================
#
# 从 Lerobot_v20 和 Lerobot_v21 下所有数据集的 meta/info.json 读取
# total_frames 和 fps，计算 duration = total_frames / fps。
# 子数据集的时长相加作为大数据集的总时长。
# 默认输出：JSON、XLSX 表格、PNG 饼图（同时间戳命名）。
#
# 使用方法:
#   1. 按需修改下方配置参数
#   2. 运行: bash calc_video_duration.sh
#
# =============================================================================

# ========================== 配置参数 ==========================

# 数据根目录 (空格分隔多个路径)
DATA_ROOTS=(
    "/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v20"
    "/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21"
)

# 并行线程数 (留空则自动设为 CPU 核数)
NUM_WORKERS=

# Conda 环境名称
CONDA_ENV="any4lerobot"

# ========================== 以下一般不需要修改 ==========================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/calc_video_duration.py"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${SCRIPT_DIR}/dataset_durations_${TIMESTAMP}.json"

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  LeRobot Dataset Duration Calculator${NC}"
echo -e "${GREEN}  (via info.json: total_frames / fps)${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查 Python 脚本
if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo -e "${RED}Error: Python script not found: $SCRIPT_PATH${NC}"
    exit 1
fi

# 激活 conda
echo -e "${YELLOW}Activating conda environment: ${CONDA_ENV}${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
if [[ $? -ne 0 ]]; then
    echo -e "${RED}Error: Failed to activate conda environment: $CONDA_ENV${NC}"
    exit 1
fi

# 构建命令（默认输出 json + xlsx + png 饼图）
CMD="python \"$SCRIPT_PATH\""
CMD="$CMD --roots ${DATA_ROOTS[*]}"
CMD="$CMD --output \"$OUTPUT_FILE\""
CMD="$CMD --xlsx"

if [[ -n "$NUM_WORKERS" ]]; then
    CMD="$CMD --num-workers $NUM_WORKERS"
fi

# 与 --output 同目录、同时间戳的 xlsx 和 png 路径
OUTPUT_STEM="${SCRIPT_DIR}/dataset_durations_${TIMESTAMP}"
OUTPUT_XLSX="${OUTPUT_STEM}.xlsx"
OUTPUT_PNG="${OUTPUT_STEM}_pie.png"

echo -e "${CYAN}Configuration:${NC}"
echo "  Data roots:   ${DATA_ROOTS[*]}"
echo "  Workers:      ${NUM_WORKERS:-auto}"
echo "  Output JSON:  $OUTPUT_FILE"
echo "  Output XLSX:  $OUTPUT_XLSX"
echo "  Output PNG:   $OUTPUT_PNG"
echo ""
echo -e "${YELLOW}Running:${NC}"
echo "  $CMD"
echo ""
echo -e "${GREEN}----------------------------------------${NC}"
echo ""

eval $CMD
EXIT_CODE=$?

echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}✓ Done! Results saved to:${NC}"
    echo -e "  JSON: $OUTPUT_FILE"
    echo -e "  XLSX: $OUTPUT_XLSX"
    echo -e "  PNG:  $OUTPUT_PNG"
else
    echo -e "${RED}✗ Failed with exit code: $EXIT_CODE${NC}"
fi

exit $EXIT_CODE
