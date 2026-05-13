#!/bin/bash
# =============================================================================
# LeRobot v2.1 Dataset Verification Script
# =============================================================================
# 
# 本脚本支持三种验证模式：
#   模式1: 单数据集验证 - 验证单个 LeRobot v2.1 数据集
#   模式2: 递归批量验证 - 递归搜索目录下所有 v2.1 数据集并验证
#   模式3: 全量子目录验证 - 验证根目录下的所有直接子目录数据集 (带进度条)
#
# 使用方法:
#   1. 选择验证模式 (修改 MODE 变量为 1, 2, 或 3)
#   2. 修改对应模式的配置参数
#   3. 运行: bash verify_v21_dataset.sh
#
# =============================================================================

# 提高单进程可打开文件数，避免 "Too many open files"（模式2/3 验证大量数据集时会打开很多文件）
ulimit -n 1048576 2>/dev/null || true

# ========================== 选择验证模式 ==========================
# 模式选择: 1 = 单数据集, 2 = 递归批量, 3 = 全量子目录
MODE=1

# ========================== 通用配置参数 ==========================
# 采样验证的 episode 数量 (详细检查时随机采样的 episode 数)
SAMPLE_SIZE=1

# 是否显示详细输出 (true/false)
# - VERBOSE=true:  实时显示所有详细输出
# - VERBOSE=false: 静默模式，将详细输出保存到日志文件，最后汇总显示错误信息
VERBOSE=true

# Conda 环境名称
CONDA_ENV="any4lerobot"

# 脚本路径 (一般不需要修改)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/verify_v21_dataset.py"

# 错误日志目录
LOGS_DIR="${SCRIPT_DIR}/logs"

# 路径前缀 (用于生成日志文件名时去除)
PATH_PREFIX="/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21/"

# =============================================================================
# ========================== 模式1: 单数据集验证 ==========================
# 验证单个 LeRobot v2.1 数据集
# =============================================================================
# MODE1_ROOT="/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21/Galaxea-Open-World-Dataset/Adjust_The_Air_Conditioner_Temperature_20250711_006/Adjust_The_Air_Conditioner_Temperature_20250711_006"
# MODE1_ROOT="/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21/RoboMIND_lerobot_v21/benchmark1_0_compressed/agilex_3rgb/1_potatooven"
# MODE1_ROOT="/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21/RoboCOIN_add1201/RoboCOIN/leju_robot_box_storage_parcel_i"
# MODE1_ROOT="/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/RoboMindV2.0-Lerobot/agilex/arrange_blocks_and_place_orange_in_center_with_arms"

MODE1_ROOT="/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/RoboMindV2.0-Lerobot/franka_sim/135-pass_drainage_pipe_clamp"
# =============================================================================


# ========================== 模式2: 递归批量验证 ==========================
# 递归搜索目录下所有包含 meta/info.json 的数据集 (版本检查在验证时进行)
# =============================================================================
# MODE2_ROOT="/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21/RoboCOIN"
# MODE2_ROOT="/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21/Galaxea-Open-World-Dataset" 
# MODE2_ROOT="/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21/RoboMIND_lerobot_v21/benchmark1_0_compressed/agilex_3rgb"
# MODE2_ROOT="/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21/RoboMIND_lerobot_v21"
# MODE2_ROOT="/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21/Galaxea-Open-World-Dataset"
# MODE2_ROOT="/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21/RoboCOIN"/
MODE2_ROOT="/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21/RoboMIND_lerobot_v2.1_0211zj"
# MODE2_ROOT="/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21/RoboCOIN/AgiBot-g1_battery_storage_b"
# MODE2_ROOT="/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21/RoboCOIN_add1201"
# MODE2_ROOT="/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21/droid_1.0.1"
# =============================================================================



# ========================== 模式3: 全量子目录验证 ==========================
# 验证根目录下的所有直接子目录数据集，带 tqdm 进度条
# 假设目录结构:
#   datasets_root/
#   ├── dataset_1/ (子数据集)
#   ├── dataset_2/ (子数据集)
#   └── ...
# =============================================================================
# MODE3_DATASETS_ROOT="/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21/RoboMIND_lerobot_v21"
# MODE3_DATASETS_ROOT="/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21/RoboCOIN"
# MODE3_DATASETS_ROOT="/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/RoboMindV2.0-Lerobot/agilex_mobile"
# MODE3_DATASETS_ROOT="/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/RoboMindV2.0-Lerobot/tienyi"
MODE3_DATASETS_ROOT="/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21/RoboMIND_lerobot_v2.1_0211zj"

# MODE3_DATASETS_ROOT="/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21/RH20T"
# MODE3_DATASETS_ROOT="/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21/Galaxea-Open-World-Dataset" 测试
# =============================================================================
# ========================== 以下内容一般不需要修改 ==========================
# =============================================================================

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# 临时日志文件路径 (用于收集完整输出)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEMP_LOG_FILE="${SCRIPT_DIR}/.verify_v21_${TIMESTAMP}.log"

# 创建日志目录
mkdir -p "$LOGS_DIR"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  LeRobot v2.1 Dataset Verification${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查脚本是否存在
if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo -e "${RED}Error: Script not found: $SCRIPT_PATH${NC}"
    exit 1
fi

# 激活 conda 环境
echo -e "${YELLOW}Activating conda environment: ${CONDA_ENV}${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

if [[ $? -ne 0 ]]; then
    echo -e "${RED}Error: Failed to activate conda environment: $CONDA_ENV${NC}"
    exit 1
fi

# 根据模式构建命令，并设置 CURRENT_ROOT 用于生成日志文件名
case $MODE in
    1)
        echo -e "${CYAN}验证模式: 模式1 - 单数据集验证${NC}"
        echo ""
        
        # 检查路径
        if [[ ! -e "$MODE1_ROOT" ]]; then
            echo -e "${RED}Error: Path does not exist: $MODE1_ROOT${NC}"
            exit 1
        fi
        
        CURRENT_ROOT="$MODE1_ROOT"
        MODE_STR="mode1"
        # CMD 将在 LOG_NAME 设置后构建
        
        echo -e "${YELLOW}Configuration:${NC}"
        echo "  数据集路径:   $MODE1_ROOT"
        echo "  采样数量:     $SAMPLE_SIZE"
        echo "  详细输出:     $VERBOSE"
        ;;
    
    2)
        echo -e "${CYAN}验证模式: 模式2 - 递归批量验证${NC}"
        echo ""
        
        # 检查路径
        if [[ ! -e "$MODE2_ROOT" ]]; then
            echo -e "${RED}Error: Path does not exist: $MODE2_ROOT${NC}"
            exit 1
        fi
        
        CURRENT_ROOT="$MODE2_ROOT"
        MODE_STR="mode2"
        # CMD 将在 ERROR_SUMMARY_JSON 设置后构建
        
        echo -e "${YELLOW}Configuration:${NC}"
        echo "  搜索根目录:   $MODE2_ROOT"
        echo "  采样数量:     $SAMPLE_SIZE"
        echo "  搜索方式:     递归搜索所有包含 meta/info.json 的数据集"
        ;;
    
    3)
        echo -e "${CYAN}验证模式: 模式3 - 全量子目录验证 (带进度条)${NC}"
        echo ""
        
        # 检查路径
        if [[ ! -e "$MODE3_DATASETS_ROOT" ]]; then
            echo -e "${RED}Error: Path does not exist: $MODE3_DATASETS_ROOT${NC}"
            exit 1
        fi
        
        CURRENT_ROOT="$MODE3_DATASETS_ROOT"
        MODE_STR="mode3"
        # CMD 将在 ERROR_SUMMARY_JSON 设置后构建
        
        echo -e "${YELLOW}Configuration:${NC}"
        echo "  数据集根目录: $MODE3_DATASETS_ROOT"
        echo "  采样数量:     $SAMPLE_SIZE"
        echo "  搜索方式:     验证所有直接子目录"
        ;;
    
    *)
        echo -e "${RED}Error: Invalid MODE=$MODE. Please set MODE to 1, 2, or 3.${NC}"
        exit 1
        ;;
esac

# 生成日志文件名: 去除前缀，将 / 替换为 -，加上模式和时间戳
LOG_NAME=$(echo "$CURRENT_ROOT" | sed "s|${PATH_PREFIX}||" | sed 's|/|-|g' | sed 's|^-||')
ERROR_LOG_FILE="${LOGS_DIR}/${LOG_NAME}_${MODE_STR}_${TIMESTAMP}.log"
ERROR_SUMMARY_JSON="${LOGS_DIR}/${LOG_NAME}_${MODE_STR}_${TIMESTAMP}_error.json"

# 根据模式构建 CMD（需在 ERROR_SUMMARY_JSON 确定后传入 --error-json）
case $MODE in
    1)
        CMD="python \"$SCRIPT_PATH\" --root \"$MODE1_ROOT\" --sample-size $SAMPLE_SIZE --error-json \"$ERROR_SUMMARY_JSON\""
        if [[ "$VERBOSE" == "true" ]]; then
            CMD="$CMD --verbose"
        fi
        ;;
    2)
        CMD="python \"$SCRIPT_PATH\" --root \"$MODE2_ROOT\" --batch --sample-size $SAMPLE_SIZE --error-json \"$ERROR_SUMMARY_JSON\""
        ;;
    3)
        CMD="python \"$SCRIPT_PATH\" --full-datasets --datasets-root \"$MODE3_DATASETS_ROOT\" --sample-size $SAMPLE_SIZE --error-json \"$ERROR_SUMMARY_JSON\""
        if [[ "$VERBOSE" == "true" ]]; then
            CMD="$CMD --verbose"
        fi
        ;;
esac

echo ""
echo -e "${YELLOW}Running command:${NC}"
echo "  $CMD"
echo -e "${YELLOW}验证日志将保存到:${NC}"
echo "  $ERROR_LOG_FILE"
echo ""
echo -e "${GREEN}----------------------------------------${NC}"
echo ""

# =============================================================================
# 执行验证，同时捕获输出用于生成错误日志
# =============================================================================

# 始终捕获完整输出到临时文件
if [[ "$VERBOSE" == "false" ]]; then
    # VERBOSE=false: 静默模式，不实时显示
    echo -e "${CYAN}运行模式: 静默模式 (VERBOSE=false)${NC}"
    echo -e "${CYAN}  - 详细输出将保存到临时日志文件${NC}"
    echo -e "${CYAN}  - 错误信息将在最后汇总显示${NC}"
    echo ""
    eval $CMD > "$TEMP_LOG_FILE" 2>&1
    EXIT_CODE=$?
else
    # VERBOSE=true: 详细模式，实时显示同时保存到文件
    eval $CMD 2>&1 | tee "$TEMP_LOG_FILE"
    EXIT_CODE=${PIPESTATUS[0]}
fi

# =============================================================================
# 生成验证日志文件 (保留完整输出，去除 ANSI 转义码)
# =============================================================================
# 若临时日志很大（如模式2验证数百个数据集），此处可能需数十秒，属正常现象
echo -e "${CYAN}正在生成验证日志（若日志较大请稍候）...${NC}"

# 函数: 去除 ANSI 转义码 (更全面的处理)
strip_ansi() {
    # 使用 perl 处理更复杂的 ANSI 转义序列
    perl -pe 's/\e\[[0-9;]*m//g; s/\[([0-9]+m|0m)//g'
}

generate_error_log() {
    local temp_log="$1"
    local error_log="$2"
    
    # 清空或创建日志文件
    > "$error_log"
    
    # 写入日志头部
    cat >> "$error_log" << 'HEADER'
######################################################################
#                                                                    #
#           LeRobot v2.1 数据集验证 - 完整日志                       #
#                                                                    #
######################################################################
HEADER
    echo "" >> "$error_log"
    echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$error_log"
    echo "验证路径: $CURRENT_ROOT" >> "$error_log"
    echo "验证模式: 模式${MODE}" >> "$error_log"
    echo "采样数量: $SAMPLE_SIZE" >> "$error_log"
    echo "" >> "$error_log"
    cat >> "$error_log" << 'HEADER2'
######################################################################
#                        完整验证输出                                #
######################################################################

HEADER2
    
    # 将完整输出（去除 ANSI 转义码）追加到日志文件
    strip_ansi < "$temp_log" >> "$error_log"
    
    # 添加日志结束标记
    cat >> "$error_log" << 'FOOTER'

######################################################################
#                        日志生成完成                                #
######################################################################
FOOTER
}

# 生成错误日志
generate_error_log "$TEMP_LOG_FILE" "$ERROR_LOG_FILE"
echo -e "${GREEN}验证日志已写入: $ERROR_LOG_FILE${NC}"

# =============================================================================
# 显示结果汇总
# =============================================================================
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}        验证执行完成${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 显示简要统计信息
TOTAL_DATASETS=$(grep -c "开始验证数据集:" "$TEMP_LOG_FILE" 2>/dev/null || echo "0")
PASSED_COUNT=$(grep -c "验证通过" "$TEMP_LOG_FILE" 2>/dev/null || echo "0")
FAILED_COUNT=$(grep -c "验证失败\|验证异常" "$TEMP_LOG_FILE" 2>/dev/null || echo "0")

echo -e "${CYAN}统计信息:${NC}"
echo "  验证的数据集总数: $TOTAL_DATASETS"
echo -e "  ${GREEN}通过: $PASSED_COUNT${NC}"
echo -e "  ${RED}失败: $FAILED_COUNT${NC}"
echo ""

# 如果有失败，在 VERBOSE=false 模式下显示简要错误汇总
if [[ "$VERBOSE" == "false" ]] && [[ $FAILED_COUNT -gt 0 ]]; then
    echo -e "${RED}${BOLD}失败的数据集列表:${NC}"
    grep -E ">>> .+: 验证失败|>>> .+: 验证异常" "$TEMP_LOG_FILE" | head -20 | while read line; do
        echo -e "  ${RED}$line${NC}"
    done
    if [[ $FAILED_COUNT -gt 20 ]]; then
        echo -e "  ${YELLOW}... 还有 $((FAILED_COUNT - 20)) 个失败的数据集，请查看错误日志${NC}"
    fi
    echo ""
fi

echo -e "${CYAN}日志文件:${NC}"
echo -e "  ${YELLOW}验证日志: $ERROR_LOG_FILE${NC}"
echo -e "  ${YELLOW}错误汇总: $ERROR_SUMMARY_JSON${NC}"

echo ""
echo -e "${GREEN}----------------------------------------${NC}"

if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}✓ Verification completed successfully!${NC}"
else
    echo -e "${RED}✗ Verification failed with exit code: $EXIT_CODE${NC}"
fi

# 显示 error.json 内容摘要（通过参数传路径，避免双引号内变量/关键字被 bash 解析）
if [[ -f "$ERROR_SUMMARY_JSON" ]]; then
    ERROR_COUNT=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(len(d.get("error",{})))' "$ERROR_SUMMARY_JSON" 2>/dev/null || echo "0")
    if [[ "$ERROR_COUNT" -gt 0 ]]; then
        echo ""
        echo -e "${RED}${BOLD}错误汇总 (共 $ERROR_COUNT 个失败数据集):${NC}"
        python3 -c '
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
    errors = data.get("error", {})
    for i, (ds, reason) in enumerate(list(errors.items())[:10]):
        print(f"  {i+1}. {ds}")
        msg = reason[:100] + "..." if len(reason) > 100 else reason
        print(f"     原因: {msg}")
    if len(errors) > 10:
        print(f"  ... 还有 {len(errors) - 10} 个错误，请查看错误汇总 JSON 文件")
' "$ERROR_SUMMARY_JSON" 2>/dev/null
    fi
fi

# 清理临时日志文件 (完整日志已保存到 ERROR_LOG_FILE)
rm -f "$TEMP_LOG_FILE" 2>/dev/null

exit $EXIT_CODE
