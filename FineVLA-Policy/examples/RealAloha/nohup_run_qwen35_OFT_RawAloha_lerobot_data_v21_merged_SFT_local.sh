#!/bin/bash
set -eo pipefail

PROJECT_ROOT=/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/StarVLA_YJH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_ROOT_DIR="${PROJECT_ROOT}/results/Checkpoints/OFT_ALOHA"
RUN_ID=RawAloha_OFT_lerobot_data_v21_merged_SFT_local
LOG_DIR="${PROJECT_ROOT}/results/logs/RealAloha"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

mkdir -p "${RUN_ROOT_DIR}/${RUN_ID}"
mkdir -p "${LOG_DIR}"

nohup bash "${SCRIPT_DIR}/run_qwen35_OFT_RawAloha_lerobot_data_v21_merged_SFT_local.sh" > "${LOG_FILE}" 2>&1 &
PID=$!

echo "Started ${RUN_ID} in background, PID=${PID}"
echo "Log file: ${LOG_FILE}"
