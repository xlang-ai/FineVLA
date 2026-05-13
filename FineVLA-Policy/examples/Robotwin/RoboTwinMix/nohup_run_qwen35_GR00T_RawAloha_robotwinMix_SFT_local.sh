#!/bin/bash
set -eo pipefail

PROJECT_ROOT=/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/StarVLA_YJH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_ROOT_DIR="${PROJECT_ROOT}/results/Checkpoints"
RUN_ID=qwen35_GR00T_RawAloha_RoboTwinMix_SFT
LOG_DIR="${PROJECT_ROOT}/results/logs/Robotwin/RoboTwinMix"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

mkdir -p "${RUN_ROOT_DIR}/${RUN_ID}"
mkdir -p "${LOG_DIR}"

nohup bash "${SCRIPT_DIR}/run_qwen35_GR00T_RawAloha_robotwinMix_SFT_local.sh" > "${LOG_FILE}" 2>&1 &
PID=$!

echo "Started ${RUN_ID} in background, PID=${PID}"
echo "Log file: ${LOG_FILE}"
