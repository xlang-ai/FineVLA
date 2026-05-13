#!/bin/bash

ROOT_DIR="/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0222_QwenGR00TN1d6_epx3_randominti_multi_robot_4B"
STEP_LIST=(

  "10000"
  "10000"
  "20000"
  "30000"
  "40000"
  "50000"
  "60000"
  "70000"
  "80000"
  "90000"
)

OUT_DIR="${ROOT_DIR}/eval_visuals/robocas"
mkdir -p "${OUT_DIR}"

for STEP in "${STEP_LIST[@]}"; do
  relative_path="videos/steps_${STEP}_pytorch_model/n_action_steps_12_max_episode_steps_720_n_envs_1_gr1_unified"
  TASK_ROOT="${ROOT_DIR}/${relative_path}"
  OUT_FILE="${OUT_DIR}/step_${STEP}.log"

  if [ ! -d "${TASK_ROOT}" ]; then
    echo "[WARN] Missing task root, skip step ${STEP}: ${TASK_ROOT}"
    continue
  fi

  echo "[INFO] Processing step ${STEP}"
  python /mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/examples/Robocasa_tabletop/eval_files/bar/robocasa_video_success.py \
    --root "${TASK_ROOT}" \
    --expected_tasks 24 \
    --out "${OUT_FILE}"
done


