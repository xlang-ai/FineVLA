#!/bin/bash

# Debug: Print the current Python environment
echo "Using Python: $(which python)"


# Set required environment variables
export star_vla_python=/root/miniconda3/envs/starVLA/bin/python
export sim_python=/root/miniconda3/envs/behavior/bin/python
export TASKS_JSONL_PATH=./examples/Behavior/tasks.jsonl
export BEHAVIOR_ASSET_PATH=/mnt/workspace/jinhuiye/Projects/Pai-PhysxEvalTools/BenchDataEngines/BEHAVIOR-1K/datasets
export PYTHONPATH=$(pwd):${PYTHONPATH}

# Force Vulkan to use only the NVIDIA ICD to avoid duplicate ICDs seen by the loader
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
# Prefer NVIDIA GLX vendor when any GL deps are touched
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# Configure model path and port
# set the eval parameters
MODEL_PATH=/mnt/workspace/jinhuiye/Projects/Pai-PhysxEvalTools/InferModelEngines/starVLA/results/Checkpoints/StarVLA_0110_BEHAVIOR_rgp_dual_normbase_QwenOFT_task0_fullimage/checkpoints/steps_30000_pytorch_model.pt

PORT=10197
WRAPPERS="DefaultWrapper"
USE_STATE=False  # Whether to include state as part of the observation

# Configure task name
TASK_NAME="turning_on_radio"  # Choose a simple task


# Run a single task
export DEBUG=true
echo "▶️ Running task '${TASK_NAME}'..."
CUDA_VISIBLE_DEVICES=0 ${sim_python} examples/Behavior/start_behavior_env.py \
    --ckpt-path ${MODEL_PATH} \
    --eval-instance-ids "0 1 2 3 4 5 6 7 8 9" \
    --eval-on-train-instances False \
    --port ${PORT} \
    --task-name ${TASK_NAME} \
    --behavior-tasks-jsonl-path ${TASKS_JSONL_PATH} \
    --behavior-asset-path ${BEHAVIOR_ASSET_PATH} \
    --wrappers ${WRAPPERS} \
    --use-state ${USE_STATE}
    


