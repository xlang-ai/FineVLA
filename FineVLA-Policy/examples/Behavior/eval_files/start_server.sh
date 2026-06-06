#!/bin/bash

# Debug: Print the current Python environment
echo "Using Python: $(which python)"

# Set required environment variables
export star_vla_python=/gpfs/wangzixuan/conda_envs/starVLA/bin/python
export sim_python=/gpfs/wangzixuan/conda_envs/behavior/bin/python
export BEHAVIOR_PATH=/gpfs/wangzixuan/Jinhui/llavavla0/playground/Datasets/BEHAVIOR_challenge
export PYTHONPATH=$(pwd):${PYTHONPATH}

# Force Vulkan to use only the NVIDIA ICD to avoid duplicate ICDs seen by the loader
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
# Prefer NVIDIA GLX vendor when any GL deps are touched
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# Configure model path and port
MODEL_PATH="/gpfs/wangzixuan/Jinhui/llavavla0/playground/Checkpoints/1031_BEHAVIOR_challenge_qwengroot/checkpoints/steps_40000_pytorch_model.pt"
PORT=10197
WRAPPERS="DefaultWrapper"
USE_STATE=False  # Whether to include state as part of the observation

# Configure task name
TASK_NAME="turning_on_radio"  # Choose a simple task
CLIENT_LOG_FILE="/gpfs/wangzixuan/Jinhui/llavavla0/playground/Checkpoints/1031_BEHAVIOR_challenge_qwengroot/client_log/log_${TASK_NAME}.txt"
SERVER_LOG_FILE="/gpfs/wangzixuan/Jinhui/llavavla0/playground/Checkpoints/1031_BEHAVIOR_challenge_qwengroot/server_log/log_${TASK_NAME}.txt"


# Start the service
echo "▶️ Starting server on port ${PORT}..."
CUDA_VISIBLE_DEVICES=6 ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${MODEL_PATH} \
    --port ${PORT} \
    --is_debug \
    --use_bf16
    
    #  > ${SERVER_LOG_FILE} 2>&1 &


# SERVER_PID=$!
# sleep 15  # Wait for the server to start

# # Check if the server started successfully
# if ps -p ${SERVER_PID} > /dev/null; then
#     echo "✅ Server started successfully (PID: ${SERVER_PID})"
# else
#     echo "❌ Failed to start server"
#     exit 1
# fi