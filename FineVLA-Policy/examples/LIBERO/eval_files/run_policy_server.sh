#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
export starVLA_python=/root/miniconda3/envs/starVLA/bin/python  # Path to the Python environment

export LIBERO_Python=/root/miniconda3/envs/LIBERO/bin/python
your_ckpt=./results/Checkpoints/0126_libero_all_qwengr00tn1d6/checkpoints/steps_10000_pytorch_model.pt

gpu_id=7
port=5694
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${starVLA_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################
