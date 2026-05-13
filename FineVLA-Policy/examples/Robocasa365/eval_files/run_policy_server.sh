#!/bin/bash

export PYTHONPATH=$(pwd):${PYTHONPATH}
export starVLA_python=/root/miniconda3/envs/starVLA/bin/python

your_ckpt=./results/Checkpoints/robocasa365_qwenoft/checkpoints/steps_10000_pytorch_model.pt
gpu_id=0
port=5678

CUDA_VISIBLE_DEVICES=$gpu_id ${starVLA_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16
