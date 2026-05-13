#!/bin/bash

cpkt_list=(
  "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0127_QwenGR00TN1d6_epx3_multi_robot/checkpoints/steps_120000_pytorch_model.pt"
  "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0127_QwenGR00TN1d6_epx3_multi_robot/checkpoints/steps_130000_pytorch_model.pt"
  "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0127_QwenGR00TN1d6_epx3_multi_robot/checkpoints/steps_140000_pytorch_model.pt"
  "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0127_QwenGR00TN1d6_epx3_multi_robot/checkpoints/steps_150000_pytorch_model.pt"
  "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0127_QwenGR00TN1d6_epx3_multi_robot/checkpoints/steps_160000_pytorch_model.pt"
  "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0127_QwenGR00TN1d6_epx3_multi_robot/checkpoints/steps_170000_pytorch_model.pt"
  "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0127_QwenGR00TN1d6_epx3_multi_robot/checkpoints/steps_180000_pytorch_model.pt"
  "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0127_QwenGR00TN1d6_epx3_multi_robot/checkpoints/steps_190000_pytorch_model.pt"
)


for ckpt in "${cpkt_list[@]}"; do
    echo "Evaluating checkpoint: $ckpt"
    bash /mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/examples/SimplerEnv/eval_files/bar/star_bridge.sh 6150 "$ckpt"
    sleep 10 # add a short delay between evaluations to avoid potential resource contention
    echo "Finished evaluating checkpoint: $ckpt"
done
