#!/bin/bash

cpkt_list=(
  "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0222_QwenGR00TN1d6_epx3_randominti_multi_robot_4B/checkpoints/steps_20000_pytorch_model.pt"
  "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0222_QwenGR00TN1d6_epx3_randominti_multi_robot_4B/checkpoints/steps_30000_pytorch_model.pt"
  "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0222_QwenGR00TN1d6_epx3_randominti_multi_robot_4B/checkpoints/steps_40000_pytorch_model.pt"
  "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0222_QwenGR00TN1d6_epx3_randominti_multi_robot_4B/checkpoints/steps_50000_pytorch_model.pt"
  "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0222_QwenGR00TN1d6_epx3_randominti_multi_robot_4B/checkpoints/steps_60000_pytorch_model.pt"
  "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0222_QwenGR00TN1d6_epx3_randominti_multi_robot_4B/checkpoints/steps_70000_pytorch_model.pt"
  "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0222_QwenGR00TN1d6_epx3_randominti_multi_robot_4B/checkpoints/steps_80000_pytorch_model.pt"
  "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0222_QwenGR00TN1d6_epx3_randominti_multi_robot_4B/checkpoints/steps_90000_pytorch_model.pt"
)


for ckpt in "${cpkt_list[@]}"; do
    echo "Evaluating checkpoint: $ckpt"
    bash /mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/examples/SimplerEnv/eval_files/bar/star_bridge.sh 6350 "$ckpt"
    sleep 10 # add a short delay between evaluations to avoid potential resource contention
    echo "Finished evaluating checkpoint: $ckpt"
done
