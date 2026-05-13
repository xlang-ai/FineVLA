#!/bin/bash

ckpt_list=(
  "./results/Checkpoints/0127_QwenGR00TN1d6_epx3_multi_robot/checkpoints/steps_110000_pytorch_model.pt"
  "./results/Checkpoints/0127_QwenGR00TN1d6_epx3_multi_robot/checkpoints/steps_120000_pytorch_model.pt"
  "./results/Checkpoints/0127_QwenGR00TN1d6_epx3_multi_robot/checkpoints/steps_130000_pytorch_model.pt"
  "./results/Checkpoints/0127_QwenGR00TN1d6_epx3_multi_robot/checkpoints/steps_140000_pytorch_model.pt"
)


cd /mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA

for ckpt in "${ckpt_list[@]}"; do
    echo "Evaluating checkpoint: $ckpt"
    bash /mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/examples/SimplerEnv/eval_files/bar/star_bridge.sh 6350 "$ckpt"
    sleep 10 # add a short delay between evaluations to avoid potential resource contention
    echo "Finished evaluating checkpoint: $ckpt"
done
