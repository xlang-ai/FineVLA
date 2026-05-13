


#!/bin/bash

ckpt_list=(
  "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0127_QwenGR00TN1d6_epx3_multi_robot/checkpoints/steps_20000_pytorch_model.pt"
  "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0127_QwenGR00TN1d6_epx3_multi_robot/checkpoints/steps_30000_pytorch_model.pt"
  "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0127_QwenGR00TN1d6_epx3_multi_robot/checkpoints/steps_40000_pytorch_model.pt"
  "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0127_QwenGR00TN1d6_epx3_multi_robot/checkpoints/steps_50000_pytorch_model.pt"
  "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0127_QwenGR00TN1d6_epx3_multi_robot/checkpoints/steps_60000_pytorch_model.pt"
  "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0127_QwenGR00TN1d6_epx3_multi_robot/checkpoints/steps_70000_pytorch_model.pt"
  "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0127_QwenGR00TN1d6_epx3_multi_robot/checkpoints/steps_80000_pytorch_model.pt"
  "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0127_QwenGR00TN1d6_epx3_multi_robot/checkpoints/steps_90000_pytorch_model.pt"
)

run_index_base=24
cd /mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA

for ckpt in "${ckpt_list[@]}"; do
    echo "Evaluating checkpoint: $ckpt"
    bash /mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/examples/LIBERO/eval_files/bar/auto_eval_scripts/auto_eval_libero.sh "$ckpt" $((run_index_base + 0))
    sleep 10 # add a short delay between evaluations to avoid potential resource contention
    echo "Finished evaluating checkpoint: $ckpt"
done
