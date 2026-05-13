


#!/bin/bash

ckpt_list=(
  "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0127_QwenGR00TN1d6_epx3_multi_robot/checkpoints/steps_120000_pytorch_model.pt"
  "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0127_QwenGR00TN1d6_epx3_multi_robot/checkpoints/steps_130000_pytorch_model.pt"
  "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0127_QwenGR00TN1d6_epx3_multi_robot/checkpoints/steps_140000_pytorch_model.pt"
  "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0127_QwenGR00TN1d6_epx3_multi_robot/checkpoints/steps_150000_pytorch_model.pt"
  "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0127_QwenGR00TN1d6_epx3_multi_robot/checkpoints/steps_160000_pytorch_model.pt"
  "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0127_QwenGR00TN1d6_epx3_multi_robot/checkpoints/steps_170000_pytorch_model.pt"
  "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0127_QwenGR00TN1d6_epx3_multi_robot/checkpoints/steps_180000_pytorch_model.pt"
  "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0127_QwenGR00TN1d6_epx3_multi_robot/checkpoints/steps_190000_pytorch_model.pt"
)

run_index_base=88
cd /mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA

for ckpt in "${ckpt_list[@]}"; do
    echo "Evaluating checkpoint: $ckpt"
    bash /mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/examples/LIBERO/eval_files/bar/auto_eval_scripts/auto_eval_libero.sh "$ckpt" $((run_index_base + 0))
    sleep 10 # add a short delay between evaluations to avoid potential resource contention
    echo "Finished evaluating checkpoint: $ckpt"
done
