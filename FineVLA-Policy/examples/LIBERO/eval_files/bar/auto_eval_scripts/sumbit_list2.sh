


#!/bin/bash

ckpt_list=(
  "./results/Checkpoints/0222_QwenGR00TN1d6_epx3_randominti_multi_robot_4B/checkpoints/steps_60000_pytorch_model.pt"
  "./results/Checkpoints/0222_QwenGR00TN1d6_epx3_randominti_multi_robot_4B/checkpoints/steps_70000_pytorch_model.pt"
  "./results/Checkpoints/0222_QwenGR00TN1d6_epx3_randominti_multi_robot_4B/checkpoints/steps_80000_pytorch_model.pt"
  "./results/Checkpoints/0222_QwenGR00TN1d6_epx3_randominti_multi_robot_4B/checkpoints/steps_90000_pytorch_model.pt"
  "./results/Checkpoints/0222_QwenGR00TN1d6_epx3_randominti_multi_robot_4B/checkpoints/steps_100000_pytorch_model.pt"
)

run_index_base=80
cd /mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA

for ckpt in "${ckpt_list[@]}"; do
    echo "Evaluating checkpoint: $ckpt"
    bash /mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/examples/LIBERO/eval_files/bar/auto_eval_scripts/auto_eval_libero.sh "$ckpt" $((run_index_base + 0))
    sleep 10 # add a short delay between evaluations to avoid potential resource contention
    echo "Finished evaluating checkpoint: $ckpt"
done
