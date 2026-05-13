#!/bin/bash

cd /mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA
SCRIPT_PATH="./bar/starVLA-0227/examples/LIBERO/eval_files/bar/auto_eval_scripts/eval_libero_server_and_client.sh"

your_ckpt=/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/bar/starVLA-0227/results/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt
run_index_base=198


your_ckpt=${1:-$your_ckpt}
run_index_base=${2:-$run_index_base}

# #####################################################
# task_suite_name=libero_10 # align with your model
# run_index=$((run_index_base + 0))
# bash $SCRIPT_PATH $your_ckpt $task_suite_name $run_index
# #####################################################

# sleep 15
# #####################################################
task_suite_name=libero_goal # align with your model
run_index=$((run_index_base + 1))
bash $SCRIPT_PATH $your_ckpt $task_suite_name $run_index
#####################################################
# sleep 15
# #####################################################
# task_suite_name=libero_object # align with your model
# run_index=$((run_index_base + 2))
# bash $SCRIPT_PATH $your_ckpt $task_suite_name $run_index &
# #####################################################
# sleep 15
####################################################
# task_suite_name=libero_spatial # align with your model
# run_index=$((run_index_base + 3))
# bash $SCRIPT_PATH $your_ckpt $task_suite_name $run_index &
#####################################################