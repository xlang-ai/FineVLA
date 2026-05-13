#!/bin/bash
set -eo pipefail

# NCCL configuration for DLC multi-node training
# export NCCL_IB_TC=136
# export NCCL_IB_SL=5
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth
# export NCCL_DEBUG=INFO
# export NCCL_IB_HCA=mlx5
# export NCCL_IB_TIMEOUT=220
# export NCCL_IB_QPS_PER_CONNECTION=8
# export NCCL_MIN_NCHANNELS=4
# export NCCL_NET_PLUGIN=none
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USER=$(whoami)
export PRODUCT=1

# WANDB config
export WANDB_BASE_URL="http://47.92.143.221:7900/"
export WANDB_API_KEY="local-1647bcc5479ab0da87b5c153002383e4c8a83867"
export WANDB_DISABLE_CODE=true
export WANDB_CONSOLE="off"
export WANDB_MODE=online

# DLC distributed training args
GPUS_PER_NODE=8
NODE_ID=$RANK
MASTER_ADDR=$MASTER_ADDR
MASTER_PORT=$MASTER_PORT
TOTAL_GPUS=$(($GPUS_PER_NODE*$WORLD_SIZE))

echo "Node ID: $NODE_ID"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "World Size: $WORLD_SIZE"
echo "Total GPUs: $TOTAL_GPUS"

echo "RUNNING: QwenOFT + Qwen3.5-4B SFT from FGOnly_5w checkpoint on RoboTwin-Clean"

PROJECT_ROOT=/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/StarVLA_YJH
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
###########################################################################################
Framework_name=QwenOFT
freeze_module_list=''
base_vlm=/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/Pretrained_models/Qwen3.5-4B
config_yaml="${PROJECT_ROOT}/starVLA/config/training/starvla_robotwin_sft.yaml"
data_root=/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/StarVLA_YJH/Benchmark/RoboTwin-Clean
data_mix=robotwin_clean
run_root_dir="${PROJECT_ROOT}/results/Checkpoints"
run_id=qwen35_OFT_FGOnly_5w_RoboTwin_SFT
pretrained_checkpoint=/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/Qwen-StarVLA/results/Checkpoints/xintonghu_qwen3.5_OFT_FG_Only_dlc/checkpoints/steps_50000_pytorch_model.pt
###########################################################################################

output_dir="${run_root_dir}/${run_id}"
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

accelerate launch \
  --config_file "${PROJECT_ROOT}/starVLA/config/deepseeds/deepspeed_zero2.yaml" \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --machine_rank $RANK \
  --num_machines $WORLD_SIZE \
  --num_processes=${TOTAL_GPUS} \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --datasets.vla_data.data_root_dir "${data_root}" \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.pretrained_checkpoint "${pretrained_checkpoint}" \
  --trainer.max_train_steps 200000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --trainer.learning_rate.base 5e-6 \
  --trainer.learning_rate.qwen_vl_interface 5e-6 \
  --trainer.learning_rate.action_model 5e-5 \
  --trainer.lr_scheduler_type constant \
  --trainer.num_warmup_steps 0 \
  --run_root_dir "${run_root_dir}" \
  --run_id ${run_id} \
  --wandb_project xintonghu_qwen3.5_OFT_RoboTwin_SFT \
  --wandb_entity rjucdvfh04
