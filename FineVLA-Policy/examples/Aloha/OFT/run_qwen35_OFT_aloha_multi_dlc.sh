#!/bin/bash
set -eo pipefail

export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export WANDB_BASE_URL="http://47.92.143.221:7900/"
export WANDB_API_KEY="local-1647bcc5479ab0da87b5c153002383e4c8a83867"
export WANDB_DISABLE_CODE=true
export WANDB_CONSOLE="off"
export WANDB_MODE=online

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_DETERMINISTIC=1
export NCCL_ASYNC_ERROR_HANDLING=0

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

echo "RUNNING: QwenOFT + Qwen3.5-4B on Aloha Multi-Dataset (7 groups, 595 tasks)"

PROJECT_ROOT=/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/StarVLA_YJH
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

###########################################################################################
Framework_name=QwenOFT
freeze_module_list=''
base_vlm=/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/Pretrained_models/Qwen3.5-4B
config_yaml="${PROJECT_ROOT}/starVLA/config/training/starvla_aloha_multi_OFT.yaml"
data_root=/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt
data_mix=aloha_multi_mix
run_root_dir="${PROJECT_ROOT}/results/Checkpoints/OFT_ALOHA"
run_id=RawAloha_OFT
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
  --datasets.vla_data.per_device_batch_size 4 \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps 200000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --trainer.learning_rate.base 2e-5 \
  --trainer.learning_rate.qwen_vl_interface 1e-5 \
  --trainer.learning_rate.action_model 1e-4 \
  --trainer.lr_scheduler_type cosine_with_min_lr \
  --trainer.num_warmup_steps 5000 \
  --run_root_dir "${run_root_dir}" \
  --run_id ${run_id} \
  --wandb_project qwen35_OFT_aloha_multi \
  --wandb_entity rjucdvfh04
