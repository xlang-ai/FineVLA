#!/bin/bash
set -eo pipefail

ulimit -n 1048576
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

source /root/miniconda3/etc/profile.d/conda.sh
conda activate starVLA

PROJECT_ROOT=/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/StarVLA_YJH
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

###########################################################################################
Framework_name=QwenOFT
freeze_module_list=''
base_vlm=/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/Pretrained_models/Qwen3.5-4B
config_yaml="${PROJECT_ROOT}/starVLA/config/training/starvla_aloha_multi_OFT.yaml"
data_root=/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt
data_mix=aloha_multi_FG1_1
run_root_dir="${PROJECT_ROOT}/results/Checkpoints/OFT_ALOHA"
run_id=aloha_multi_oft_fg1_1_test
###########################################################################################

output_dir="${run_root_dir}/${run_id}"
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

accelerate launch \
  --config_file "${PROJECT_ROOT}/starVLA/config/deepseeds/deepspeed_zero2.yaml" \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --framework.qwenvl.max_seq_length 1024 \
  --datasets.vla_data.data_root_dir "${data_root}" \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 4 \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps 1000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 10000 \
  --trainer.learning_rate.base 2e-5 \
  --trainer.learning_rate.qwen_vl_interface 1e-5 \
  --trainer.learning_rate.action_model 1e-4 \
  --trainer.lr_scheduler_type cosine_with_min_lr \
  --trainer.num_warmup_steps 5000 \
  --run_root_dir "${run_root_dir}" \
  --run_id ${run_id} \
  --trackers jsonl
