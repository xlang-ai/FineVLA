#!/bin/bash
set -eo pipefail

ulimit -n 1048576
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
# export WANDB_BASE_URL="http://47.92.143.221:7900/"
# export WANDB_API_KEY="local-1647bcc5479ab0da87b5c153002383e4c8a83867"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate starVLA

PROJECT_ROOT=/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/StarVLA_YJH
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

###########################################################################################
Framework_name=QwenGR00T
freeze_module_list=''
base_vlm=/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/Pretrained_models/Qwen3.5-4B
config_yaml="${PROJECT_ROOT}/starVLA/config/training/starvla_rdt_yhq_GR00TN1d6_epx3.yaml"
data_root=/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21/RDT-yhq
data_mix=rdt_yhq_FG1_1
run_root_dir="${PROJECT_ROOT}/results/Checkpoints"
run_id=qwen35_GR00T_FG1_1_RDT_local
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
  --datasets.vla_data.data_root_dir "${data_root}" \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps 10 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --trainer.learning_rate.base 2e-5 \
  --trainer.learning_rate.qwen_vl_interface 1e-5 \
  --trainer.learning_rate.action_model 1e-4 \
  --trainer.lr_scheduler_type cosine_with_min_lr \
  --trainer.num_warmup_steps 5000 \
  --run_root_dir "${run_root_dir}" \
  --run_id ${run_id} \
  --wandb_project qwen35_GR00T_RDT\
  --wandb_entity rjucdvfh04
