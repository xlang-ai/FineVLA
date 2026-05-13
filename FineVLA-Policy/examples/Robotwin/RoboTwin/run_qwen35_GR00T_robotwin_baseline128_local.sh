#!/bin/bash
set -eo pipefail

ulimit -n 1048576
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export TOKENIZERS_PARALLELISM=false

source /root/miniconda3/etc/profile.d/conda.sh
conda activate starVLA

###########################################################################################
# Configuration — edit these variables as needed
PROJECT_ROOT=/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/StarVLA_YJH
Framework_name=QwenGR00T
base_vlm=/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/Pretrained_models/Qwen3.5-4B
config_yaml="${PROJECT_ROOT}/starVLA/config/training/starvla_robotwin_GR00T.yaml"
data_root=/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/StarVLA_YJH/Benchmark/RoboTwin-Clean
data_mix=robotwin_clean
run_root_dir="${PROJECT_ROOT}/results/Checkpoints"
run_id=qwen35_GR00T_RoboTwin_baseline128_local
pretrained_checkpoint=${run_root_dir}/${run_id}/checkpoints/steps_150000_pytorch_model.pt
###########################################################################################

cd "${PROJECT_ROOT}"

output_dir="${run_root_dir}/${run_id}"
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
accelerate launch \
  --config_file "${PROJECT_ROOT}/starVLA/config/deepseeds/deepspeed_zero2.yaml" \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --datasets.vla_data.data_root_dir "${data_root}" \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.freeze_modules '' \
  --trainer.pretrained_checkpoint "${pretrained_checkpoint}" \
  --trainer.is_resume true \
  --trainer.max_train_steps 200000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --trainer.learning_rate.base 2e-5 \
  --trainer.learning_rate.qwen_vl_interface 2e-5 \
  --trainer.learning_rate.action_model 2e-5 \
  --trainer.lr_scheduler_type cosine_with_min_lr \
  --trainer.num_warmup_steps 1000 \
  --run_root_dir "${run_root_dir}" \
  --run_id ${run_id} \
  --wandb_project xintonghu_qwen3.5_GR00T_RoboTwin_SFT \
  --wandb_entity rjucdvfh04
