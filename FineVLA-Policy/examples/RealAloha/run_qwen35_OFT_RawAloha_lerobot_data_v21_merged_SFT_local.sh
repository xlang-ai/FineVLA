#!/bin/bash
set -eo pipefail

ulimit -n 1048576
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# WANDB config
export WANDB_BASE_URL="http://47.92.143.221:7900/"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_DISABLE_CODE=true
export WANDB_CONSOLE="off"
export WANDB_MODE=online

# ===== Reproducibility Settings =====
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# ===== End of Reproducibility Settings =====

source /root/miniconda3/etc/profile.d/conda.sh
conda activate starVLA

echo "RUNNING: QwenOFT SFT from RawAloha_OFT on lerobot_data_v21_merged"

PROJECT_ROOT=/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/StarVLA_YJH
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

###########################################################################################
Framework_name=QwenOFT
freeze_module_list=''
base_vlm=/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/Pretrained_models/Qwen3.5-4B
config_yaml="${PROJECT_ROOT}/starVLA/config/training/starvla_aloha_multi_OFT.yaml"
data_root=/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/xuhong.hxh
data_mix=lerobot_data_v21_merged_single
run_root_dir="${PROJECT_ROOT}/results/Checkpoints/OFT_ALOHA"
run_id=RawAloha_OFT_lerobot_data_v21_merged_SFT_local
pretrained_checkpoint="${PROJECT_ROOT}/results/Checkpoints/OFT_ALOHA/RawAloha_OFT/checkpoints/steps_200000_pytorch_model.pt"
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
  --trainer.pretrained_checkpoint "${pretrained_checkpoint}" \
  --trainer.max_train_steps 50000 \
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
  --wandb_project qwen35_OFT_aloha_multi_SFT \
  --wandb_entity rjucdvfh04
