#!/bin/bash

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export PYTHONPATH=$(pwd):${PYTHONPATH}
export NO_ALBUMENTATIONS_UPDATE=1

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenOFT
freeze_module_list=${FREEZE_MODULES:-''}
base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct
config_yaml=./examples/Robocasa365/train_files/starvla_train_robocasa365.yaml
robocasa365_data_root=playground/Datasets
data_mix=robocasa365_single
run_root_dir=./results/Checkpoints
run_id=${RUN_ID:-robocasa365_qwenoft}

# Optional overrides for quick validation
num_processes=${NUM_PROCESSES:-8}
max_train_steps=${MAX_TRAIN_STEPS:-100000}
save_interval=${SAVE_INTERVAL:-10000}
logging_frequency=${LOGGING_FREQUENCY:-50}
eval_interval=${EVAL_INTERVAL:-1000}
# === End of environment variable configuration ===
###########################################################################################

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${robocasa365_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps ${max_train_steps} \
  --trainer.save_interval ${save_interval} \
  --trainer.logging_frequency ${logging_frequency} \
  --trainer.eval_interval ${eval_interval} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project StarVLA_RoboCasa365 \
  --wandb_entity jinhuiye
