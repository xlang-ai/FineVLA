export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_IB_HCA=mlx5

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)

cd /mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenGR00TN1d6_epx3
freeze_module_list=''
base_vlm=./playground/Pretrained_models/Qwen3.5-VL-4B-Instruct
config_yaml=./examples/MultiRobot/train_files/starvla_cotrain_multiRobot_exp3.yaml
data_root_dir=playground/Datasets/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
data_mix=fourier_gr1_unified_1000
run_root_dir=./results/Checkpoints
run_id=0222_${Framework_name}_${data_mix}_XR_FT
pretrained_checkpoint=./results/Checkpoints/0127_QwenGR00TN1d6_epx3_multi_robot/checkpoints/steps_120000_pytorch_model.pt
# === End of environment variable configuration ===
###########################################################################################

export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.action_model.action_model_type ${DIT_TYPE} \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 24 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 1000 \
  --trainer.learning_rate.base 3e-5 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_robocasa \
  --wandb_entity jinhuiye \
  # --is_debug True


