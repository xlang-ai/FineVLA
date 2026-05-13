

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_IB_HCA=mlx5

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)
cd /root/Jinhui/Projects/starVLA

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenPI
base_vlm=./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct
freeze_module_list=''
config_yaml=./starVLA/config/training/starvla_cotrain_libero.yaml
libero_data_root=playground/Datasets/OXE_LEROBOT_DATASET
data_mix=libero_all_with_history
run_root_dir=./results/Checkpoints
run_id=1118_libero4in1_qwenpi_4history
pretrained_checkpoint=./results/Checkpoints/1117_libero4in1_qwenpi_history/checkpoints/steps_20000_pytorch_model.pt
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
  --datasets.vla_data.data_root_dir ${libero_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 4 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --trainer.learning_rate.base 2e-5 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA \
  --wandb_entity jinhuiye \
  # --is_debug True


