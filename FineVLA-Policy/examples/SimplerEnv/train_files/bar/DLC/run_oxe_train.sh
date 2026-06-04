
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_DISABLE=1
export NCCL_IB_HCA=mlx5

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)


# export WANDB_BASE_URL='http://47.92.143.221:7900/'
# export WANDB_API_KEY='local-1647bcc5479ab0da87b5c153002383e4c8a83867'
export WANDB_MODE=disabled

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenGR00TN1d6_epx3
freeze_module_list=''
base_vlm=./playground/Pretrained_models/Qwen3.5-VL-4B-Instruct
config_yaml=./examples/MultiRobot/train_files/starvla_cotrain_multiRobot_exp3.yaml
data_root=playground/Datasets/OXE_LEROBOT_DATASET
data_mix=bridge_rt_1
run_root_dir=./results/Checkpoints
run_id=0222_${Framework_name}_${data_mix}_XR_FT
pretrained_checkpoint=./results/Checkpoints/0127_QwenGR00TN1d6_epx3_multi_robot/checkpoints/steps_120000_pytorch_model.pt
# === End of environment variable configuration ===
###########################################################################################




# # 初始化 conda（只需执行一次或在新 shell 中执行）
# source /root/miniconda3/etc/profile.d/conda.sh
# # 或：eval "$(conda shell.bash hook)"
# # 激活环境
# conda activate starVLA


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
  --datasets.vla_data.data_root_dir ${data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.pretrained_checkpoint ${pretrained_checkpoint} \
  --trainer.max_train_steps 200000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_xRobot \
  --wandb_entity jinhuiye \
  # --trainer.is_resume True \
  # --is_debug True



##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
