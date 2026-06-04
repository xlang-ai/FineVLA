set -eo pipefail

# NCCL configuration for DLC multi-node training - optimized for performance
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=220
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_MIN_NCHANNELS=4
export NCCL_NET_PLUGIN=none
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USER=whoami
export PRODUCT=1
# export PATH="/usr/bin:$PATH" 


# ls -la /usr/bin
# which python

# ls -la /usr/bin/python*

# setup distributed training args for 2 nodes
GPUS_PER_NODE=8

# DLC environment variables - these will be automatically set by DLC platform
NODE_ID=$RANK
MASTER_ADDR=$MASTER_ADDR
MASTER_PORT=$MASTER_PORT
TOTAL_GPUS=$(($GPUS_PER_NODE*$WORLD_SIZE))

echo "Node ID: $NODE_ID"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "World Size: $WORLD_SIZE"
echo "Total GPUs: $TOTAL_GPUS"

subfix=`date "+%H-%M"`

echo "RUNNING:"

# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate starVLA

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenGR00TN1d6_epx3
freeze_module_list=''
base_vlm=./playground/Pretrained_models/Qwen3.5-VL-4B-Instruct
config_yaml=./examples/MultiRobot/train_files/starvla_cotrain_multiRobot_exp3.yaml
data_root_dir=playground/Datasets/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
data_mix=fourier_gr1_unified_1000
run_root_dir=./results/Checkpoints
run_id=0222_${Framework_name}_${data_mix}_XR_FT_debug_256
# pretrained_checkpoint=./results/Checkpoints/0127_QwenGR00TN1d6_epx3_multi_robot/checkpoints/steps_120000_pytorch_model.pt
# === End of environment variable configuration ===
###########################################################################################


export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --machine_rank $RANK \
  --num_machines $WORLD_SIZE \
  --num_processes=${TOTAL_GPUS} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.4 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 2000 \
  --trainer.learning_rate.base 2e-5 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Simpler \
  --wandb_entity jinhuiye \
  # --is_debug True


