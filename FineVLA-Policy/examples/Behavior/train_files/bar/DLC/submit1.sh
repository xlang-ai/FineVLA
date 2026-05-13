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

cd /mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenOFT
base_vlm=./playground/Pretrained_models/Qwen3-VL-4B-Instruct
freeze_module_list='' # just for fast debug, sota is under fully FT, i.g., freeze_module_list=""
DIT_TYPE="DiT-B"
config_yaml=./examples/Behavior/train_files/starvla_cotrain_behavior.yaml
data_root_dir=./playground/Datasets/behavior-1k
data_mix=BEHAVIOR_challenge
run_root_dir=./playground/Checkpoints
run_id=0109_starvla_${Framework_name}
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
  --framework.action_model.action_model_type ${DIT_TYPE} \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 4 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 500000 \
  --trainer.save_interval 20000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100 \
  --trainer.learning_rate.base 3e-5 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_behavior \
  --wandb_entity jinhuiye \


