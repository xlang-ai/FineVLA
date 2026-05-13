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

# cd /mnt/cpfs_m6_29eu38p1/data/shared/public/liuye/transformers-internal/
# git switch starvla-4-57 && pip install -e .

# cd -
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
Framework_name=QwenOFT_multiRobo
freeze_module_list=''
base_vlm=./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
config_yaml=./examples/MultiRobot/train_files/starvla_cotrain_multiRobot_exp3.yaml
data_root=playground/Datasets
data_mix=multi_robot
run_root_dir=./results/Checkpoints
run_id=0224_${Framework_name}_${data_mix}_256
# === End of environment variable configuration ===
###########################################################################################

export WANDB_BASE_URL='http://47.92.143.221:7900/'
export WANDB_API_KEY='local-1647bcc5479ab0da87b5c153002383e4c8a83867'
# export WANDB_MODE=disabled

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
  --datasets.vla_data.data_root_dir ${data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 250000 \
  --trainer.num_warmup_steps 10000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 2000000 \
  --trainer.learning_rate.base 2e-5 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_xRobo \
  --wandb_entity rjucdvfh04 \
  # --is_debug True


