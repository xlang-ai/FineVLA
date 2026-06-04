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
export PATH="/usr/bin:$PATH" 
alias accelerate="/usr/local/bin/accelerate"


# ls -la /usr/bin
which python

# ls -la /usr/bin/python*

# setup distributed training args for 2 nodes
GPUS_PER_NODE=8

# DLC environment variables - these will be automatically set by DLC platform
# NODE_ID=$RANK
# MASTER_ADDR=$MASTER_ADDR
# MASTER_PORT=$MASTER_PORT
# TOTAL_GPUS=$(($GPUS_PER_NODE*$WORLD_SIZE))

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
Framework_name=QwenPI
base_vlm=./playground/Pretrained_models/Qwen3.5-VL-4B-Instruct
freeze_module_list=''
config_yaml=./starVLA/config/training/starvla_cotrain_libero.yaml
libero_data_root=playground/Datasets/OXE_LEROBOT_DATASET
data_mix=libero_all_with_history
run_root_dir=./results/Checkpoints
run_id=1122_libero4in1_qwen3oft_5history
pretrained_checkpoint=results/Checkpoints/1117_libero4in1_qwenpi_history/checkpoints/steps_20000_pytorch_model.pt
# === End of environment variable configuration ===
###########################################################################################


export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/




accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes=${GPUS_PER_NODE} \
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


