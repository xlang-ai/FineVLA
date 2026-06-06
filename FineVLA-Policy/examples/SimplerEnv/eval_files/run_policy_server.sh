export PYTHONPATH=$(pwd):${PYTHONPATH}
star_vla_python=/root/miniconda3/envs/starVLA/bin/python

##### get parameters #####
# Check if an argument was provided; otherwise use the default value
if [ -n "$1" ]; then
  MODEL_PATH="$1"
else
  MODEL_PATH=results/Checkpoints/bridge_rt_1__init/checkpoints/steps_40000_pytorch_model.pt
fi

port=${2:-5678}
gpu_id=${3:-0}

##### get parameters #####

#### build output directory #####
# Get the checkpoint directory and create the output_server directory
ckpt_dir=$(dirname "${MODEL_PATH}")
ckpt_base=$(basename "${MODEL_PATH}")
ckpt_name="${ckpt_base%.*}"
output_server_dir="${ckpt_dir}/output_server"
mkdir -p "${output_server_dir}"
log_file="${output_server_dir}/${ckpt_name}_port${port}.log"

#### build output directory #####


cd /mnt/petrelfs/yejinhui/Projects/starVLA
export PYTHONPATH=$(pwd):${PYTHONPATH}

port=6678
gpu_id=2
# export DEBUG=true
export star_vla_python=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/starVLA/bin/python

your_ckpt=./results/Checkpoints/1208_bridge_rt_1_Qwen3PI/final_model/pytorch_model.pt

#### build output directory #####
ckpt_dir=$(dirname "${your_ckpt}")
ckpt_base=$(basename "${your_ckpt}")
ckpt_name="${ckpt_base%.*}"
output_server_dir="${ckpt_dir}/output_server"
mkdir -p "${output_server_dir}"
log_file="${output_server_dir}/${ckpt_name}_policy_server_${port}.log"


#### run server #####
CUDA_VISIBLE_DEVICES=${gpu_id} ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16 \
    2>&1 | tee "${log_file}"