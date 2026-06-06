export PYTHONPATH=$(pwd):${PYTHONPATH}


cd /mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA
if [ -n "$1" ]; then
  ckpt_dir="$1"
else
  ckpt_dir="./results/Checkpoints/1225__bridge_rt_1_1119-qwen3-4b-s3-baselinesft"
fi
ckpt_dir=${ckpt_dir}/checkpoints

port_base=6892
total_gpus=8

# Single checkpoint evaluation function
eval_ckpt() {
    local ckpt_path="$1"
    local port="$2"
    local total_gpus="$total_gpus"
    local gpu_id=$((port % total_gpus))
    echo "[INFO] Evaluating $ckpt_path on port $port (GPU $gpu_id)"
    bash examples/SimplerEnv/eval_files/run_policy_server.sh "$ckpt_path" "$port" "$gpu_id" &
    sleep 10
    bash examples/SimplerEnv/eval_files/start_simpler_env.sh "$ckpt_path" "$port" &
    # Optional: kill %1
}

# Iterate over all checkpoints in the directory and evaluate
eval_all_ckpts_in_dir() {
    local ckpt_dir="$1"
    local port_base=${2:-5680}
    local total_gpus=${3:-8}
    local idx=0
    for ckpt in "$ckpt_dir"/steps*_pytorch_model.pt; do
        if [ -f "$ckpt" ]; then
            local port=$((port_base + idx))
            echo "[INFO] start evaluating $ckpt"
            eval_ckpt "$ckpt" "$port" "$total_gpus"
            ((idx++))
        fi
    done
}



## Run the script to automatically evaluate a specific checkpoint
# ckpt="/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/bridge_rt_1__init/checkpoints/steps_30000_pytorch_model.pt"
# echo "[INFO] start evaluating $ckpt"
# port=5990
# eval_ckpt "$ckpt" "$port"



### Run the script to automatically evaluate all checkpoints in the specified directory
eval_all_ckpts_in_dir "$ckpt_dir" "$port_base" "$total_gpus"


# Wait for all start_simpler_env.py processes to finish
while pgrep -f "start_simpler_env.py" > /dev/null; do
    echo "[INFO] Waiting for start_simpler_env.py processes to finish..."
    sleep 100
done
echo "[INFO] All start_simpler_env.py processes finished."
