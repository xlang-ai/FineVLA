#!/bin/bash

###########################################################################################
# === Please modify the following paths according to your environment ===
export ROBOCASA_Python=/root/miniconda3/envs/robocasa/bin/python
export PYTHONPATH=$(pwd):${PYTHONPATH}

host="127.0.0.1"
port=5678
env_name="robocasa/PickPlaceCounterToCabinet"
your_ckpt=./results/Checkpoints/robocasa365_qwenoft/checkpoints/steps_10000_pytorch_model.pt
unnorm_key="new_embodiment"
# === End of environment variable configuration ===
###########################################################################################

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
video_out_path="results/robocasa365/${folder_name}"

${ROBOCASA_Python} ./examples/Robocasa365/eval_files/simulation_env.py \
    --args.pretrained-path ${your_ckpt} \
    --args.unnorm-key ${unnorm_key} \
    --args.host ${host} \
    --args.port ${port} \
    --args.env-name ${env_name} \
    --args.n-episodes 50 \
    --args.n-envs 1 \
    --args.max-episode-steps 600 \
    --args.n-action-steps 8 \
    --args.video-out-path ${video_out_path}
