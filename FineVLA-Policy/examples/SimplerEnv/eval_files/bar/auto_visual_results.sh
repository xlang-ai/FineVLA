
ckpt_root=/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints
ckpt_patten=0222_QwenGR00TN1d6_epx3_randominti_multi_robot_4B

# /mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0222_QwenGR00TN1d6_epx3_bridge_rt_1_XR_FT
for ckpt_dir in ${ckpt_root}/${ckpt_patten}; do
    eval_dir="${ckpt_dir}/checkpoints"
    mkdir -p "${eval_dir}/eval_visuals"
    if [ -d "$eval_dir" ]; then
        echo "[INFO] Visualizing $eval_dir"
        python examples/SimplerEnv/eval_files/bar/visual_eval_results.py --log_dir "$eval_dir"
    else
        echo "[WARN] $eval_dir not found, skip."
    fi
done


