



# ckpt list


ckpt_list=(
    "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0222_QwenGR00TN1d6_epx3_randominti_multi_robot_4B/checkpoints/steps_100000_pytorch_model.pt"
    "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0222_QwenGR00TN1d6_epx3_randominti_multi_robot_4B/checkpoints/steps_110000_pytorch_model.pt"
    "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0222_QwenGR00TN1d6_epx3_randominti_multi_robot_4B/checkpoints/steps_130000_pytorch_model.pt"
    "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0222_QwenGR00TN1d6_epx3_randominti_multi_robot_4B/checkpoints/steps_140000_pytorch_model.pt"
    "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0222_QwenGR00TN1d6_epx3_randominti_multi_robot_4B/checkpoints/steps_150000_pytorch_model.pt"
    "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0222_QwenGR00TN1d6_epx3_randominti_multi_robot_4B/checkpoints/steps_160000_pytorch_model.pt"
    "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0222_QwenGR00TN1d6_epx3_randominti_multi_robot_4B/checkpoints/steps_170000_pytorch_model.pt"
    "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0222_QwenGR00TN1d6_epx3_randominti_multi_robot_4B/checkpoints/steps_80000_pytorch_model.pt"
    "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0222_QwenGR00TN1d6_epx3_randominti_multi_robot_4B/checkpoints/steps_90000_pytorch_model.pt"
)


for ckpt in "${ckpt_list[@]}"; do
    echo "Evaluating checkpoint: $ckpt"
    bash /mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/examples/Robocasa_tabletop/eval_files/bar/batch_eval_args.sh "$ckpt"
done

