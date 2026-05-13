



# ckpt list


ckpt_list=(
    "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0222_QwenGR00TN1d6_epx3_fourier_gr1_unified_1000_XR_FT/checkpoints/steps_10000_pytorch_model.pt"
    "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0222_QwenGR00TN1d6_epx3_fourier_gr1_unified_1000_XR_FT/checkpoints/steps_20000_pytorch_model.pt"
    "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0222_QwenGR00TN1d6_epx3_fourier_gr1_unified_1000_XR_FT/checkpoints/steps_30000_pytorch_model.pt"
    "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0222_QwenGR00TN1d6_epx3_fourier_gr1_unified_1000_XR_FT/checkpoints/steps_40000_pytorch_model.pt"
    "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0222_QwenGR00TN1d6_epx3_fourier_gr1_unified_1000_XR_FT/checkpoints/steps_50000_pytorch_model.pt"
    "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0222_QwenGR00TN1d6_epx3_fourier_gr1_unified_1000_XR_FT/checkpoints/steps_60000_pytorch_model.pt"
    "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0222_QwenGR00TN1d6_epx3_fourier_gr1_unified_1000_XR_FT/checkpoints/steps_70000_pytorch_model.pt"
    "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0222_QwenGR00TN1d6_epx3_fourier_gr1_unified_1000_XR_FT/checkpoints/steps_80000_pytorch_model.pt"
    "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0222_QwenGR00TN1d6_epx3_fourier_gr1_unified_1000_XR_FT/checkpoints/steps_90000_pytorch_model.pt"
)


for ckpt in "${ckpt_list[@]}"; do
    echo "Evaluating checkpoint: $ckpt"
    bash /mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/examples/Robocasa_tabletop/eval_files/bar/batch_eval_args.sh "$ckpt"
done

