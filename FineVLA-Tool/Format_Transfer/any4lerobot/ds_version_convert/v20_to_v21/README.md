# LeRobot Dataset v20 to v21

## Get started

1. Install v2.1 lerobot
    ```bash
    git clone https://github.com/huggingface/lerobot.git
    git checkout -f d602e8169cbad9e93a4a3b3ee1dd8b332af7ebf8 # 必须 加上 -f 进行强制性 checkout 避免 提交之间的冲突
    pip install -e .
    ```

2. Run the converter:
    ```bash
    python convert_dataset_v20_to_v21.py \
        --repo-id=your_id \
        --root=your_local_dir \
        --delete-old-stats \
        --push-to-hub \
        --num-workers=8
    ```


        python convert_dataset_v20_to_v21.py \
        --repo-id=your_id \
        --root=your_local_dir \
        --delete-old-stats \
        --push-to-hub \
        --num-workers=8



        # 1. 复制到新目录
cp -r /cpfs04/shared/Group-m6/tongzai.hxt/Qwen_VLA/LZX_VLA_Data/DROID_LEROBOT_DATASET \
      /cpfs04/shared/Group-m6/tongzai.hxt/Qwen_VLA/DROID_LEROBOT_V21

# 2. 对副本进行转换
cd /cpfs04/shared/Group-m6/tongzai.hxt/Qwen_VLA/any4lerobot/ds_version_convert/v20_to_v21

python convert_dataset_v20_to_v21.py \
    --repo-id=1.0.0 \
    --root=/cpfs04/shared/Group-m6/tongzai.hxt/Qwen_VLA/DROID_LEROBOT_V21 \
    --delete-old-stats \
    --num-workers=16