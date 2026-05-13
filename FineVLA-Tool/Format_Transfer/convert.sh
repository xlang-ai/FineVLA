#!/usr/bin/env bash
# 问一下GPT，这些路径如何修改
# mode 选择image最好，可以test一下video
# fps最好结合paper和原来的数据check，保证准确，如果没有，那么默认20（如果没有，记得sheet中标注，或者自己做好记录）


set -euo pipefail

RAW_DIR="/home2/xhhuang/towel_fold2"
OUT_DIR="/home2/xhhuang/towel_lerobot"
REPO_ID="local/towel_fold2"

python /home2/xhhuang/convert_mobile_aloha_to_lerobot.py \
  --raw-dir "$RAW_DIR" \
  --out-dir "$OUT_DIR" \
  --repo-id "$REPO_ID" \
  --task "fold towel" \
  --fps 20 \
  --mode image \
  --overwrite 
