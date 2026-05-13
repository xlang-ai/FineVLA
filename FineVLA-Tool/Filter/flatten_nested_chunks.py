"""将 droid_RoboInter 的三层嵌套 chunk 目录扁平化。

data 结构:   data/chunk-NNN/chunk-NNN/chunk-NNN/episode_*.parquet
videos 结构: videos/chunk-NNN/chunk-NNN/chunk-NNN/observation.images.*/episode_*.mp4

目标: 去掉中间两层，变成 chunk-NNN/ 直接包含内容。

用法:
    python flatten_nested_chunks.py
"""

import os
import shutil
from tqdm import tqdm

ROOT = "/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21/RH20T-RoboInter"
DIRS = ["data", "videos"]


def flatten_one_dir(base_dir):
    if not os.path.isdir(base_dir):
        print(f"  [跳过] {base_dir} 不存在")
        return

    chunks = sorted([d for d in os.listdir(base_dir)
                     if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("chunk-")])
    print(f"  共 {len(chunks)} 个 chunk")

    done = 0
    cleaned = 0
    skipped = 0

    for c in tqdm(chunks, desc=f"  {os.path.basename(base_dir)}"):
        chunk_dir = os.path.join(base_dir, c)
        mid_dir = os.path.join(chunk_dir, c)
        inner_dir = os.path.join(mid_dir, c)

        if not os.path.isdir(mid_dir):
            skipped += 1
            continue

        if os.path.isdir(inner_dir):
            try:
                inner_has_files = len(os.listdir(inner_dir)) > 0
            except:
                inner_has_files = False

            if inner_has_files:
                tmp_dir = os.path.join(base_dir, f"_tmp_{c}")
                os.rename(inner_dir, tmp_dir)
                try:
                    os.rmdir(mid_dir)
                except OSError:
                    pass
                for f in os.listdir(tmp_dir):
                    src = os.path.join(tmp_dir, f)
                    dst = os.path.join(chunk_dir, f)
                    if not os.path.exists(dst):
                        os.rename(src, dst)
                shutil.rmtree(tmp_dir, ignore_errors=True)
                done += 1
            else:
                os.rmdir(inner_dir)
                os.rmdir(mid_dir)
                cleaned += 1
        else:
            try:
                os.rmdir(mid_dir)
                cleaned += 1
            except OSError:
                pass

    print(f"  结果: 移动 {done}, 清理空壳 {cleaned}, 已扁平 {skipped}")


for d in DIRS:
    base = os.path.join(ROOT, d)
    print(f"\n=== 处理 {d}/ ===")
    flatten_one_dir(base)

print("\n全部完成")
