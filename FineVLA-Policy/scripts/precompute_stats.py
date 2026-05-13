#!/usr/bin/env python3
"""Pre-compute stats_gr00t.json for all datasets in aloha_multi_mix.
Run this once before training to avoid 'Too many open files' during multi-GPU startup.
Uses multiprocessing for parallel computation (CPU only, no GPU needed)."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from multiprocessing import Pool, cpu_count

DATA_ROOT = "/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt"
MIX_NAME = "aloha_multi_mix"
NUM_WORKERS = 4  # 减少 worker 避免 Too many open files


def compute_one(args):
    idx, total, d_name, robot_type = args
    stats_path = Path(DATA_ROOT) / d_name / "meta" / "stats_gr00t.json"
    if stats_path.exists():
        return f"[{idx+1}/{total}] SKIP (cached): {d_name}"
    try:
        from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
        data_cfg = {"video_backend": "pyav", "lerobot_version": "v2.0"}
        ds = make_LeRobotSingleDataset(Path(DATA_ROOT), d_name, robot_type, data_cfg=data_cfg)
        return f"[{idx+1}/{total}] OK ({len(ds)} steps): {d_name}"
    except Exception as e:
        return f"[{idx+1}/{total}] ERROR: {d_name} -> {e}"


if __name__ == "__main__":
    from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
    mixture = DATASET_NAMED_MIXTURES[MIX_NAME]
    total = len(mixture)

    tasks = [(i, total, d_name, robot_type) for i, (d_name, _, robot_type) in enumerate(mixture)]

    # Count how many need computing
    need = sum(1 for _, _, d_name, _ in tasks if not (Path(DATA_ROOT) / d_name / "meta" / "stats_gr00t.json").exists())
    print(f"Total: {total} datasets, need computing: {need}, cached: {total - need}")
    print(f"Using {NUM_WORKERS} workers")

    with Pool(NUM_WORKERS) as pool:
        for result in pool.imap_unordered(compute_one, tasks):
            print(result, flush=True)

    print("All done!")
