#!/usr/bin/env python3
"""Check which datasets in aloha_multi_mix have complete cache files."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES

DATA_ROOT = Path("/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt")
MIX_NAME = "aloha_multi_mix"
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache_check_report.txt")

mix = DATASET_NAMED_MIXTURES[MIX_NAME]
total = len(mix)

missing_stats = []
missing_steps = []
ok_count = 0

for d_name, _, robot_type in mix:
    meta = DATA_ROOT / d_name / "meta"
    has_stats = (meta / "stats_gr00t.json").exists()
    has_steps = (meta / "steps_data_index.pkl").exists()
    if not has_stats:
        missing_stats.append(d_name)
    if not has_steps:
        missing_steps.append(d_name)
    if has_stats and has_steps:
        ok_count += 1

lines = []
lines.append(f"=== Cache Status Report for {MIX_NAME} ===")
lines.append(f"Total datasets: {total}")
lines.append(f"Fully cached (both files): {ok_count}")
lines.append(f"Missing stats_gr00t.json: {len(missing_stats)}")
lines.append(f"Missing steps_data_index.pkl: {len(missing_steps)}")
lines.append("")

if missing_stats:
    lines.append("--- Datasets without stats_gr00t.json ---")
    for d in missing_stats:
        lines.append(f"  {d}")
    lines.append("")

if missing_steps:
    lines.append("--- Datasets without steps_data_index.pkl ---")
    for d in missing_steps:
        lines.append(f"  {d}")
    lines.append("")

if ok_count == total:
    lines.append("ALL DATASETS FULLY CACHED - READY TO TRAIN!")
else:
    lines.append(f"WARNING: {total - ok_count} dataset(s) not fully cached.")

report = "\n".join(lines)
print(report)

with open(OUTPUT_FILE, "w") as f:
    f.write(report + "\n")
print(f"\nReport saved to: {OUTPUT_FILE}")
