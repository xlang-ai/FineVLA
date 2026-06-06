"""
Checkpoint evaluation results visualization script
=====================

Features:
    - Collects and visualizes success rates of checkpoints at different steps across multiple tasks.
    - Supports multiple evaluation runs for the same step and task, automatically averaging the results.
    - Only counts results from log files containing 'Average success'.
Usage:
    python visual_eval_results.py [log_dir]
    # log_dir is the path to the log directory (e.g., .../output_eval), optional, defaults to the built-in path.
Output:
    - eval_results.png  visualization image
    - eval_results.csv  success rate table for each step, each task, and the average
    (both saved in the grandparent directory of log_dir)

Dependencies:
    numpy, matplotlib, csv

Example:
    python visual_eval_results.py /path/to/output_eval
"""


import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import csv
import sys
import argparse

def visualize_ckpt_results(log_dir=None):
    if log_dir is None:
        log_dir = "/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/bridge_rt_1__init/checkpoints/output_eval"
    # Output paths
    out_dir = os.path.abspath(os.path.join(log_dir, "../.."))
    img_path = os.path.join(out_dir, "eval_visuals/eval_results.png")
    csv_path = os.path.join(out_dir, "eval_visuals/eval_results.csv")
    os.makedirs(os.path.dirname(img_path), exist_ok=True)

    # Match file names
    log_pattern = re.compile(r"steps_(\d+)_pytorch_model_(.+)-v0_run(\d+)\.log")

    # Stats structure: step -> task -> [success_rate, ...]
    results = defaultdict(lambda: defaultdict(list))

    # Parse each log file
    for fname in os.listdir(log_dir):
        m = log_pattern.match(fname)
        if not m:
            continue
        step, task, run_idx = m.groups()
        step = int(step)
        task = task.replace("_", " ")
        avg_success = None
        with open(os.path.join(log_dir, fname), "r") as f:
            for line in f:
                if line.strip().startswith("Average success"):
                    try:
                        avg_success = float(line.strip().split()[-1])
                    except Exception:
                        avg_success = None
        if avg_success is not None:
            results[step][task].append(avg_success)

    # Colors and line styles
    color_map = {
        'PutCarrotOnPlateInScene': 'tab:blue',
        'PutEggplantInBasketScene': 'tab:green',
        'PutSpoonOnTableClothInScene': 'tab:orange',
        'StackGreenCubeOnYellowCubeBakedTexInScene': 'tab:purple',
    }
    linestyles = ['dashed', 'dotted', (0, (5, 1)), (0, (3, 5, 1, 5))]

    # Plot
    plt.figure(figsize=(10, 6))
    all_steps = sorted(results.keys(), key=int)
    task_names = sorted({t for v in results.values() for t in v})

    # Save CSV data
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['step'] + task_names + ['Avg']
        writer.writerow(header)
        avg_y = []
        for idx, step in enumerate(all_steps):
            row = [step]
            vals = []
            for task in task_names:
                v = results[step][task]
                if v:
                    mean_v = np.mean(v)
                    row.append(mean_v)
                    vals.append(mean_v)
                else:
                    row.append('')
            if vals:
                avg_val = np.mean(vals)
            else:
                avg_val = ''
            row.append(avg_val)
            writer.writerow(row)
            avg_y.append(avg_val if avg_val != '' else np.nan)

    for i, task in enumerate(task_names):
        color = color_map.get(task, None)
        linestyle = linestyles[i % len(linestyles)]
        y = []
        for step in all_steps:
            vals = results[step][task]
            if vals:
                y.append(np.mean(vals))
            else:
                y.append(np.nan)
        plt.plot([int(s) for s in all_steps], y, label=task, color=color, linestyle=linestyle, alpha=0.7)

    plt.plot([int(s) for s in all_steps], avg_y, color='red', linewidth=2, label='Avg', alpha=0.9)

    plt.xlabel('Step')
    plt.ylabel('Success Rate')
    plt.title('Success Rate of Different Tasks per Checkpoint')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(img_path)
    print(f"Saved plot to {img_path}")
    print(f"Saved csv to {csv_path}")
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="Checkpoint evaluation results visualization script")
    parser.add_argument('--log_dir', type=str, default=None, help='Path to the log directory (output_eval)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    visualize_ckpt_results(args.log_dir)


