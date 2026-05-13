"""
ckpt评测结果可视化脚本
=====================

功能：
    - 统计并可视化不同step的ckpt在多个task下的评测成功率。
    - 支持同一step同一task多次评测，自动取均值。
    - 只统计log文件中包含 'Average success' 的结果。
用法：
    python visual_eval_results.py [log_dir]
    # log_dir 为日志文件夹路径（如 .../output_eval），可选，默认用脚本内置路径。
输出：
    - eval_results.png  可视化图片
    - eval_results.csv  每个step每个task及均值的成功率表格
    （均保存在log_dir的上上级目录）

依赖：
    numpy, matplotlib, csv

示例：
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
    # 输出路径
    out_dir = os.path.abspath(os.path.join(log_dir, "../"))
    img_path = os.path.join(out_dir, "eval_visuals/eval_results.png")
    csv_path = os.path.join(out_dir, "eval_visuals/eval_results.csv")
    os.makedirs(os.path.dirname(img_path), exist_ok=True)

    # 匹配文件名
    log_pattern = re.compile(r"steps_(\d+)_pytorch_model_infer_(.+)-v0\.log\.run(\d+)")

    # 统计结构: step -> task -> [success_rate, ...]
    results = defaultdict(lambda: defaultdict(list))

    # 解析每个log文件
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

    # 颜色和线型
    color_map = {
        'PutCarrotOnPlateInScene': 'tab:blue',
        'PutEggplantInBasketScene': 'tab:green',
        'PutSpoonOnTableClothInScene': 'tab:orange',
        'StackGreenCubeOnYellowCubeBakedTexInScene': 'tab:purple',
    }
    linestyles = ['dashed', 'dotted', (0, (5, 1)), (0, (3, 5, 1, 5))]

    # 画图
    plt.figure(figsize=(10, 6))
    all_steps = sorted(results.keys(), key=int)
    task_names = sorted({t for v in results.values() for t in v})

    # 保存csv数据
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
    parser = argparse.ArgumentParser(description="ckpt评测结果可视化脚本")
    parser.add_argument('--log_dir', type=str, default=None, help='日志文件夹路径（output_eval）')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    visualize_ckpt_results(args.log_dir)


