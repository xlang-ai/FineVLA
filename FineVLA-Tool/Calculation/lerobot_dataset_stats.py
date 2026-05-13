#!/usr/bin/env python3
"""Lerobot v2.1 数据集统计：从 meta 目录读取 info.json 与 episodes.jsonl，汇总 robot_type、trajectory/task 数量、fps、episode 帧数分布，并绘制柱状图。

- 仅读取 meta/ 下文件，不遍历 videos、data。
- 单层数据集：根目录下一级子文件夹；多层数据集：按顶层文件夹聚合所有子数据集的 meta。
- 输出：同一目录下汇总 JSON + 每个数据集一张柱状图 {数据集名称}_episode_frames.png。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 图表仅使用英文，采用默认字体避免乱码
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# 忽略的顶层数据集名（不参与统计）
IGNORE_DATASETS = {"IPEC-COMMUNITY", "RoboCOIN_annotations_backup"}

# robot_type -> 单臂/双臂/灵巧手
ROBOT_TYPE_TO_ARM = {
    "widowx": "单臂",
    "xarm": "单臂",
    "google_robot": "单臂",
    "franka": "单臂",
    "ur": "单臂",
    "panda": "单臂",
    "sawyer": "单臂",
    "r1lite": "单臂",
    "agilex": "单臂",
    "rh20t_custom_7dof": "单臂",
    "discover_robotics_aitbot_mmk2": "单臂",
    "airbot_mmk2": "单臂",
    "a2d": "单臂",
    "bimanual": "双臂",
    "aloha": "双臂",
    "allegro": "灵巧手",
    "dexterous": "灵巧手",
    "leap_hand": "灵巧手",
    "shadow_hand": "灵巧手",
}


def _safe_dataset_name(name: str) -> str:
    """将数据集名称转为可作文件名的安全字符串。"""
    s = re.sub(r"[^\w\-.]", "_", name)
    return s.strip("_") or "dataset"


def _arm_type(robot_type: str) -> str:
    if not robot_type:
        return "unknown"
    r = (robot_type or "").strip().lower()
    return ROBOT_TYPE_TO_ARM.get(r, "unknown")


def scan_all_info_jsons(root: str) -> list[tuple[str, str]]:
    """在 root 下用 glob 发现所有 meta/info.json，返回 (info_json 路径, 相对 dataset_name)。
    跳过路径中包含 videos 或 data 的项。
    """
    root_path = Path(root)
    if not root_path.exists():
        return []
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    # 支持深度 1～4：单层(BC_Z)、二层(Galaxea/xxx)、三层(agibotworld/xxx)、四层(RoboMindV1.0/benchmark/robot/task)
    for pattern in (
        "*/meta/info.json",
        "*/*/meta/info.json",
        "*/*/*/meta/info.json",
        "*/*/*/*/meta/info.json",
    ):
        for fp in sorted(root_path.glob(pattern)):
            fp_str = str(fp)
            try:
                rel = fp.relative_to(root_path)
            except ValueError:
                continue
            # 只跳过“相对路径”中以 videos 或 data 为目录名的层级，避免误伤根路径里的 data
            parts = rel.parts
            if "videos" in parts or "data" in parts:
                continue
            if fp_str in seen:
                continue
            # dataset_name: 单层如 "Bridge"，多层如 "RoboMindV1.0/SubA"
            dataset_name = str(rel.parent.parent)
            top = dataset_name.split(os.sep)[0]
            if top in IGNORE_DATASETS:
                continue
            seen.add(fp_str)
            out.append((fp_str, dataset_name))
    return out


def read_info(path: str) -> dict | None:
    """读取 meta/info.json，返回 dict；失败返回 None。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def count_observation_image_keys(info: dict) -> int:
    """Count keys in info['features'] that start with 'observation.images' (number of camera views)."""
    features = info.get("features") or {}
    return sum(1 for k in features if k.startswith("observation.images"))


def read_episode_lengths(meta_dir: str) -> list[int]:
    """读取 meta 目录下的 episodes.jsonl，提取每行的 length。若不存在或无 length 则返回空列表。"""
    path = Path(meta_dir) / "episodes.jsonl"
    if not path.is_file():
        return []
    lengths = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                L = obj.get("length")
                if L is not None:
                    lengths.append(int(L))
    except Exception:
        pass
    return lengths


def aggregate_by_top_level(items: list[tuple[str, str]]) -> dict[str, list[tuple[str, str]]]:
    """按顶层数据集名聚合：(info_path, dataset_name) -> { top_level: [(info_path, dataset_name), ...] }"""
    by_top: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for info_path, dataset_name in items:
        top = dataset_name.split(os.sep)[0]
        by_top[top].append((info_path, dataset_name))
    return dict(by_top)


def compute_bins(lengths: list[int], num_bins: int = 8) -> tuple[list[tuple[float, float]], list[int], float, float, float]:
    """将 lengths 分成 num_bins 个等长区间，左闭右开，最后一区间闭。
    返回 (bin_edges 列表 [(lo, hi), ...], 每 bin 的 episode 数量, min_val, max_val, avg_val)。
    """
    if not lengths:
        return [], [], 0.0, 0.0, 0.0
    arr = np.array(lengths, dtype=float)
    min_val = float(arr.min())
    max_val = float(arr.max())
    avg_val = float(arr.mean())
    if min_val >= max_val:
        edges = [min_val] + [max_val] * num_bins
        counts = [len(lengths)] + [0] * (num_bins - 1)
        bins_tuples = [(min_val, max_val)] + [(max_val, max_val)] * (num_bins - 1)
        return bins_tuples[:num_bins], counts[:num_bins], min_val, max_val, avg_val
    step = (max_val - min_val) / num_bins
    edges = [min_val + i * step for i in range(num_bins + 1)]
    counts, _ = np.histogram(arr, bins=edges)
    # 最后一区间右端闭
    bins_tuples = [(edges[i], edges[i + 1]) for i in range(num_bins)]
    return bins_tuples, counts.tolist(), min_val, max_val, avg_val


def process_one_dataset(
    top_level: str,
    info_items: list[tuple[str, str]],
    num_bins: int = 8,
) -> dict:
    """聚合一个顶层数据集下所有 meta 的 info 与 episode 帧数，并返回统计 dict。"""
    total_episodes = 0
    total_tasks = 0
    total_frames = 0
    fps_list: list[float] = []
    robot_types: list[str] = []
    num_image_views_list: list[int] = []
    all_lengths: list[int] = []

    for info_path, _ in info_items:
        info = read_info(info_path)
        if not info:
            continue
        total_episodes += int(info.get("total_episodes", 0))
        total_tasks += int(info.get("total_tasks", 0))
        total_frames += int(info.get("total_frames", 0))
        fps = info.get("fps")
        if fps is not None:
            try:
                fps_list.append(float(fps))
            except (TypeError, ValueError):
                pass
        rt = info.get("robot_type")
        if rt:
            robot_types.append(rt)
        n_views = count_observation_image_keys(info)
        num_image_views_list.append(n_views)
        meta_dir = str(Path(info_path).parent)
        lengths = read_episode_lengths(meta_dir)
        all_lengths.extend(lengths)

    # 若没有 episodes.jsonl，用 total_frames / total_episodes 作为平均帧数，无分布
    if not all_lengths and total_episodes and total_frames:
        avg_frames = total_frames / total_episodes
    elif all_lengths:
        avg_frames = sum(all_lengths) / len(all_lengths)
    else:
        avg_frames = 0.0

    bins_tuples, bin_counts, min_frames, max_frames, avg_frames_computed = compute_bins(
        all_lengths, num_bins=num_bins
    )
    if all_lengths and avg_frames == 0:
        avg_frames = avg_frames_computed

    # 臂型：取出现最多的 robot_type 再映射
    robot_type_primary = robot_types[0] if robot_types else ""
    arm_type = _arm_type(robot_type_primary)
    fps_primary = fps_list[0] if fps_list else None
    if len(set(fps_list)) > 1:
        fps_display = fps_list
    else:
        fps_display = fps_primary
    # 视角数：observation.images 开头字段数，多子集取最大值
    num_image_views = max(num_image_views_list) if num_image_views_list else 0

    return {
        "dataset_name": top_level,
        "robot_type": robot_type_primary or "unknown",
        "arm_type": arm_type,
        "num_image_views": num_image_views,
        "trajectory_count": total_episodes,
        "task_count": total_tasks,
        "fps": fps_display,
        "episode_avg_frames": round(avg_frames, 2),
        "episode_min_frames": int(min_frames) if all_lengths else None,
        "episode_max_frames": int(max_frames) if all_lengths else None,
        "frame_bin_edges": [[round(b[0], 2), round(b[1], 2)] for b in bins_tuples],
        "frame_bin_counts": bin_counts,
        "sub_meta_count": len(info_items),
        "has_episode_lengths": len(all_lengths) > 0,
    }


def plot_bar_chart(
    bin_edges: list[tuple[float, float]],
    bin_counts: list[int],
    dataset_name: str,
    output_path: str,
) -> None:
    """Draw bar chart for episode frame distribution (English only, no CJK)."""
    if not bin_edges or not bin_counts:
        return
    # Use ASCII-safe title to avoid font/garbled issues
    title_name = _safe_dataset_name(dataset_name).replace("_", " ")
    labels = [f"[{b[0]:.0f}, {b[1]:.0f})" for b in bin_edges]
    if len(labels) > 0:
        labels[-1] = f"[{bin_edges[-1][0]:.0f}, {bin_edges[-1][1]:.0f}]"
    x = np.arange(len(bin_counts))
    width = 0.7
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, bin_counts, width, color="steelblue", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Episode count")
    ax.set_xlabel("Frame range")
    ax.set_title(f"Episode frame distribution - {title_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="统计 Lerobot v2.1 数据集：robot_type、trajectory/task 数、fps、episode 帧数分布与柱状图",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21",
        help="Lerobot 数据根目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="统计结果与图表的输出目录，默认与脚本同目录",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="汇总 JSON 文件名，默认 output_dir/lerobot_dataset_stats.json",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=8,
        help="帧数区间数量",
    )
    args = parser.parse_args()

    root = args.root
    output_dir = Path(args.output_dir or os.path.dirname(os.path.abspath(__file__)))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_json or str(output_dir / "lerobot_dataset_stats.json")

    print("扫描 meta/info.json ...")
    items = scan_all_info_jsons(root)
    if not items:
        print("未发现任何 meta/info.json，请检查 --root 路径。")
        sys.exit(1)
    print(f"  发现 {len(items)} 个 meta/info.json")

    by_top = aggregate_by_top_level(items)
    print(f"  聚合为 {len(by_top)} 个顶层数据集\n")

    results: list[dict] = []
    for top_level in sorted(by_top.keys()):
        info_list = by_top[top_level]
        row = process_one_dataset(top_level, info_list, num_bins=args.bins)
        results.append(row)
        if row["has_episode_lengths"] and row["frame_bin_counts"]:
            safe_name = _safe_dataset_name(top_level)
            chart_path = output_dir / f"{safe_name}_episode_frames.png"
            plot_bar_chart(
                [(b[0], b[1]) for b in row["frame_bin_edges"]],
                row["frame_bin_counts"],
                top_level,
                str(chart_path),
            )
            print(f"  图表: {chart_path}")

    summary = {
        "root": root,
        "total_datasets": len(results),
        "datasets": results,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n汇总已写入: {output_json}")
    print("\n统计摘要:")
    print("-" * 80)
    for r in results:
        print(
            f"  {r['dataset_name']}: robot_type={r['robot_type']}, arm={r['arm_type']}, "
            f"views={r['num_image_views']}, trajectories={r['trajectory_count']}, tasks={r['task_count']}, "
            f"fps={r['fps']}, avg_frames={r['episode_avg_frames']}"
        )


if __name__ == "__main__":
    main()
