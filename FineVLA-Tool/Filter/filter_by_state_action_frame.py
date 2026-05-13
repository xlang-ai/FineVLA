# -*- coding: utf-8 -*-
"""
综合筛选脚本：对单个数据集（扁平模式）或含子数据集的数据集（子数据集模式）执行三项检查。

三项检查：
  1. 帧数过少 —— episode 帧数 < config.DATASET_MIN_FRAME[dataset] 阈值
  2. task 为空 —— episode 的 tasks 字段为空列表或不存在
  3. L2 异常 —— state 与 action 轨迹的 range-normalized per-frame L2 距离超过阈值

模式自动检测：
  - 扁平模式：<path>/meta/episodes.jsonl 存在 → 单数据集处理
  - 子数据集模式：否则递归查找所有含 meta/episodes.jsonl 的叶子目录，共享 modality.json

用法：
    python filter_by_state_action_frame.py <dataset_path> [options]

示例：
    # 扁平模式
    python filter_by_state_action_frame.py /path/to/Lerobot_v21/BC_Z --force-reconvert

    # 子数据集模式（自动检测）
    python filter_by_state_action_frame.py /path/to/Galaxea-Open-World-Dataset --episodes 3

    # 子数据集模式 - 按机器人类型
    python filter_by_state_action_frame.py /path/to/RoboMindV2.0/franka --episodes 5

    python filter_by_state_action_frame.py \
      /mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21/RH20T-RoboInter \
      --episodes 10--force-reconvert
"""

import argparse
import json
import os
import sys
import subprocess
import glob as glob_mod
import time
import multiprocessing as mp
from functools import partial

import numpy as np
import pyarrow.parquet as pq

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import config
from cal_distance import compute_vla_l2_score, compute_episode_similarity

GRIPPER_SLOTS = {"left_gripper", "right_gripper"}


# ============================================================
# 帧数 / task 筛选
# ============================================================

def _get_min_frame(dataset_key: str):
    """根据数据集键获取 min_frame：先精确匹配，再按顶层目录名匹配。"""
    if dataset_key in config.DATASET_MIN_FRAME:
        return config.DATASET_MIN_FRAME[dataset_key]
    top_level = dataset_key.split("/")[0]
    return config.DATASET_MIN_FRAME.get(top_level)


def _tasks_empty(tasks) -> bool:
    """判断 tasks 是否为空：None、空列表、或列表中全部为空字符串。"""
    if tasks is None:
        return True
    if isinstance(tasks, list):
        if len(tasks) == 0:
            return True
        if all(isinstance(t, str) and t.strip() == "" for t in tasks):
            return True
    return False


def _find_episode_jsonl(dataset_path: str) -> str | None:
    """在 dataset_path/meta/ 下查找 episodes.jsonl 或 episode.jsonl。"""
    meta_dir = os.path.join(dataset_path, "meta")
    if not os.path.isdir(meta_dir):
        return None
    for name in config.EPISODE_FILENAMES:
        fp = os.path.join(meta_dir, name)
        if os.path.isfile(fp):
            return fp
    return None


def _load_tasks_jsonl(dataset_path: str) -> dict:
    """加载 tasks.jsonl，返回 {task_index: task_string} 映射。"""
    tasks_file = os.path.join(dataset_path, "meta", "tasks.jsonl")
    task_map = {}
    if not os.path.isfile(tasks_file):
        return task_map
    with open(tasks_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = obj.get("task_index")
            task = obj.get("task", "")
            if idx is not None:
                task_map[idx] = task
    return task_map


def _resolve_episode_tasks(obj: dict, task_map: dict):
    """从 episode 对象解析 task 列表。
    优先使用内联的 "tasks" 字段；若不存在则通过 "task_index" 查找 tasks.jsonl。
    """
    tasks = obj.get("tasks")
    if tasks is not None:
        return tasks
    # 通过 task_index 查找
    task_index = obj.get("task_index")
    if task_index is not None and task_map:
        task_text = task_map.get(task_index)
        if task_text is not None:
            return [task_text]
    return None


def check_frame_and_task(dataset_path: str, dataset_key: str, episodes: int = None, quiet: bool = False) -> dict:
    """检查帧数和 task，返回 {ep_id: [reason, ...]}。

    Parameters
    ----------
    episodes : int, optional
        只检查前 N 个 episode。None 表示检查全部。
    """
    result = {}
    min_frame = _get_min_frame(dataset_key)

    ep_file = _find_episode_jsonl(dataset_path)
    if ep_file is None:
        if not quiet:
            print(f"[WARN] 未找到 episodes.jsonl，跳过帧数/task 检查")
        return result

    # 加载 tasks.jsonl 用于 task_index 查找
    task_map = _load_tasks_jsonl(dataset_path)

    if not quiet:
        print(f"帧数/task 检查: {ep_file}")
        print(f"  min_frame 阈值: {min_frame if min_frame is not None else '未设置（跳过帧数检查）'}")
        if episodes is not None:
            print(f"  限制检查前 {episodes} 个 episode")
        if task_map:
            print(f"  已加载 tasks.jsonl（{len(task_map)} 条 task）")

    n_frame_bad = 0
    n_task_bad = 0
    n_total = 0

    with open(ep_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 如果指定了 episodes 限制，只处理前 N 个
            if episodes is not None and n_total >= episodes:
                break

            n_total += 1
            episode_index = obj.get("episode_index")
            length = obj.get("length", 0)
            tasks = _resolve_episode_tasks(obj, task_map)
            ep_id = str(episode_index)

            reasons = []
            if min_frame is not None and length < min_frame:
                reasons.append(f"frame数为{length},小于阈值{min_frame}")
                n_frame_bad += 1
            if _tasks_empty(tasks):
                reasons.append("task为空")
                n_task_bad += 1

            if reasons:
                result[ep_id] = reasons

    if not quiet:
        print(f"  共 {n_total} 个 episode，帧数不足: {n_frame_bad}，task为空: {n_task_bad}")
    return result


# ============================================================
# parquet 文件完整性检查
# ============================================================

def check_corrupted_parquets(dataset_path: str, episodes: int = None, quiet: bool = False) -> dict:
    """扫描 data/ 下 episode parquet 文件，检测损坏文件（0 字节、无法读取 schema 等）。
    episodes: 只检查前 N 个文件。None 表示全部。
    返回 {ep_id: [reason, ...]}。
    """
    import pyarrow.parquet as pq_check

    data_dir = os.path.join(dataset_path, "data")
    if not os.path.isdir(data_dir):
        if not quiet:
            print(f"[WARN] 未找到 data/ 目录，跳过 parquet 完整性检查")
        return {}

    parquet_files = sorted(glob_mod.glob(os.path.join(data_dir, "**", "episode_*.parquet"),
                                         recursive=True))
    total = len(parquet_files)
    if episodes is not None and episodes < total:
        parquet_files = parquet_files[:episodes]
    if not quiet:
        print(f"parquet 完整性检查: 检查 {len(parquet_files)}/{total} 个 parquet 文件")

    result = {}
    n_corrupted = 0
    for pf in parquet_files:
        ep_name = os.path.basename(pf)
        # 从文件名提取 episode id: episode_000118.parquet -> 118
        ep_id = ep_name.replace("episode_", "").replace(".parquet", "").lstrip("0") or "0"

        reason = None
        try:
            fsize = os.path.getsize(pf)
        except OSError as e:
            reason = f"parquet文件无法访问: {e}"
        else:
            if fsize == 0:
                reason = "parquet文件损坏(0字节)"
            else:
                try:
                    pq_check.read_schema(pf)
                except Exception as e:
                    reason = f"parquet文件损坏(schema无法读取): {e}"

        if reason:
            result[ep_id] = [reason]
            n_corrupted += 1

    if not quiet:
        print(f"  损坏文件: {n_corrupted}")
    return result


# ============================================================
# L2 state-action 筛选
# ============================================================

def _resolve_dataset_key(dataset_path: str) -> str:
    norm = os.path.normpath(dataset_path)
    data_root = os.path.normpath(config.DATA_ROOT)
    if norm.startswith(data_root):
        rel = os.path.relpath(norm, data_root)
    else:
        rel = os.path.basename(norm)
    if rel in config.STATE_ACTION_COMPARE_SLOTS:
        return rel
    top = rel.split(os.sep)[0]
    if top in config.STATE_ACTION_COMPARE_SLOTS:
        return top
    return rel


def _normalize_slot_list(slot) -> list[str]:
    """将 slot 配置统一为列表形式。支持字符串或列表。"""
    if isinstance(slot, str):
        return [slot]
    if isinstance(slot, list):
        return slot
    raise ValueError(f"slot 配置必须是字符串或列表，但得到: {type(slot)}")


def _get_compare_slots(dataset_key: str):
    """返回 (state_slots, action_slots)，均为列表形式。"""
    slot_cfg = config.STATE_ACTION_COMPARE_SLOTS.get(dataset_key)
    if slot_cfg is None:
        return None
    state_slots = _normalize_slot_list(slot_cfg["state_slot"])
    action_slots = _normalize_slot_list(slot_cfg["action_slot"])
    for s in state_slots + action_slots:
        if s in GRIPPER_SLOTS:
            print(f"[ERROR] 配置中 {dataset_key} 使用了 gripper slot '{s}'，已跳过。", file=sys.stderr)
            return None
    return state_slots, action_slots


def _ensure_unified_output(dataset_path: str, episodes: int = None, force: bool = False) -> str:
    output_dir = os.path.join(dataset_path, "unified_output")
    meta_path = os.path.join(output_dir, "unified_meta.json")

    if not force:
        if os.path.isdir(output_dir) and os.path.isfile(meta_path):
            parquets = glob_mod.glob(os.path.join(output_dir, "episode_*.parquet"))
            if parquets:
                print(f"已有 unified_output（{len(parquets)} 个 parquet），直接复用。"
                      f"如需补齐请加 --force-reconvert")
                return output_dir

    print(f"调用 convert_unified.py 生成/补齐 unified_output...")
    convert_script = os.path.join(SCRIPT_DIR, "convert_unified.py")
    cmd = [
        sys.executable, convert_script,
        dataset_path,
        "--output-dir", output_dir,
        "--skip-existing",
    ]
    if episodes is not None:
        cmd += ["--episodes", str(episodes)]
    print(f"  命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] convert_unified.py 失败:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"convert_unified.py 失败 (exit code {result.returncode})")
    print(result.stdout)
    return output_dir


def _load_unified_meta(output_dir: str) -> dict:
    meta_path = os.path.join(output_dir, "unified_meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return meta["slot_layout"]


def _list_unified_parquets(output_dir: str) -> list:
    return sorted(glob_mod.glob(os.path.join(output_dir, "episode_*.parquet")))


def _extract_slot_arrays(parquet_path, state_ranges, action_ranges):
    """从 unified parquet 中提取多个 slot 范围并拼接。

    Parameters
    ----------
    parquet_path : str
    state_ranges : list of (start, end) — state 各 slot 的维度范围
    action_ranges : list of (start, end) — action 各 slot 的维度范围
    """
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    episode_name = os.path.basename(parquet_path)

    state_all = np.array(df["state_unified"].tolist(), dtype=np.float64)
    action_all = np.array(df["action_unified"].tolist(), dtype=np.float64)
    mask_s_all = np.array(df["mask_state"].tolist(), dtype=bool)
    mask_a_all = np.array(df["mask_action"].tolist(), dtype=bool)
    frame_indices = df["frame_index"].values.astype(int)

    # 拼接多个 slot 范围
    state_parts, action_parts = [], []
    mask_s_parts, mask_a_parts = [], []
    for s_start, s_end in state_ranges:
        state_parts.append(state_all[:, s_start:s_end])
        mask_s_parts.append(mask_s_all[:, s_start:s_end])
    for a_start, a_end in action_ranges:
        action_parts.append(action_all[:, a_start:a_end])
        mask_a_parts.append(mask_a_all[:, a_start:a_end])

    state_slot = np.concatenate(state_parts, axis=1)
    action_slot = np.concatenate(action_parts, axis=1)
    mask_s_slot = np.concatenate(mask_s_parts, axis=1)
    mask_a_slot = np.concatenate(mask_a_parts, axis=1)

    valid_dim_mask = (mask_s_slot & mask_a_slot).all(axis=0)
    if not valid_dim_mask.any():
        return episode_name, None, None, frame_indices

    return (episode_name,
            state_slot[:, valid_dim_mask],
            action_slot[:, valid_dim_mask],
            frame_indices)


def _process_episode_worker(parquet_path, state_ranges, action_ranges):
    episode_name = os.path.basename(parquet_path)
    try:
        episode_name, state_valid, action_valid, frame_indices = \
            _extract_slot_arrays(parquet_path, state_ranges, action_ranges)
    except Exception as e:
        print(f"[WARN] 跳过损坏的 unified parquet {parquet_path}: {e}")
        return {
            "episode_name": episode_name,
            "valid_frames": 0,
            "l2_score": None,
            "corrupted": True,
        }

    if state_valid is None:
        return {
            "episode_name": episode_name,
            "valid_frames": 0,
            "l2_score": None,
        }

    score = compute_vla_l2_score(state_valid, action_valid)
    return {
        "episode_name": episode_name,
        "valid_frames": int(state_valid.shape[0]),
        "l2_score": score,
    }


def process_episode_local(parquet_path, state_ranges, action_ranges):
    episode_name = os.path.basename(parquet_path)
    try:
        episode_name, state_valid, action_valid, frame_indices = \
            _extract_slot_arrays(parquet_path, state_ranges, action_ranges)
    except Exception as e:
        print(f"[WARN] 跳过损坏的 unified parquet {parquet_path}: {e}")
        return {
            "episode_name": episode_name,
            "valid_frames": 0,
            "l2_score": None,
            "frame_indices": [],
            "similarity": None,
        }

    if state_valid is None:
        return {
            "episode_name": episode_name,
            "valid_frames": 0,
            "l2_score": None,
            "frame_indices": frame_indices.tolist(),
            "similarity": None,
        }

    sim = compute_episode_similarity(state_valid, action_valid)
    return {
        "episode_name": episode_name,
        "valid_frames": int(state_valid.shape[0]),
        "l2_score": sim["l2_score"],
        "frame_indices": frame_indices.tolist(),
        "similarity": sim,
    }


def plot_episode(ep_result: dict, state_slot: str, action_slot: str, plot_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep_name = ep_result["episode_name"]
    sim = ep_result.get("similarity")
    if sim is None or ep_result["valid_frames"] < 2:
        return None

    state_norm = sim["state_norm"]
    action_norm = sim["action_norm"]
    l2_score = sim["l2_score"]
    T, D = state_norm.shape
    frames = np.arange(T)

    n_rows = D
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3 * n_rows), sharex=True, squeeze=False)
    axes = axes.flatten()

    fig.suptitle(
        f"{ep_name}  |  state.{state_slot} vs action.{action_slot}\n"
        f"L2 score = {l2_score:.4f}",
        fontsize=13,
    )

    for d in range(D):
        ax = axes[d]
        ax.plot(frames, state_norm[:, d], label=f"state dim{d}", color="steelblue", linewidth=1.2)
        ax.plot(frames, action_norm[:, d], label=f"action dim{d}", color="darkorange", linewidth=1.2, linestyle="--")
        ax.set_ylabel(f"dim {d}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("frame_index")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, ep_name.replace(".parquet", ".png"))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def check_l2(dataset_path: str, dataset_key: str, episodes: int, threshold: float,
              force_reconvert: bool, plot: bool, plot_dir: str) -> tuple[dict, dict]:
    """执行 L2 检查，返回 (l2_reasons, l2_report_info)。
    l2_reasons: {ep_id: [reason, ...]}
    l2_report_info: 详细报告 dict（统计信息等）
    """
    slots = _get_compare_slots(dataset_key)
    if slots is None:
        print(f"数据集 {dataset_key} 未配置 state-action 对比 slot 或配置为 None，跳过 L2 检查。")
        return {}, {}

    # 检查 modality.json 是否存在
    modality_path = os.path.join(dataset_path, "meta", "modality.json")
    if not os.path.isfile(modality_path):
        print(f"[WARN] 数据集缺少 meta/modality.json，无法执行 L2 检查。")
        print(f"       如需执行 L2 检查，请先创建 modality.json 文件。")
        return {}, {}

    state_slots, action_slots = slots
    state_slot_label = "+".join(state_slots)
    action_slot_label = "+".join(action_slots)
    print(f"\nL2 检查: state.{state_slot_label} vs action.{action_slot_label}")
    print(f"  阈值: {threshold if threshold is not None else '无（仅记录统计）'}")

    # 确保 unified_output 存在
    output_dir = _ensure_unified_output(dataset_path, episodes, force=force_reconvert)

    # 读取 unified_meta.json
    slot_layout = _load_unified_meta(output_dir)
    for s in state_slots:
        if s not in slot_layout:
            raise RuntimeError(f"state_slot '{s}' 不在 slot_layout 中。可用: {list(slot_layout.keys())}")
    for s in action_slots:
        if s not in slot_layout:
            raise RuntimeError(f"action_slot '{s}' 不在 slot_layout 中。可用: {list(slot_layout.keys())}")

    # 构建多 slot 的维度范围列表
    state_ranges = [(slot_layout[s]["start"], slot_layout[s]["end"]) for s in state_slots]
    action_ranges = [(slot_layout[s]["start"], slot_layout[s]["end"]) for s in action_slots]

    for s, (start, end) in zip(state_slots, state_ranges):
        print(f"  State slot [{s}]: 维度 [{start}:{end})")
    for s, (start, end) in zip(action_slots, action_ranges):
        print(f"  Action slot [{s}]: 维度 [{start}:{end})")

    # 准备 episode 列表
    parquet_files = _list_unified_parquets(output_dir)
    n_episodes = min(episodes, len(parquet_files)) if episodes is not None else len(parquet_files)
    parquet_files = parquet_files[:n_episodes]
    print(f"  共 {n_episodes} 个 episode 待处理")

    # 并行计算 L2
    num_workers = min(config.NUM_WORKERS, n_episodes)
    print(f"  并行 worker 数: {num_workers}")

    worker_fn = partial(_process_episode_worker,
                        state_ranges=state_ranges, action_ranges=action_ranges)

    t_start = time.time()
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(worker_fn, parquet_files)
    t_l2 = time.time() - t_start
    print(f"  L2 并行计算完成，耗时 {t_l2:.1f}s（{n_episodes} episodes, {num_workers} workers）")

    # 汇总
    l2_scores = []
    flagged_episodes = []
    for i, r in enumerate(results):
        score = r["l2_score"]
        flag = ""
        if score is not None and not np.isnan(score):
            l2_scores.append(score)
            if threshold is not None and score > threshold:
                flag = " [FLAGGED]"
                flagged_episodes.append(r["episode_name"])
        if r["valid_frames"] > 0 and score is not None:
            print(f"    [{i+1}/{n_episodes}] {r['episode_name']}: "
                  f"L2={score:.4f}, valid_frames={r['valid_frames']}{flag}")
        else:
            print(f"    [{i+1}/{n_episodes}] {r['episode_name']}: 无有效帧（mask 不满足）")

    # 绘图
    if plot:
        print(f"\n  开始绘图...")
        t_plot = time.time()
        for i, pf in enumerate(parquet_files):
            ep_local = process_episode_local(pf, state_ranges, action_ranges)
            png_path = plot_episode(ep_local, state_slot_label, action_slot_label, plot_dir)
            if png_path:
                print(f"    [{i+1}/{n_episodes}] {os.path.basename(pf)} -> {png_path}")
        print(f"  绘图完成，耗时 {time.time() - t_plot:.1f}s")

    # 统计（过滤掉 inf，避免破坏均值/分位数）
    l2_arr = np.array(l2_scores)
    l2_finite = l2_arr[np.isfinite(l2_arr)]
    n_inf = int(np.sum(~np.isfinite(l2_arr)))
    if n_inf > 0:
        print(f"  注意: {n_inf} 个 episode 的 L2 为 inf（state/action 一方静止另一方不静止）")
    stats = {}
    if len(l2_finite) > 0:
        stats = {
            "mean_l2": float(l2_finite.mean()),
            "median_l2": float(np.median(l2_finite)),
            "std_l2": float(l2_finite.std()),
            "min_l2": float(l2_finite.min()),
            "max_l2": float(l2_finite.max()),
            "p90_l2": float(np.percentile(l2_finite, 90)),
            "p95_l2": float(np.percentile(l2_finite, 95)),
            "p99_l2": float(np.percentile(l2_finite, 99)),
            "n_inf": n_inf,
        }
        print(f"  L2 统计: mean={stats['mean_l2']:.4f}, median={stats['median_l2']:.4f}, "
              f"p90={stats['p90_l2']:.4f}, p95={stats['p95_l2']:.4f}, max={stats['max_l2']:.4f}")

    # 收集 L2 原因
    l2_reasons = {}
    for r in results:
        ep_name = r["episode_name"]
        score = r["l2_score"]
        ep_id = ep_name.replace("episode_", "").replace(".parquet", "").lstrip("0") or "0"

        reasons = []
        if r.get("corrupted"):
            reasons.append("parquet文件损坏(unified_output中文件无法读取)")
        elif r["valid_frames"] == 0 or score is None:
            reasons.append("无有效帧(state/action mask不满足)")
        elif np.isnan(score):
            reasons.append(f"帧数过少(valid_frames={r['valid_frames']}),无法计算L2")
        elif threshold is not None and score > threshold:
            reasons.append(f"L2={score:.4f},超过阈值{threshold}")

        if reasons:
            l2_reasons[ep_id] = reasons

    # 详细报告
    report_info = {
        "state_slot": state_slots if len(state_slots) > 1 else state_slots[0],
        "action_slot": action_slots if len(action_slots) > 1 else action_slots[0],
        "metric": "range_norm_l2",
        "threshold": threshold,
        "total_episodes": len(results),
        "valid_episodes": len(l2_scores),
        "flagged_episodes": flagged_episodes,
        "statistics": stats,
        "episodes": [
            {
                "episode_name": r["episode_name"],
                "valid_frames": r["valid_frames"],
                "l2_score": r["l2_score"],
            }
            for r in results
        ],
    }

    return l2_reasons, report_info


# ============================================================
# 统计摘要计算（扁平模式和子数据集模式共用）
# ============================================================

def _compute_summary(merged: dict, dataset_key: str, dataset_path: str,
                     threshold, l2_report_info: dict) -> dict:
    """根据合并后的 reasons 计算统计摘要。"""
    n_frame_task_only = sum(1 for v in merged.values()
                            if all("frame" in r or "task" in r for r in v))
    n_l2_only = sum(1 for v in merged.values()
                     if all("L2" in r or "无有效帧" in r or "无法计算" in r for r in v))
    n_both = len(merged) - n_frame_task_only - n_l2_only

    n_frame_bad = sum(1 for v in merged.values()
                      if any("frame" in r for r in v))
    n_task_bad = sum(1 for v in merged.values()
                     if any("task" in r for r in v))
    n_l2_bad = sum(1 for v in merged.values()
                    if any("L2" in r or "无有效帧" in r or "无法计算" in r for r in v))
    n_corrupted = sum(1 for v in merged.values()
                      if any("parquet文件损坏" in r or "parquet文件无法访问" in r for r in v))

    summary = {
        "dataset": dataset_key,
        "dataset_path": dataset_path,
        "l2_threshold": threshold,
        "total_problem_episodes": len(merged),
        "frame_task_only": n_frame_task_only,
        "l2_only": n_l2_only,
        "both": n_both,
        "breakdown": {
            "frame_too_short": n_frame_bad,
            "task_empty": n_task_bad,
            "l2_abnormal": n_l2_bad,
            "parquet_corrupted": n_corrupted,
        },
    }
    if l2_report_info.get("statistics"):
        summary["l2_statistics"] = l2_report_info["statistics"]
    return summary


def _merge_reasons(*reason_dicts) -> dict:
    """合并多个 reasons dict（帧数/task、L2、parquet 损坏等）。"""
    all_ids = set()
    for rd in reason_dicts:
        all_ids |= set(rd.keys())
    merged = {}
    for ep_id in sorted(all_ids, key=lambda x: int(x)):
        combined = []
        for rd in reason_dicts:
            if ep_id in rd:
                combined.extend(rd[ep_id])
        if combined:
            merged[ep_id] = combined
    return merged


# ============================================================
# 扁平模式（原有逻辑）
# ============================================================

def _run_flat_mode(dataset_path: str, args):
    """扁平数据集模式：顶层有 meta/episodes.jsonl。"""
    plot = args.plot or config.PLOT
    dataset_key = _resolve_dataset_key(dataset_path)

    # 确定 L2 阈值
    if args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = config.DATASET_L2_THRESHOLD.get(
            dataset_key, config.DEFAULT_L2_THRESHOLD)

    print(f"{'='*60}")
    print(f"数据集: {dataset_path}")
    print(f"配置 key: {dataset_key}")
    print(f"模式: 扁平")
    print(f"L2 阈值: {threshold if threshold is not None else '无（仅生成报告）'}")
    print(f"{'='*60}")

    # 1. parquet 完整性检查
    print(f"\n--- 第 1 步: parquet 完整性检查 ---")
    corrupted_reasons = check_corrupted_parquets(dataset_path, episodes=args.episodes)

    # 2. 帧数 / task 检查
    print(f"\n--- 第 2 步: 帧数 / task 检查 ---")
    frame_task_reasons = check_frame_and_task(dataset_path, dataset_key, episodes=args.episodes)

    # 3. L2 检查
    print(f"\n--- 第 3 步: L2 state-action 检查 ---")
    plot_dir = args.plot_dir or os.path.join(SCRIPT_DIR, "plots", dataset_key)
    l2_reasons, l2_report_info = check_l2(
        dataset_path, dataset_key, args.episodes, threshold,
        args.force_reconvert, plot, plot_dir)

    # 4. 合并
    print(f"\n--- 第 4 步: 合并结果 ---")
    merged = _merge_reasons(corrupted_reasons, frame_task_reasons, l2_reasons)
    summary = _compute_summary(merged, dataset_key, dataset_path,
                               threshold, l2_report_info)

    # 5. 写入报告
    safe_key = dataset_key.replace(os.sep, "_").replace("/", "_")
    filter_report_path = os.path.join(SCRIPT_DIR, f"{safe_key}_filter_report.json")
    report = {
        "summary": summary,
        dataset_key: merged,
    }
    with open(filter_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 6. L2 详细报告
    if l2_report_info:
        l2_detail_path = os.path.join(SCRIPT_DIR, "state_action_diff_report.json")
        l2_report_info["dataset"] = dataset_key
        l2_report_info["dataset_path"] = dataset_path
        with open(l2_detail_path, "w", encoding="utf-8") as f:
            json.dump(l2_report_info, f, ensure_ascii=False, indent=2)
        print(f"L2 详细报告: {l2_detail_path}")

    # 7. 打印摘要
    bd = summary["breakdown"]
    print(f"\n{'='*60}")
    print(f"筛选报告: {filter_report_path}")
    print(f"共 {summary['total_problem_episodes']} 个问题 episode:")
    print(f"  parquet损坏: {bd['parquet_corrupted']}")
    print(f"  帧数不足: {bd['frame_too_short']}")
    print(f"  task为空: {bd['task_empty']}")
    print(f"  L2异常:  {bd['l2_abnormal']}")
    print(f"  ---")
    print(f"  仅帧数/task 问题: {summary['frame_task_only']}")
    print(f"  仅 L2 问题:      {summary['l2_only']}")
    print(f"  多种问题:         {summary['both']}")
    print(f"{'='*60}")
    if plot:
        print(f"曲线图目录: {plot_dir}")


# ============================================================
# 子数据集模式
# ============================================================


def _discover_leaf_datasets(root_path: str) -> list[str]:
    """递归查找 root_path 下所有含 meta/episodes.jsonl 的叶子数据集目录。
    跳过 SKIP_DIRS 和 meta 目录。
    """
    skip = config.SKIP_DIRS | {"meta"}
    leaves = []

    def _walk(path):
        try:
            entries = sorted(os.listdir(path))
        except PermissionError:
            return
        for entry in entries:
            if entry in skip:
                continue
            full = os.path.join(path, entry)
            if not os.path.isdir(full):
                continue
            # 检查这个目录是否是叶子数据集
            if _find_episode_jsonl(full) is not None:
                leaves.append(full)
            else:
                # 继续递归
                _walk(full)

    _walk(root_path)
    return leaves



def _process_one_subdataset(leaf_path: str, dataset_key: str, episodes: int,
                            threshold, force_reconvert: bool, plot: bool,
                            plot_dir: str) -> dict:
    """处理单个子数据集，返回结果 dict。
    返回: {"summary": {...}, "episodes": {...}, "l2_scores": [...]}
    """
    leaf_name = os.path.basename(leaf_path)

    # parquet 完整性检查
    corrupted_reasons = check_corrupted_parquets(leaf_path, episodes=episodes, quiet=True)

    # 帧数 / task 检查
    frame_task_reasons = check_frame_and_task(leaf_path, dataset_key, episodes=episodes, quiet=True)

    # L2 检查
    l2_reasons, l2_report_info = check_l2(
        leaf_path, dataset_key, episodes, threshold,
        force_reconvert, plot, plot_dir)

    # 合并
    merged = _merge_reasons(corrupted_reasons, frame_task_reasons, l2_reasons)

    # 获取总 episode 数
    ep_file = _find_episode_jsonl(leaf_path)
    total_episodes = 0
    if ep_file:
        with open(ep_file, "r", encoding="utf-8") as f:
            total_episodes = sum(1 for line in f if line.strip())

    # 统计
    summary = _compute_summary(merged, leaf_name, leaf_path, threshold, l2_report_info)
    summary["total_episodes"] = total_episodes

    # 收集所有 L2 分数用于全局统计
    all_l2_scores = []
    if l2_report_info.get("episodes"):
        for ep in l2_report_info["episodes"]:
            s = ep.get("l2_score")
            if s is not None and not np.isnan(s):
                all_l2_scores.append(s)

    return {
        "summary": summary,
        "episodes": merged,
        "l2_scores": all_l2_scores,
    }


def _run_subdataset_mode(dataset_path: str, args):
    """子数据集模式：递归查找叶子数据集，逐个处理后汇总。"""
    plot = args.plot or config.PLOT
    dataset_key = _resolve_dataset_key(dataset_path)

    # 确定 L2 阈值
    if args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = config.DATASET_L2_THRESHOLD.get(
            dataset_key, config.DEFAULT_L2_THRESHOLD)

    # 发现叶子数据集
    leaves = _discover_leaf_datasets(dataset_path)
    if not leaves:
        print(f"[ERROR] 在 {dataset_path} 下未发现任何含 meta/episodes.jsonl 的子数据集。", file=sys.stderr)
        sys.exit(1)

    # 限制子数据集数量（用于测试）
    total_leaves = len(leaves)
    if args.max_subsets_num is not None and args.max_subsets_num < total_leaves:
        leaves = leaves[:args.max_subsets_num]

    print(f"{'='*60}")
    print(f"数据集: {dataset_path}")
    print(f"配置 key: {dataset_key}")
    print(f"模式: 子数据集")
    print(f"发现 {total_leaves} 个子数据集" +
          (f"，本次处理前 {len(leaves)} 个" if len(leaves) < total_leaves else ""))
    print(f"L2 阈值: {threshold if threshold is not None else '无（仅生成报告）'}")
    print(f"{'='*60}")

    # 逐个处理子数据集
    subdatasets_results = {}
    all_l2_scores_global = []
    total_corrupted = 0
    total_frame_bad = 0
    total_task_bad = 0
    total_l2_bad = 0
    total_problem_eps = 0
    total_episodes_global = 0
    n_processed = 0
    n_failed = 0

    for idx, leaf_path in enumerate(leaves):
        leaf_name = os.path.basename(leaf_path)
        print(f"\n[{idx+1}/{len(leaves)}] {leaf_name}")

        # 检查叶子数据集是否有 modality.json
        leaf_modality = os.path.join(leaf_path, "meta", "modality.json")
        if not os.path.isfile(leaf_modality):
            print(f"  [SKIP] 缺少 meta/modality.json，跳过")
            n_failed += 1
            continue

        try:
            leaf_plot_dir = os.path.join(
                args.plot_dir or os.path.join(SCRIPT_DIR, "plots", dataset_key),
                leaf_name)

            result = _process_one_subdataset(
                leaf_path, dataset_key, args.episodes, threshold,
                args.force_reconvert, plot, leaf_plot_dir)

            sub_summary = result["summary"]
            sub_bd = sub_summary["breakdown"]
            n_prob = sub_summary["total_problem_episodes"]

            subdatasets_results[leaf_name] = {
                "summary": sub_summary,
                "episodes": result["episodes"],
            }

            all_l2_scores_global.extend(result["l2_scores"])
            total_corrupted += sub_bd["parquet_corrupted"]
            total_frame_bad += sub_bd["frame_too_short"]
            total_task_bad += sub_bd["task_empty"]
            total_l2_bad += sub_bd["l2_abnormal"]
            total_problem_eps += n_prob
            total_episodes_global += sub_summary.get("total_episodes", 0)
            n_processed += 1

            # 简洁进度
            print(f"  -> {n_prob} 个问题 episode"
                  f" (损坏:{sub_bd['parquet_corrupted']}, 帧数:{sub_bd['frame_too_short']},"
                  f" task:{sub_bd['task_empty']}, L2:{sub_bd['l2_abnormal']})"
                  f"  共 {sub_summary.get('total_episodes', '?')} 个 episode")

        except Exception as e:
            print(f"  [ERROR] 处理失败: {e}")
            n_failed += 1
            continue

    # 全局 L2 统计
    global_l2_stats = {}
    if all_l2_scores_global:
        l2_arr = np.array(all_l2_scores_global)
        global_l2_stats = {
            "mean_l2": float(l2_arr.mean()),
            "median_l2": float(np.median(l2_arr)),
            "std_l2": float(l2_arr.std()),
            "min_l2": float(l2_arr.min()),
            "max_l2": float(l2_arr.max()),
            "p90_l2": float(np.percentile(l2_arr, 90)),
            "p95_l2": float(np.percentile(l2_arr, 95)),
            "p99_l2": float(np.percentile(l2_arr, 99)),
        }

    # 全局摘要
    global_summary = {
        "dataset": dataset_key,
        "dataset_path": dataset_path,
        "mode": "subdataset",
        "total_subdatasets": len(leaves),
        "processed_subdatasets": n_processed,
        "failed_subdatasets": n_failed,
        "total_episodes": total_episodes_global,
        "total_problem_episodes": total_problem_eps,
        "l2_threshold": threshold,
        "breakdown": {
            "parquet_corrupted": total_corrupted,
            "frame_too_short": total_frame_bad,
            "task_empty": total_task_bad,
            "l2_abnormal": total_l2_bad,
        },
    }
    if global_l2_stats:
        global_summary["l2_statistics"] = global_l2_stats

    # 写入报告
    filter_report_path = os.path.join(SCRIPT_DIR, f"{dataset_key}_filter_report.json")
    report = {
        "summary": global_summary,
        "subdatasets": subdatasets_results,
    }
    with open(filter_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 打印全局摘要
    print(f"\n{'='*60}")
    print(f"筛选报告: {filter_report_path}")
    print(f"子数据集: {n_processed} 成功, {n_failed} 失败 (共 {len(leaves)} 个)")
    print(f"总 episode 数: {total_episodes_global}")
    print(f"共 {total_problem_eps} 个问题 episode:")
    print(f"  parquet损坏: {total_corrupted}")
    print(f"  帧数不足: {total_frame_bad}")
    print(f"  task为空: {total_task_bad}")
    print(f"  L2异常:  {total_l2_bad}")
    if global_l2_stats:
        print(f"  L2 全局统计: mean={global_l2_stats['mean_l2']:.4f}, "
              f"median={global_l2_stats['median_l2']:.4f}, "
              f"p90={global_l2_stats['p90_l2']:.4f}, "
              f"p95={global_l2_stats['p95_l2']:.4f}, "
              f"max={global_l2_stats['max_l2']:.4f}")
    print(f"{'='*60}")


# ============================================================
# main: 自动检测模式
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="综合筛选：帧数 + task + L2(state, action)，输出 {数据集}_filter_report.json。"
                    "自动检测扁平/子数据集模式。")
    parser.add_argument("dataset_path", help="LeRobot v2.1 数据集路径（扁平或含子数据集的父目录）")
    parser.add_argument("--episodes", type=int, default=None,
                        help="处理的 episode 数量（默认: 全部）。子数据集模式下限制每个子数据集的 episode 数")
    parser.add_argument("--threshold", type=float, default=None,
                        help="L2 距离阈值，超过则标记为异常（覆盖 config 中该数据集的阈值）")
    parser.add_argument("--plot", action="store_true",
                        help="是否为每个 episode 输出 state vs action 对比曲线图")
    parser.add_argument("--plot-dir", type=str, default=None,
                        help="曲线图输出目录（默认: Filter/plots/<dataset_key>/）")
    parser.add_argument("--force-reconvert", action="store_true",
                        help="强制重新生成 unified_output（增量模式：跳过已有 parquet，只补齐缺失的）")
    parser.add_argument("--max-subsets-num", type=int, default=None,
                        help="子数据集模式下最多处理的子数据集数量（默认: 全部）。用于快速测试")
    args = parser.parse_args()

    dataset_path = args.dataset_path.rstrip("/")

    # 自动检测模式
    if _find_episode_jsonl(dataset_path) is not None:
        # 扁平模式
        _run_flat_mode(dataset_path, args)
    else:
        # 子数据集模式
        _run_subdataset_mode(dataset_path, args)


if __name__ == "__main__":
    main()
