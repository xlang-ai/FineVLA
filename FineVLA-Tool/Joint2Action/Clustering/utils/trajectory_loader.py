"""
从 LeRobot v2.1 parquet 文件中加载轨迹数据。
支持多种数据集格式：
  - EEF quaternion (Galaxea, RT-1, agibotworld)
  - EEF euler (Bridge, BC_Z, droid, RoboCOIN, egodex)
  - Joint-only (RDT, RH20T, RoboMindV2)
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pyarrow.parquet as pq

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DatasetConfig, ArmConfig


# ═══════════════════════════════════════════════════════════
#  Trajectory 数据结构
# ═══════════════════════════════════════════════════════════

@dataclass
class Trajectory:
    episode_id: int
    file_path: str
    combined: np.ndarray       # (T, D) 统一格式的轨迹向量
    rot_type: str = "quaternion"

    @property
    def length(self) -> int:
        return self.combined.shape[0]

    @property
    def dim(self) -> int:
        return self.combined.shape[1]


# ═══════════════════════════════════════════════════════════
#  Euler → Quaternion 转换
# ═══════════════════════════════════════════════════════════

def euler_to_quat(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """
    将 Euler 角 (roll, pitch, yaw) 批量转为 quaternion (qx, qy, qz, qw)。
    输入 shape: (T,) 各一列，输出 shape: (T, 4)。
    """
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.stack([qx, qy, qz, qw], axis=-1)


# ═══════════════════════════════════════════════════════════
#  Parquet 列解析
# ═══════════════════════════════════════════════════════════

def _parse_column(col) -> np.ndarray:
    """将 parquet 中一列（可能是 ndarray/list/scalar）转为 (N, D) 数组。"""
    first = col.iloc[0]
    if isinstance(first, np.ndarray):
        return np.stack(col.values).astype(np.float64)
    elif isinstance(first, (list, tuple)):
        return np.array(col.tolist(), dtype=np.float64)
    else:
        return col.values.astype(np.float64).reshape(-1, 1)


def _concat_columns(df, column_names: list[str]) -> np.ndarray:
    """从 DataFrame 中提取多列并水平拼接为 (T, D) 数组。"""
    arrays = []
    for col_name in column_names:
        if col_name not in df.columns:
            raise KeyError(f"Column '{col_name}' not found. Available: {list(df.columns)}")
        arrays.append(_parse_column(df[col_name]))
    if not arrays:
        return np.empty((len(df), 0), dtype=np.float64)
    return np.hstack(arrays)


# ═══════════════════════════════════════════════════════════
#  核心: 按 config 构建轨迹向量
# ═══════════════════════════════════════════════════════════

def _build_combined_vector(
    df,
    arm: ArmConfig,
    rot_type: str,
) -> np.ndarray:
    """
    根据 ArmConfig 从 DataFrame 构建统一的轨迹向量。

    返回格式:
      rot_type == "quaternion" → [x, y, z, qx, qy, qz, qw, gripper]  (8D)
      rot_type == "euler"      → 先转 quat → [x, y, z, qx, qy, qz, qw, gripper]  (8D)
      rot_type == "none"       → [joints..., gripper]  (D 可变)
    """

    if rot_type == "none":
        # Joint-only mode
        all_cols = arm.joint_columns + arm.gripper_columns
        if not all_cols:
            raise ValueError("Joint-only mode requires joint_columns or gripper_columns")
        raw = _concat_columns(df, all_cols)

        if arm.joint_indices or arm.grip_indices:
            # 使用显式指定的索引
            joints = raw[:, arm.joint_indices] if arm.joint_indices else np.empty((len(df), 0))
            grip_idx = arm.grip_indices
            gripper = raw[:, grip_idx] if grip_idx else np.zeros((len(df), 1))
        else:
            # 自动推断：最后一列为 gripper，其余为 joints
            D = raw.shape[1]
            joints = raw[:, :D - 1]
            gripper = raw[:, D - 1:D]
        return np.hstack([joints, gripper])

    # EEF mode (quaternion or euler)
    all_cols = arm.eef_columns + arm.gripper_columns
    if not all_cols:
        raise ValueError(f"EEF mode (rot_type={rot_type}) requires eef_columns")
    raw = _concat_columns(df, all_cols)

    pos = raw[:, arm.pos_indices]
    rot_raw = raw[:, arm.rot_indices]
    grip_idx = arm.grip_indices
    gripper = raw[:, grip_idx] if grip_idx else np.zeros((len(df), 1))

    if rot_type == "euler":
        # euler [roll, pitch, yaw] → quaternion [qx, qy, qz, qw]
        if rot_raw.shape[1] != 3:
            raise ValueError(
                f"Euler mode expects 3 rotation components, got {rot_raw.shape[1]}"
            )
        quat = euler_to_quat(rot_raw[:, 0], rot_raw[:, 1], rot_raw[:, 2])
    elif rot_type == "quaternion":
        if rot_raw.shape[1] != 4:
            raise ValueError(
                f"Quaternion mode expects 4 rotation components, got {rot_raw.shape[1]}"
            )
        quat = rot_raw
    else:
        raise ValueError(f"Unknown rot_type: {rot_type}")

    if gripper.ndim == 1:
        gripper = gripper.reshape(-1, 1)

    return np.hstack([pos, quat, gripper])  # (T, 8)


# ═══════════════════════════════════════════════════════════
#  特征归一化
# ═══════════════════════════════════════════════════════════

def normalize_trajectories(
    trajectories: list[Trajectory],
    rot_type: str,
) -> tuple[list[Trajectory], dict]:
    """
    对轨迹特征做 min-max 归一化到 [0, 1]，使不同量纲的分量可比。

    归一化策略:
      EEF 模式 (quaternion/euler) → 向量 [x,y,z, qx,qy,qz,qw, grip]
        - position (dim 0-2): 归一化 ✓
        - quaternion (dim 3-6): 不动 (geodesic 自然范围 [0,π])
        - gripper (dim 7+):  归一化 ✓

      Joint 模式 (none) → 向量 [joints..., grip]
        - 所有维度: 归一化 ✓

    统计量基于当前所有 trajectory 的全局 min/max。

    Returns
    -------
    normalized : 归一化后的 Trajectory 列表
    stats : 每个归一化维度的 {min, max, range} 诊断信息
    """
    if len(trajectories) < 2:
        return trajectories, {}

    all_data = np.vstack([t.combined for t in trajectories])
    D = all_data.shape[1]

    if rot_type in ("quaternion", "euler"):
        norm_dims = list(range(3)) + list(range(7, D))
    else:
        norm_dims = list(range(D))

    mins = all_data.min(axis=0)
    maxs = all_data.max(axis=0)
    ranges = maxs - mins
    ranges[ranges < 1e-12] = 1.0

    stats = {}
    for d in norm_dims:
        stats[f"dim_{d}"] = {
            "min": round(float(mins[d]), 6),
            "max": round(float(maxs[d]), 6),
            "range": round(float(ranges[d]), 6),
        }

    normalized = []
    for t in trajectories:
        data = t.combined.copy()
        for d in norm_dims:
            data[:, d] = (data[:, d] - mins[d]) / ranges[d]
        normalized.append(Trajectory(
            episode_id=t.episode_id,
            file_path=t.file_path,
            combined=data,
            rot_type=t.rot_type,
        ))

    return normalized, stats


# ═══════════════════════════════════════════════════════════
#  RoboMindV2.0 动态配置加载
# ═══════════════════════════════════════════════════════════

def load_modality_config(robot_type_path: str, side: str) -> dict:
    """
    从 modality.json 加载指定侧的配置。

    Parameters
    ----------
    robot_type_path : robot_type 目录路径 (e.g., /path/to/RoboMindV2.0/agilex)
    side : 'left' 或 'right'

    Returns
    -------
    config_dict : {
        "joint_column": str,
        "joint_indices": list[int],
        "effector_column": str,  # gripper 或 hand 的列名
        "effector_indices": list[int],
        "effector_type": "gripper" | "hand"
    }
    """
    modality_path = os.path.join(robot_type_path, "modality.json")
    if not os.path.exists(modality_path):
        raise FileNotFoundError(f"modality.json not found at {modality_path}")

    with open(modality_path, encoding="utf-8") as f:
        modality = json.load(f)

    action_config = modality.get("action", {})

    # 读取 joint 配置
    joint_key = f"{side}_joint"
    if joint_key not in action_config:
        raise KeyError(f"'{joint_key}' not found in modality.json action config")

    joint_info = action_config[joint_key]
    joint_column = joint_info["original_key"]
    joint_indices = joint_info["indices"]

    # 读取 effector 配置（优先 hand，其次 gripper）
    hand_key = f"{side}_hand"
    gripper_key = f"{side}_gripper"

    if hand_key in action_config:
        # 有灵巧手
        effector_info = action_config[hand_key]
        effector_type = "hand"
        effector_key = hand_key
    elif gripper_key in action_config:
        # 有夹爪
        effector_info = action_config[gripper_key]
        effector_type = "gripper"
        effector_key = gripper_key
    else:
        raise KeyError(f"Neither '{hand_key}' nor '{gripper_key}' found in modality.json action config")

    effector_column = effector_info["original_key"]
    effector_indices = effector_info["indices"]

    return {
        "joint_column": joint_column,
        "joint_indices": joint_indices,
        "effector_column": effector_column,
        "effector_indices": effector_indices,
        "effector_type": effector_type,
    }


def load_filter_report(filter_report_path: str) -> dict[str, set[int]]:
    """
    加载 filter_report.json，提取各个子数据集的问题 episode 列表。

    Parameters
    ----------
    filter_report_path : filter_report.json 文件路径

    Returns
    -------
    problem_episodes : {subdataset_name: set(problem_episode_ids)}
    """
    if not os.path.exists(filter_report_path):
        print(f"[WARN] filter_report not found: {filter_report_path}")
        return {}

    with open(filter_report_path, encoding="utf-8") as f:
        data = json.load(f)

    problem_episodes = {}

    # RoboMindV2.0_filter_report.json 格式:
    # {
    #   "summary": {...},
    #   "subdatasets": {
    #     "task_name": {
    #       "episodes": {
    #         "episode_id": [reasons]
    #       }
    #     }
    #   }
    # }
    subdatasets = data.get("subdatasets", {})
    for subdataset_name, subdataset_info in subdatasets.items():
        episodes_dict = subdataset_info.get("episodes", {})
        if episodes_dict:
            problem_set = set()
            for ep_id_str in episodes_dict.keys():
                try:
                    problem_set.add(int(ep_id_str))
                except ValueError:
                    pass
            if problem_set:
                problem_episodes[subdataset_name] = problem_set

    return problem_episodes


def infer_robot_type_from_path(sub_dataset_path: str, dataset_root: str) -> str:
    """
    从子数据集路径推断 robot_type（第一级子目录）。

    例如: /path/to/RoboMindV2.0/agilex/task1 → 'agilex'
    """
    rel_path = os.path.relpath(sub_dataset_path, dataset_root)
    parts = rel_path.split(os.sep)
    if len(parts) >= 1:
        return parts[0]
    return ""


# ═══════════════════════════════════════════════════════════
#  公共 API
# ═══════════════════════════════════════════════════════════

def load_trajectories(
    dataset_root: str,
    config: DatasetConfig,
    side: str = "right",
    max_episodes: int | None = None,
    exclude_episodes: set[int] | None = None,
    robot_type: str | None = None,
) -> list[Trajectory]:
    """
    扫描 dataset_root/data/chunk-*/episode_*.parquet，
    按 episode 拆分并构建统一格式的轨迹向量。

    Parameters
    ----------
    dataset_root : 单个（子）数据集根目录（包含 data/ 子目录）
    config : 数据集配置
    side : 要分析的手臂侧
    max_episodes : 最多加载几条轨迹（None=全部）
    exclude_episodes : 要排除的 episode ID 集合
    robot_type : 机器人型号（用于 RoboMindV2.0 动态配置加载）
    """
    if side not in config.arms:
        available = list(config.arms.keys())
        # 如果只有 single，自动映射
        if "single" in config.arms:
            side = "single"
        else:
            raise ValueError(f"Side '{side}' not available. Choose from: {available}")

    arm = config.arms[side]

    # ── 动态配置加载（RoboMindV2.0）──
    if config.has_modality_json and robot_type:
        # 推断 robot_type 路径
        robot_type_path = os.path.join(
            config.dataset_path, robot_type
        )

        if os.path.exists(os.path.join(robot_type_path, "modality.json")):
            print(f"  [INFO] 从 modality.json 加载 robot_type={robot_type}, side={side}")
            try:
                modality_cfg = load_modality_config(robot_type_path, side)

                # 动态构建 ArmConfig
                arm = ArmConfig(
                    eef_columns=[],
                    gripper_columns=[modality_cfg["effector_column"]],
                    joint_columns=[modality_cfg["joint_column"]],
                    joint_indices=modality_cfg["joint_indices"],
                    grip_indices=modality_cfg["effector_indices"],
                )

                print(f"    ✓ {side}_joint({len(modality_cfg['joint_indices'])}D) + "
                      f"{side}_{modality_cfg['effector_type']}({len(modality_cfg['effector_indices'])}D)")

            except Exception as e:
                print(f"  [WARN] 无法加载 modality.json: {e}，使用默认配置")
                # fallback 到原始 config 中的 arm
                arm = config.arms[side]

    pattern = os.path.join(dataset_root, "data", "chunk-*", "episode_*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files matching: {pattern}")

    trajectories: list[Trajectory] = []
    for fpath in files:
        df = pq.read_table(fpath).to_pandas()

        # 检查必需列
        needed = arm.eef_columns + arm.gripper_columns + arm.joint_columns
        needed = [c for c in needed if c]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f"[WARN] skip {fpath}: missing columns {missing}")
            continue

        episodes = sorted(df["episode_index"].unique()) if "episode_index" in df.columns else [0]
        for ep_id in episodes:
            if exclude_episodes and int(ep_id) in exclude_episodes:
                continue
            ep_df = df[df["episode_index"] == ep_id] if "episode_index" in df.columns else df
            if len(ep_df) < 2:
                continue

            try:
                combined = _build_combined_vector(ep_df, arm, config.rot_type)
            except Exception as e:
                print(f"[WARN] skip ep_{ep_id} in {fpath}: {e}")
                continue

            trajectories.append(Trajectory(
                episode_id=int(ep_id),
                file_path=fpath,
                combined=combined,
                rot_type=config.rot_type,
            ))
            if max_episodes and len(trajectories) >= max_episodes:
                return trajectories

    print(f"Loaded {len(trajectories)} trajectories from {dataset_root}")
    return trajectories


def discover_tasks(dataset_root: str) -> tuple[dict[str, list[int]], int]:
    """
    从 meta/episodes.jsonl 解析 task→episode_indices 映射。

    Returns
    -------
    task_map : {task_name: [episode_index, ...]}
    chunk_size : 从 meta/info.json 读取的 chunk 大小
    """
    episodes_path = os.path.join(dataset_root, "meta", "episodes.jsonl")
    info_path = os.path.join(dataset_root, "meta", "info.json")

    with open(info_path) as f:
        info = json.load(f)
    chunk_size = info.get("chunks_size", 1000)

    # 尝试从 tasks.jsonl 构建 task_index → task_name 映射
    tasks_path = os.path.join(dataset_root, "meta", "tasks.jsonl")
    idx_to_task: dict[int, str] = {}
    if os.path.isfile(tasks_path):
        with open(tasks_path) as f:
            for line in f:
                t = json.loads(line)
                idx_to_task[t["task_index"]] = t["task"]

    task_map: dict[str, list[int]] = {}
    with open(episodes_path) as f:
        for line in f:
            ep = json.loads(line)
            ep_idx = ep["episode_index"]
            # 优先使用 "tasks" 字段（字符串列表，如 BC_Z / RT-1）
            task_names = ep.get("tasks", [])
            if not task_names and "task_index" in ep:
                # 通过 task_index 查找 task 名称（如 RH20T）
                ti = ep["task_index"]
                if ti in idx_to_task:
                    task_names = [idx_to_task[ti]]
            for task_name in task_names:
                task_map.setdefault(task_name, []).append(ep_idx)

    for task_name in task_map:
        task_map[task_name].sort()

    return task_map, chunk_size


def load_trajectories_by_indices(
    dataset_root: str,
    config: DatasetConfig,
    side: str,
    episode_indices: list[int],
    chunk_size: int = 1000,
) -> list[Trajectory]:
    """
    按指定的 episode_index 列表加载轨迹，根据 index 定位 parquet 文件。

    Parameters
    ----------
    dataset_root : 数据集根目录（包含 data/ 子目录）
    config : 数据集配置
    side : 要分析的手臂侧
    episode_indices : 要加载的 episode 编号列表
    chunk_size : 每个 chunk 包含的 episode 数（从 info.json 读取）
    """
    if side not in config.arms:
        if "single" in config.arms:
            side = "single"
        else:
            raise ValueError(f"Side '{side}' not in {list(config.arms.keys())}")

    arm = config.arms[side]
    trajectories: list[Trajectory] = []

    for ep_idx in episode_indices:
        chunk_id = ep_idx // chunk_size
        fpath = os.path.join(
            dataset_root, "data",
            f"chunk-{chunk_id:03d}",
            f"episode_{ep_idx:06d}.parquet",
        )
        if not os.path.exists(fpath):
            print(f"[WARN] skip ep_{ep_idx}: file not found {fpath}")
            continue

        try:
            df = pq.read_table(fpath).to_pandas()
        except Exception as e:
            print(f"[WARN] skip ep_{ep_idx}: read error {e}")
            continue

        if len(df) < 2:
            continue

        needed = arm.eef_columns + arm.gripper_columns + arm.joint_columns
        needed = [c for c in needed if c]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f"[WARN] skip ep_{ep_idx}: missing columns {missing}")
            continue

        try:
            combined = _build_combined_vector(df, arm, config.rot_type)
        except Exception as e:
            print(f"[WARN] skip ep_{ep_idx}: {e}")
            continue

        trajectories.append(Trajectory(
            episode_id=ep_idx,
            file_path=fpath,
            combined=combined,
            rot_type=config.rot_type,
        ))

    return trajectories


def discover_sub_datasets(config: DatasetConfig) -> list[dict[str, str]]:
    """
    发现数据集下的所有子数据集路径。
    根据 sub_dataset_depth 扫描目录。

    Returns
    -------
    sub_datasets : list of dict with keys:
        - 'path': 子数据集完整路径
        - 'robot_type': 机器人型号（仅对 depth>=2 有效，如 RoboMindV2.0）
        - 'name': 子数据集相对名称
    """
    if not config.has_sub_datasets:
        return [{"path": config.dataset_path, "robot_type": "", "name": ""}]

    root = config.dataset_path
    depth = config.sub_dataset_depth

    if depth == 1:
        subs = []
        for name in sorted(os.listdir(root)):
            sub_path = os.path.join(root, name)
            data_dir = os.path.join(sub_path, "data")
            if os.path.isdir(data_dir):
                subs.append({
                    "path": sub_path,
                    "robot_type": "",
                    "name": name,
                })
        return subs

    if depth == 2:
        subs = []
        for level1 in sorted(os.listdir(root)):
            l1_path = os.path.join(root, level1)
            if not os.path.isdir(l1_path):
                continue
            for level2 in sorted(os.listdir(l1_path)):
                sub_path = os.path.join(l1_path, level2)
                data_dir = os.path.join(sub_path, "data")
                if os.path.isdir(data_dir):
                    subs.append({
                        "path": sub_path,
                        "robot_type": level1,  # level1 即为 robot_type
                        "name": f"{level1}/{level2}",
                    })
        return subs

    if depth == 3:
        subs = []
        for level1 in sorted(os.listdir(root)):
            l1_path = os.path.join(root, level1)
            if not os.path.isdir(l1_path):
                continue
            for level2 in sorted(os.listdir(l1_path)):
                l2_path = os.path.join(l1_path, level2)
                if not os.path.isdir(l2_path):
                    continue
                for level3 in sorted(os.listdir(l2_path)):
                    sub_path = os.path.join(l2_path, level3)
                    data_dir = os.path.join(sub_path, "data")
                    if os.path.isdir(data_dir):
                        subs.append({
                            "path": sub_path,
                            "robot_type": level2,  # level2 可能是 robot_type
                            "name": f"{level1}/{level2}/{level3}",
                        })
        return subs

    raise ValueError(f"Unsupported sub_dataset_depth: {depth}")
