"""
Load trajectory data from LeRobot v2.1 parquet files.
Supports multiple dataset formats:
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
#  Trajectory Data Structure
# ═══════════════════════════════════════════════════════════

@dataclass
class Trajectory:
    episode_id: int
    file_path: str
    combined: np.ndarray       # (T, D) unified trajectory vector
    rot_type: str = "quaternion"

    @property
    def length(self) -> int:
        return self.combined.shape[0]

    @property
    def dim(self) -> int:
        return self.combined.shape[1]


# ═══════════════════════════════════════════════════════════
#  Euler -> Quaternion Conversion
# ═══════════════════════════════════════════════════════════

def euler_to_quat(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """
    Batch-convert Euler angles (roll, pitch, yaw) to quaternion (qx, qy, qz, qw).
    Input shape: (T,) per column; output shape: (T, 4).
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
#  Parquet Column Parsing
# ═══════════════════════════════════════════════════════════

def _parse_column(col) -> np.ndarray:
    """Convert a parquet column (possibly ndarray/list/scalar) into an (N, D) array."""
    first = col.iloc[0]
    if isinstance(first, np.ndarray):
        return np.stack(col.values).astype(np.float64)
    elif isinstance(first, (list, tuple)):
        return np.array(col.tolist(), dtype=np.float64)
    else:
        return col.values.astype(np.float64).reshape(-1, 1)


def _concat_columns(df, column_names: list[str]) -> np.ndarray:
    """Extract multiple columns from a DataFrame and horizontally concatenate into a (T, D) array."""
    arrays = []
    for col_name in column_names:
        if col_name not in df.columns:
            raise KeyError(f"Column '{col_name}' not found. Available: {list(df.columns)}")
        arrays.append(_parse_column(df[col_name]))
    if not arrays:
        return np.empty((len(df), 0), dtype=np.float64)
    return np.hstack(arrays)


# ═══════════════════════════════════════════════════════════
#  Core: Build Trajectory Vector from Config
# ═══════════════════════════════════════════════════════════

def _build_combined_vector(
    df,
    arm: ArmConfig,
    rot_type: str,
) -> np.ndarray:
    """
    Build a unified trajectory vector from a DataFrame according to the ArmConfig.

    Output format:
      rot_type == "quaternion" -> [x, y, z, qx, qy, qz, qw, gripper]  (8D)
      rot_type == "euler"      -> convert to quat -> [x, y, z, qx, qy, qz, qw, gripper]  (8D)
      rot_type == "none"       -> [joints..., gripper]  (variable D)
    """

    if rot_type == "none":
        # Joint-only mode
        all_cols = arm.joint_columns + arm.gripper_columns
        if not all_cols:
            raise ValueError("Joint-only mode requires joint_columns or gripper_columns")
        raw = _concat_columns(df, all_cols)

        if arm.joint_indices or arm.grip_indices:
            # Use explicitly specified indices
            joints = raw[:, arm.joint_indices] if arm.joint_indices else np.empty((len(df), 0))
            grip_idx = arm.grip_indices
            gripper = raw[:, grip_idx] if grip_idx else np.zeros((len(df), 1))
        else:
            # Auto-infer: last column is gripper, the rest are joints
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
#  Feature Normalization
# ═══════════════════════════════════════════════════════════

def normalize_trajectories(
    trajectories: list[Trajectory],
    rot_type: str,
) -> tuple[list[Trajectory], dict]:
    """
    Min-max normalize trajectory features to [0, 1], making components with different units comparable.

    Normalization strategy:
      EEF mode (quaternion/euler) -> vector [x,y,z, qx,qy,qz,qw, grip]
        - position (dim 0-2): normalized
        - quaternion (dim 3-6): unchanged (geodesic has natural range [0, pi])
        - gripper (dim 7+): normalized

      Joint mode (none) -> vector [joints..., grip]
        - all dimensions: normalized

    Statistics are computed from global min/max across all trajectories.

    Returns
    -------
    normalized : list of normalized Trajectory objects
    stats : diagnostic info with {min, max, range} for each normalized dimension
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
#  RoboMindV2.0 Dynamic Config Loading
# ═══════════════════════════════════════════════════════════

def load_modality_config(robot_type_path: str, side: str) -> dict:
    """
    Load configuration for the specified side from modality.json.

    Parameters
    ----------
    robot_type_path : robot_type directory path (e.g., /path/to/RoboMindV2.0/agilex)
    side : 'left' or 'right'

    Returns
    -------
    config_dict : {
        "joint_column": str,
        "joint_indices": list[int],
        "effector_column": str,  # column name for gripper or hand
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

    # Read joint config
    joint_key = f"{side}_joint"
    if joint_key not in action_config:
        raise KeyError(f"'{joint_key}' not found in modality.json action config")

    joint_info = action_config[joint_key]
    joint_column = joint_info["original_key"]
    joint_indices = joint_info["indices"]

    # Read effector config (prefer hand, fallback to gripper)
    hand_key = f"{side}_hand"
    gripper_key = f"{side}_gripper"

    if hand_key in action_config:
        # Has dexterous hand
        effector_info = action_config[hand_key]
        effector_type = "hand"
        effector_key = hand_key
    elif gripper_key in action_config:
        # Has gripper
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
    Load filter_report.json and extract problematic episode lists for each sub-dataset.

    Parameters
    ----------
    filter_report_path : path to filter_report.json

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

    # RoboMindV2.0_filter_report.json format:
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
    Infer robot_type (first-level subdirectory) from a sub-dataset path.

    Example: /path/to/RoboMindV2.0/agilex/task1 -> 'agilex'
    """
    rel_path = os.path.relpath(sub_dataset_path, dataset_root)
    parts = rel_path.split(os.sep)
    if len(parts) >= 1:
        return parts[0]
    return ""


# ═══════════════════════════════════════════════════════════
#  Public API
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
    Scan dataset_root/data/chunk-*/episode_*.parquet,
    split by episode and build unified trajectory vectors.

    Parameters
    ----------
    dataset_root : root directory of a single (sub-)dataset (containing a data/ subdirectory)
    config : dataset configuration
    side : arm side to analyze
    max_episodes : maximum number of trajectories to load (None=all)
    exclude_episodes : set of episode IDs to exclude
    robot_type : robot model name (used for RoboMindV2.0 dynamic config loading)
    """
    if side not in config.arms:
        available = list(config.arms.keys())
        # If only 'single' is available, auto-map to it
        if "single" in config.arms:
            side = "single"
        else:
            raise ValueError(f"Side '{side}' not available. Choose from: {available}")

    arm = config.arms[side]

    # ── Dynamic config loading (RoboMindV2.0) ──
    if config.has_modality_json and robot_type:
        # Infer robot_type path
        robot_type_path = os.path.join(
            config.dataset_path, robot_type
        )

        if os.path.exists(os.path.join(robot_type_path, "modality.json")):
            print(f"  [INFO] Loading from modality.json: robot_type={robot_type}, side={side}")
            try:
                modality_cfg = load_modality_config(robot_type_path, side)

                # Dynamically build ArmConfig
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
                print(f"  [WARN] Failed to load modality.json: {e}, using default config")
                # Fallback to the original arm config
                arm = config.arms[side]

    pattern = os.path.join(dataset_root, "data", "chunk-*", "episode_*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files matching: {pattern}")

    trajectories: list[Trajectory] = []
    for fpath in files:
        df = pq.read_table(fpath).to_pandas()

        # Check required columns
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
    Parse task -> episode_indices mapping from meta/episodes.jsonl.

    Returns
    -------
    task_map : {task_name: [episode_index, ...]}
    chunk_size : chunk size read from meta/info.json
    """
    episodes_path = os.path.join(dataset_root, "meta", "episodes.jsonl")
    info_path = os.path.join(dataset_root, "meta", "info.json")

    with open(info_path) as f:
        info = json.load(f)
    chunk_size = info.get("chunks_size", 1000)

    # Try to build task_index -> task_name mapping from tasks.jsonl
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
            # Prefer the "tasks" field (string list, e.g. BC_Z / RT-1)
            task_names = ep.get("tasks", [])
            if not task_names and "task_index" in ep:
                # Look up task name via task_index (e.g. RH20T)
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
    Load trajectories by a given list of episode indices, locating parquet files by index.

    Parameters
    ----------
    dataset_root : dataset root directory (containing a data/ subdirectory)
    config : dataset configuration
    side : arm side to analyze
    episode_indices : list of episode indices to load
    chunk_size : number of episodes per chunk (read from info.json)
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
    Discover all sub-dataset paths under the dataset.
    Scans directories based on sub_dataset_depth.

    Returns
    -------
    sub_datasets : list of dict with keys:
        - 'path': full path to the sub-dataset
        - 'robot_type': robot model (only meaningful for depth>=2, e.g. RoboMindV2.0)
        - 'name': relative name of the sub-dataset
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
                        "robot_type": level1,  # level1 is the robot_type
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
                            "robot_type": level2,  # level2 may be the robot_type
                            "name": f"{level1}/{level2}/{level3}",
                        })
        return subs

    raise ValueError(f"Unsupported sub_dataset_depth: {depth}")
