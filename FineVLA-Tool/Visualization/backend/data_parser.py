"""
Path parsing, metadata reading, and parquet data extraction for LeRobot v2.1 datasets.
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from dataset_config import match_dataset_config


def parse_parquet_path(parquet_path: str) -> dict:
    """Parse a parquet file path into dataset_root, chunk_number, episode_number, etc.

    Expected path structure:
      .../DatasetFamily/SubDataset/data/chunk-XXX/episode_YYYYYY.parquet
    """
    p = Path(parquet_path)
    if not p.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    filename = p.name
    ep_match = re.match(r"episode_(\d+)\.parquet", filename)
    if not ep_match:
        raise ValueError(f"Invalid parquet filename format: {filename}")
    episode_number = int(ep_match.group(1))

    chunk_dir = p.parent.name
    chunk_match = re.match(r"chunk-(\d+)", chunk_dir)
    if not chunk_match:
        raise ValueError(f"Invalid chunk directory format: {chunk_dir}")
    chunk_number = int(chunk_match.group(1))

    # dataset_root is two levels up from the parquet file (past data/chunk-XXX/)
    dataset_root = str(p.parent.parent.parent)

    family, config = match_dataset_config(parquet_path)

    return {
        "parquet_path": parquet_path,
        "dataset_root": dataset_root,
        "chunk_number": chunk_number,
        "episode_number": episode_number,
        "dataset_family": family,
        "config": config,
    }


def load_info_json(dataset_root: str) -> dict:
    """Load and return the dataset's meta/info.json."""
    info_path = os.path.join(dataset_root, "meta", "info.json")
    with open(info_path, "r") as f:
        return json.load(f)


def discover_video_keys(info: dict) -> list[str]:
    """Extract all video feature keys from info.json (dtype == 'video')."""
    return [
        key for key, feat in info.get("features", {}).items()
        if feat.get("dtype") == "video"
    ]


def build_video_paths(dataset_root: str, chunk_number: int, episode_number: int, video_keys: list[str]) -> dict:
    """Build absolute video file paths for each video key.

    Returns {video_key: absolute_path} for files that exist.
    """
    paths = {}
    for vk in video_keys:
        video_path = os.path.join(
            dataset_root, "videos",
            f"chunk-{chunk_number:03d}",
            vk,
            f"episode_{episode_number:06d}.mp4"
        )
        if os.path.exists(video_path):
            paths[vk] = video_path
    return paths


def load_tasks_map(dataset_root: str) -> dict:
    """Load tasks.jsonl and return {task_index: task_description}."""
    tasks_path = os.path.join(dataset_root, "meta", "tasks.jsonl")
    tasks_map = {}
    if os.path.exists(tasks_path):
        with open(tasks_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    tasks_map[obj["task_index"]] = obj["task"]
    return tasks_map


def load_episode_stats(dataset_root: str, episode_number: int) -> dict:
    """Load stats for a specific episode from episodes_stats.jsonl.

    Only parses the target line to avoid loading the entire file.
    """
    stats_path = os.path.join(dataset_root, "meta", "episodes_stats.jsonl")
    with open(stats_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("episode_index") == episode_number:
                return obj.get("stats", {})
    return {}


META_FIELDS = frozenset({
    "timestamp", "frame_index", "episode_index", "index",
    "task_index", "coarse_task_index", "quality_index", "coarse_quality_index",
})


def auto_discover_fields(info: dict) -> tuple[list[str], list[str]]:
    """Auto-discover state and action fields from info.json features.

    Handles all naming conventions across datasets:
      - observation.state.* / action.* (Galaxea, droid)
      - observation.states.* / actions.* (RoboMind, agibotworld)
      - observation.state / action (Bridge, RT-1, BC_Z, RDT, etc.)
      - observation.force (RH20T)
      - eef_*_state / eef_*_action (RoboCOIN)
    """
    state_fields = []
    action_fields = []
    for key, feat in info.get("features", {}).items():
        if feat.get("dtype") == "video":
            continue
        if key in META_FIELDS:
            continue

        if key.startswith("observation."):
            state_fields.append(key)
        elif key.startswith("action"):
            action_fields.append(key)
        elif "_state" in key:
            state_fields.append(key)
        elif "_action" in key:
            action_fields.append(key)

    return state_fields, action_fields


def get_field_dim_names(info: dict, field_name: str) -> list[str]:
    """Get dimension names for a field from info.json features.

    Handles two formats:
      - list: ["joint0", "joint1", ...]
      - dict: {"motors": ["x", "y", "z", ...]}
    """
    feat = info.get("features", {}).get(field_name, {})
    names = feat.get("names")

    if names is None:
        shape = feat.get("shape", [1])
        dim = shape[0] if shape else 1
        return [f"dim_{i}" for i in range(dim)]

    if isinstance(names, list):
        short_names = []
        for n in names:
            if isinstance(n, str):
                parts = n.split(".")
                short_names.append(parts[-1] if len(parts) > 1 else n)
            else:
                short_names.append(str(n))
        return short_names

    if isinstance(names, dict):
        for v in names.values():
            if isinstance(v, list):
                return [str(x) for x in v]

    shape = feat.get("shape", [1])
    dim = shape[0] if shape else 1
    return [f"dim_{i}" for i in range(dim)]


def compute_y_range(stats: dict, field_name: str) -> tuple[float, float]:
    """Compute Y-axis range from episode stats for a given field.

    For list fields, take min of all min values and max of all max values.
    """
    field_stats = stats.get(field_name, {})
    mins = field_stats.get("min", [0])
    maxs = field_stats.get("max", [1])

    if isinstance(mins, list):
        y_min = min(float(v) for v in mins)
    else:
        y_min = float(mins)

    if isinstance(maxs, list):
        y_max = max(float(v) for v in maxs)
    else:
        y_max = float(maxs)

    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5

    margin = (y_max - y_min) * 0.05
    return y_min - margin, y_max + margin


def load_parquet_data(parquet_path: str, state_fields: list[str], action_fields: list[str]) -> dict:
    """Load parquet file and extract state/action/meta data as Python lists."""
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    total_frames = len(df)

    result = {
        "total_frames": total_frames,
        "frame_tasks": [],
        "state_data": {},
        "action_data": {},
    }

    if "task_index" in df.columns:
        result["frame_tasks"] = df["task_index"].tolist()

    all_fields = [("state", f) for f in state_fields] + [("action", f) for f in action_fields]

    for category, field in all_fields:
        if field not in df.columns:
            continue
        col = df[field]
        sample = col.iloc[0] if len(col) > 0 else None

        is_array = isinstance(sample, (list, tuple, np.ndarray))
        if is_array:
            values = []
            for row in col:
                if isinstance(row, np.ndarray):
                    values.append(row.tolist())
                elif isinstance(row, (list, tuple)):
                    values.append([float(x) for x in row])
                else:
                    values.append([float(row)])
        else:
            values = [[float(v)] for v in col]

        target = "state_data" if category == "state" else "action_data"
        result[target][field] = values

    return result
