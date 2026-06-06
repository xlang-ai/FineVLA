#!/usr/bin/env python3
"""
Unified sampling entry point: dispatch to each dataset's sampling logic by dataset_name.

Usage:
  python sample_all.py robomind_v1              # single dataset
  python sample_all.py robomind_v1 droid rdt    # multiple datasets
  python sample_all.py all                      # all 8 datasets
  python sample_all.py all --workers 64 --check-videos
  python sample_all.py droid --max-tasks 500
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

ROOT_DIR = Path(__file__).resolve().parent
SEED = 42

DATASET_DIRS: dict[str, str] = {
    "robomind_v1":  "RoboMindV1.0",
    "robomind_v2":  "RoboMindV2.0",
    "galaxea":      "Galaxea-Open-World-Dataset",
    "robocoin":     "RoboCOIN",
    "rh20t":        "RH20T-fjy",
    "rdt":          "RDT-yhq",
    "agibotworld":  "agibotworld_hyy",
    "droid":        "droid_1.0.1",
}

# ═════════════════════════════════════════════════════════════════════════════
#  Common Utilities
# ═════════════════════════════════════════════════════════════════════════════

def read_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def extract_views(features: dict, view_order: list[str] | None = None) -> list[str]:
    views = []
    for key, val in features.items():
        if key.startswith("observation.images.") and val.get("dtype") == "video":
            views.append(key.removeprefix("observation.images."))
    if view_order:
        def _sort_key(name: str) -> int:
            try:
                return view_order.index(name)
            except ValueError:
                return len(view_order)
        views.sort(key=_sort_key)
    else:
        views.sort()
    return views


def build_video_relpath(template: str, episode_index: int, chunks_size: int, video_key: str) -> str:
    chunk = episode_index // chunks_size
    return template.format(episode_chunk=chunk, video_key=video_key, episode_index=episode_index)


def build_video_relpath_default(episode_index: int, chunks_size: int, video_key: str) -> str:
    chunk = episode_index // chunks_size
    return f"videos/chunk-{chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def get_resolution(features: dict, views: list[str]) -> str | None:
    for view in views:
        feat = features.get(f"observation.images.{view}", {})
        fi = feat.get("info", {})
        h = fi.get("video.height") or (feat.get("shape", [None])[0])
        w = fi.get("video.width") or (feat.get("shape", [None, None])[1] if len(feat.get("shape", [])) > 1 else None)
        if h and w:
            return f"{w}x{h}"
    return None


def make_record(
    sample_id: str, dataset: str, dataset_dir: str,
    views: list[str], videos: dict,
    instruction: str | None, duration_sec: float | None,
    fps: int | None, resolution: str | None, robot_type: str,
    has_raw_steps: int = 0, step_source: str = "none",
    steps_raw: list | None = None, **extra,
) -> dict:
    rec = {
        "sample_id": sample_id,
        "dataset": dataset,
        "dataset_dir": dataset_dir,
        "views": views,
        "videos": videos,
        "instruction_raw": instruction,
        "duration_sec": duration_sec,
        "fps": fps,
        "resolution": resolution,
        "robot_type": robot_type,
        "has_raw_steps": has_raw_steps,
        "step_source": step_source,
        "steps_raw": steps_raw,
        "task_type": None,
        "object_categories": None,
        "review_level": None,
        "review_time_min": None,
        "review_outcome": None,
        "double_review": 0,
        "error_tags": [],
    }
    for k, v in extra.items():
        rec[k] = v
    return rec


def subdirs(parent: Path) -> list[str]:
    try:
        with os.scandir(parent) as it:
            return sorted(e.name for e in it if e.is_dir(follow_symlinks=False))
    except FileNotFoundError:
        return []


def get_instruction(tasks_path: Path, ep: dict) -> str | None:
    # Prefer the episode's own tasks field
    if ep.get("tasks"):
        return ep["tasks"][0] if ep["tasks"] else None

    # If the episode has no tasks field, look up from tasks.jsonl
    instruction = None
    if tasks_path.exists():
        items = read_jsonl(tasks_path)
        if items:
            # If episode has task_index, use it to find the corresponding task
            if "task_index" in ep:
                task_index = ep["task_index"]
                for item in items:
                    if item.get("task_index") == task_index:
                        instruction = item.get("task")
                        break
            else:
                # No task_index, use the first one
                instruction = items[0].get("task")

    return instruction


def get_video_paths(info: dict, views: list[str], episode_index: int) -> dict[str, str]:
    template = info.get("video_path", "")
    chunks_size = info.get("chunks_size", 1000)
    videos: dict[str, str] = {}
    for view in views:
        video_key = f"observation.images.{view}"
        if template:
            videos[view] = build_video_relpath(template, episode_index, chunks_size, video_key)
        else:
            videos[view] = build_video_relpath_default(episode_index, chunks_size, video_key)
    return videos


# ═════════════════════════════════════════════════════════════════════════════
#  RoboMind V1
# ═════════════════════════════════════════════════════════════════════════════
_V1_VIEW_ORDER = [
    "camera_front", "camera_top", "camera_left", "camera_right",
    "camera_left_wrist", "camera_right_wrist", "camera_wrist", "camera_overhead",
]
_V1_BENCHMARKS = ["benchmark1_0_compressed", "benchmark1_1_compressed", "benchmark1_2_compressed"]

_FRAME_NUM_RE = re.compile(r"_(\d+)\.jpg$")

def _v1_frame_num(s: str | None) -> int | None:
    if not s:
        return None
    m = _FRAME_NUM_RE.search(s)
    return int(m.group(1)) if m else None


def _v1_parse_steps(episode: dict):
    raw = episode.get("action_config_json") or episode.get("action_config")
    if raw is None:
        return None, 0, "none"
    if isinstance(raw, str):
        try:
            config = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None, 0, "none"
    elif isinstance(raw, dict):
        config = raw
    else:
        return None, 0, "none"
    steps = config.get("steps")
    if not steps or not isinstance(steps, list):
        return None, 0, "none"
    filtered = []
    for i, step in enumerate(steps):
        desc = step.get("step_description", "")
        if desc is None:
            continue
        desc = desc.strip()
        if desc in ("", "none", "[none]"):
            continue
        filtered.append({"i": i, "desc": desc,
                         "start": _v1_frame_num(step.get("start_frame")),
                         "end": _v1_frame_num(step.get("end_frame"))})
    if not filtered:
        return None, 0, "none"
    return filtered, 1, "action_config_json"


def _v1_process(bench, bench_id, robot_name, task_name, task_path, rng, check_videos):
    meta = task_path / "meta"
    info_path = meta / "info.json"
    if not info_path.exists():
        return None, dict(task=f"{bench_id}-{robot_name}-{task_name}", reason="missing info.json")
    info = json.load(open(info_path, encoding="utf-8"))
    eps_path = meta / "episodes.jsonl"
    if not eps_path.exists():
        return None, dict(task=f"{bench_id}-{robot_name}-{task_name}", reason="missing episodes.jsonl")
    episodes = read_jsonl(eps_path)
    if not episodes:
        return None, dict(task=f"{bench_id}-{robot_name}-{task_name}", reason="empty episodes")
    ep = rng.choice(episodes)
    ei = ep["episode_index"]
    instruction = get_instruction(meta / "tasks.jsonl", ep)
    features = info.get("features", {})
    views = extract_views(features, _V1_VIEW_ORDER)
    if not views:
        return None, dict(task=f"{bench_id}-{robot_name}-{task_name}", reason="no views")
    videos = get_video_paths(info, views, ei)
    if check_videos:
        missing = [v for v in views if not (task_path / videos[v]).is_file()]
        if len(missing) == len(views):
            return None, dict(task=f"{bench_id}-{robot_name}-{task_name}", reason="all videos missing")
    steps_raw, has_raw_steps, step_source = _v1_parse_steps(ep)
    ds_dir = str(task_path.relative_to(ROOT_DIR))
    return make_record(
        sample_id=f"robomindv1-{bench_id}-{robot_name}-{task_name}-{ei}",
        dataset="robomind_v1.0", dataset_dir=ds_dir, views=views, videos=videos,
        instruction=instruction,
        duration_sec=round(ep["length"] / info["fps"], 3) if ep.get("length") and info.get("fps") else None,
        fps=info.get("fps"), resolution=get_resolution(features, views),
        robot_type=info.get("robot_type", "unknown"),
        has_raw_steps=has_raw_steps, step_source=step_source, steps_raw=steps_raw,
    ), None


def collect_robomind_v1(base: Path, check_videos: bool):
    jobs = []
    for bench in _V1_BENCHMARKS:
        bid = bench.replace("benchmark1_", "").replace("_compressed", "")
        bench_path = base / bench
        for robot in subdirs(bench_path):
            for task in subdirs(bench_path / robot):
                jobs.append(dict(
                    fn=_v1_process, bench=bench, bench_id=bid,
                    robot_name=robot, task_name=task,
                    task_path=bench_path / robot / task, check_videos=check_videos,
                ))
    return jobs


# ═════════════════════════════════════════════════════════════════════════════
#  RoboMind V2
# ═════════════════════════════════════════════════════════════════════════════
_V2_VIEW_ORDER = [
    "camera_front", "camera_top", "camera_head",
    "camera_left", "camera_right",
    "camera_wrist_left", "camera_wrist_right",
    "camera_left_wrist", "camera_right_wrist",
    "camera_wrist", "camera_overhead",
]


def _v2_process(robot_name, task_name, task_path, rng, check_videos, **_):
    meta = task_path / "meta"
    info_path = meta / "info.json"
    if not info_path.exists():
        return None, dict(task=f"{robot_name}-{task_name}", reason="missing info.json")
    info = json.load(open(info_path, encoding="utf-8"))
    eps_path = meta / "episodes.jsonl"
    if not eps_path.exists():
        return None, dict(task=f"{robot_name}-{task_name}", reason="missing episodes.jsonl")
    episodes = read_jsonl(eps_path)
    if not episodes:
        return None, dict(task=f"{robot_name}-{task_name}", reason="empty episodes")
    ep = rng.choice(episodes)
    ei = ep["episode_index"]
    instruction = get_instruction(meta / "tasks.jsonl", ep)
    features = info.get("features", {})
    views = extract_views(features, _V2_VIEW_ORDER)
    if not views:
        return None, dict(task=f"{robot_name}-{task_name}", reason="no views")
    videos = get_video_paths(info, views, ei)
    if check_videos:
        missing = [v for v in views if not (task_path / videos[v]).is_file()]
        if len(missing) == len(views):
            return None, dict(task=f"{robot_name}-{task_name}", reason="all videos missing")
    ds_dir = str(task_path.relative_to(ROOT_DIR))
    return make_record(
        sample_id=f"robomindv2-{robot_name}-{task_name}-{ei}",
        dataset="robomind_v2.0", dataset_dir=ds_dir, views=views, videos=videos,
        instruction=instruction,
        duration_sec=round(ep["length"] / info["fps"], 3) if ep.get("length") and info.get("fps") else None,
        fps=info.get("fps"), resolution=get_resolution(features, views),
        robot_type=info.get("robot_type", "unknown"),
    ), None


def collect_robomind_v2(base: Path, check_videos: bool):
    jobs = []
    for robot in subdirs(base):
        rp = base / robot
        for task in subdirs(rp):
            jobs.append(dict(fn=_v2_process, robot_name=robot, task_name=task,
                             task_path=rp / task, check_videos=check_videos))
    return jobs


# ═════════════════════════════════════════════════════════════════════════════
#  Galaxea
# ═════════════════════════════════════════════════════════════════════════════
_GAL_VIEW_ORDER = ["head_rgb", "head_right_rgb", "left_wrist_rgb", "right_wrist_rgb"]
_GAL_QUALITY_LABELS = {"qualified", "unqualified", "null", "None", "none", ""}


def _gal_extract_steps(parquet_path: Path, task_map: dict[int, str]):
    import pandas as pd
    df = pd.read_parquet(parquet_path,
                         columns=["frame_index", "task_index", "coarse_task_index",
                                  "quality_index", "coarse_quality_index"])
    df = df.sort_values("frame_index").reset_index(drop=True)
    coarse_idx = int(df["coarse_task_index"].iloc[0])
    coarse_text = task_map.get(coarse_idx)
    quality_idx = int(df["quality_index"].iloc[0])
    quality_text = task_map.get(quality_idx)
    segments, prev_ti, seg_start, prev_fi = [], None, 0, 0
    for _, row in df.iterrows():
        ti, fi = int(row["task_index"]), int(row["frame_index"])
        if ti != prev_ti:
            if prev_ti is not None:
                segments.append({"task_index": prev_ti, "start": seg_start, "end": prev_fi})
            seg_start = fi
            prev_ti = ti
        prev_fi = fi
    if prev_ti is not None:
        segments.append({"task_index": prev_ti, "start": seg_start, "end": prev_fi})
    steps_raw = []
    si = 0
    for seg in segments:
        desc = task_map.get(seg["task_index"], "")
        if desc.strip().lower() in _GAL_QUALITY_LABELS:
            continue
        steps_raw.append({"i": si, "desc": desc, "start": seg["start"], "end": seg["end"]})
        si += 1
    return steps_raw or None, coarse_text, quality_text


def _gal_process(task_name, task_path, rng, check_videos, **_):
    meta = task_path / "meta"
    info_path = meta / "info.json"
    if not info_path.exists():
        return None, dict(task=task_name, reason="missing info.json")
    info = json.load(open(info_path, encoding="utf-8"))
    eps_path = meta / "episodes.jsonl"
    if not eps_path.exists():
        return None, dict(task=task_name, reason="missing episodes.jsonl")
    episodes = read_jsonl(eps_path)
    if not episodes:
        return None, dict(task=task_name, reason="empty episodes")
    ep = rng.choice(episodes)
    ei = ep["episode_index"]
    task_map: dict[int, str] = {}
    tp = meta / "tasks.jsonl"
    if tp.exists():
        for item in read_jsonl(tp):
            task_map[item["task_index"]] = item["task"]
    chunks_size = info.get("chunks_size", 1000)
    data_tpl = info.get("data_path", "")
    chunk = ei // chunks_size
    pq_rel = data_tpl.format(episode_chunk=chunk, episode_index=ei)
    pq_path = task_path / pq_rel
    steps_raw, coarse_text, quality_text = None, None, None
    if pq_path.exists():
        try:
            steps_raw, coarse_text, quality_text = _gal_extract_steps(pq_path, task_map)
        except Exception as e:
            return None, dict(task=task_name, reason=f"parquet error: {e}")
    else:
        return None, dict(task=task_name, reason=f"parquet not found: {pq_rel}")
    instruction = coarse_text
    if not instruction or instruction.strip().lower() in _GAL_QUALITY_LABELS:
        for t in ep.get("tasks", []):
            if t.strip().lower() not in _GAL_QUALITY_LABELS:
                instruction = t
                break
    if instruction and instruction.strip().lower() in _GAL_QUALITY_LABELS:
        instruction = None
    features = info.get("features", {})
    views = extract_views(features, _GAL_VIEW_ORDER)
    if not views:
        return None, dict(task=task_name, reason="no views")
    videos = get_video_paths(info, views, ei)
    if check_videos:
        missing = [v for v in views if not (task_path / videos[v]).is_file()]
        if len(missing) == len(views):
            return None, dict(task=task_name, reason="all videos missing")
    ds_dir = str(task_path.relative_to(ROOT_DIR))
    return make_record(
        sample_id=f"galaxea-{task_name}-{ei}",
        dataset="galaxea_open_world", dataset_dir=ds_dir, views=views, videos=videos,
        instruction=instruction,
        duration_sec=round(ep["length"] / info["fps"], 3) if ep.get("length") and info.get("fps") else None,
        fps=info.get("fps"), resolution=get_resolution(features, views),
        robot_type=info.get("robot_type", "unknown"),
        has_raw_steps=1 if steps_raw else 0,
        step_source="parquet_task_index" if steps_raw else "none",
        steps_raw=steps_raw,
    ), None


def collect_galaxea(base: Path, check_videos: bool):
    jobs = []
    for d in sorted(os.listdir(base)):
        if (base / d / "meta").is_dir():
            jobs.append(dict(fn=_gal_process, task_name=d, task_path=base / d, check_videos=check_videos))
    return jobs


# ═════════════════════════════════════════════════════════════════════════════
#  RoboCOIN / RDT -- same "per-task-dir, no steps" pattern
# ═════════════════════════════════════════════════════════════════════════════
_COIN_VIEW_ORDER = [
    "cam_high_rgb", "cam_front_rgb", "cam_high_left_rgb", "cam_high_right_rgb",
    "cam_high_center_fisheye_rgb", "cam_high_left_fisheye_rgb", "cam_high_right_fisheye_rgb",
    "cam_back_left_fisheye_rgb", "cam_back_right_fisheye_rgb", "cam_third_view",
    "cam_chest_rgb", "cam_head_rgb", "camera_head_rgb",
    "cam_left_wrist_rgb", "cam_right_wrist_rgb",
    "cam_left_wrist_rgb_rgb", "cam_right_wrist_rgb_rgb",
    "camera_left_wrist_rgb", "camera_right_wrist_rgb",
    "color_left_wrist", "color_right_wrist",
]
_RDT_VIEW_ORDER = ["main", "left_wrist", "right_wrist"]


def _simple_task_process(dataset_label, id_prefix, view_order,
                         task_name, task_path, rng, check_videos, **_):
    meta = task_path / "meta"
    info_path = meta / "info.json"
    if not info_path.exists():
        return None, dict(task=task_name, reason="missing info.json")
    info = json.load(open(info_path, encoding="utf-8"))
    eps_path = meta / "episodes.jsonl"
    if not eps_path.exists():
        return None, dict(task=task_name, reason="missing episodes.jsonl")
    episodes = read_jsonl(eps_path)
    if not episodes:
        return None, dict(task=task_name, reason="empty episodes")
    ep = rng.choice(episodes)
    ei = ep["episode_index"]
    instruction = get_instruction(meta / "tasks.jsonl", ep)
    features = info.get("features", {})
    views = extract_views(features, view_order)
    if not views:
        return None, dict(task=task_name, reason="no views")
    videos = get_video_paths(info, views, ei)
    if check_videos:
        missing = [v for v in views if not (task_path / videos[v]).is_file()]
        if len(missing) == len(views):
            return None, dict(task=task_name, reason="all videos missing")
    ds_dir = str(task_path.relative_to(ROOT_DIR))
    return make_record(
        sample_id=f"{id_prefix}-{task_name}-{ei}",
        dataset=dataset_label, dataset_dir=ds_dir, views=views, videos=videos,
        instruction=instruction,
        duration_sec=round(ep["length"] / info["fps"], 3) if ep.get("length") and info.get("fps") else None,
        fps=info.get("fps"), resolution=get_resolution(features, views),
        robot_type=info.get("robot_type", "unknown"),
    ), None


def collect_simple_task_dirs(base: Path, check_videos: bool, dataset_label: str,
                             id_prefix: str, view_order: list[str],
                             dir_filter=None):
    jobs = []
    for d in sorted(os.listdir(base)):
        if not (base / d / "meta").is_dir():
            continue
        if dir_filter and not dir_filter(d):
            continue
        jobs.append(dict(
            fn=_simple_task_process,
            dataset_label=dataset_label, id_prefix=id_prefix, view_order=view_order,
            task_name=d, task_path=base / d, check_videos=check_videos,
        ))
    return jobs


# ═════════════════════════════════════════════════════════════════════════════
#  AgiBotWorld — per-task-dir with action_config steps + exclude_episodes
# ═════════════════════════════════════════════════════════════════════════════
_ABI_VIEW_ORDER = [
    "head", "head_center_fisheye", "head_left_fisheye", "head_right_fisheye",
    "hand_left", "hand_right", "hand_left_fisheye", "hand_right_fisheye",
    "back_left_fisheye", "back_right_fisheye",
    "left_sensor_1", "left_sensor_2", "right_sensor_1", "right_sensor_2",
]


def _abi_parse_steps(ac) -> list[dict] | None:
    if not ac:
        return None
    steps_list = ac if isinstance(ac, list) else None
    if steps_list is None:
        if isinstance(ac, str):
            try:
                steps_list = json.loads(ac)
            except (json.JSONDecodeError, TypeError):
                return None
    if not steps_list or not isinstance(steps_list, list):
        return None
    result = []
    for idx, step in enumerate(steps_list):
        if not isinstance(step, dict):
            continue
        desc = step.get("action_text", "")
        if not desc or desc.lower() in ("none", "[none]", ""):
            continue
        result.append({"i": idx, "desc": desc,
                       "start": step.get("start_frame"), "end": step.get("end_frame")})
    return result if result else None


def _abi_process(task_name, task_path, rng, check_videos, **_):
    meta = task_path / "meta"
    info_path = meta / "info.json"
    if not info_path.exists():
        return None, dict(task=task_name, reason="missing info.json")
    info = json.load(open(info_path, encoding="utf-8"))
    eps_path = meta / "episodes.jsonl"
    if not eps_path.exists():
        return None, dict(task=task_name, reason="missing episodes.jsonl")
    episodes = read_jsonl(eps_path)
    if not episodes:
        return None, dict(task=task_name, reason="empty episodes")
    excluded_set: set[int] = set()
    exc_path = meta / "exclude_episodes.json"
    if exc_path.exists():
        exc = json.load(open(exc_path, encoding="utf-8"))
        excluded_set = set(exc.get("excluded_episodes", []))
    valid = [e for e in episodes if e["episode_index"] not in excluded_set]
    if not valid:
        return None, dict(task=task_name, reason="all episodes excluded")
    ep = rng.choice(valid)
    ei = ep["episode_index"]
    instruction = get_instruction(meta / "tasks.jsonl", ep)
    features = info.get("features", {})
    views = extract_views(features, _ABI_VIEW_ORDER)
    if not views:
        return None, dict(task=task_name, reason="no views")
    videos = get_video_paths(info, views, ei)
    if check_videos:
        missing = [v for v in views if not (task_path / videos[v]).is_file()]
        if len(missing) == len(views):
            return None, dict(task=task_name, reason="all videos missing")
    steps_raw = _abi_parse_steps(ep.get("action_config"))
    ds_dir = str(task_path.relative_to(ROOT_DIR))
    return make_record(
        sample_id=f"agibotworld-{task_name}-{ei}",
        dataset="agibotworld", dataset_dir=ds_dir, views=views, videos=videos,
        instruction=instruction,
        duration_sec=round(ep["length"] / info["fps"], 3) if ep.get("length") and info.get("fps") else None,
        fps=info.get("fps"), resolution=get_resolution(features, views),
        robot_type=info.get("robot_type", "unknown"),
        has_raw_steps=1 if steps_raw else 0,
        step_source="action_config" if steps_raw else "none",
        steps_raw=steps_raw,
    ), None


def collect_agibotworld(base: Path, check_videos: bool):
    jobs = []
    for d in sorted(os.listdir(base)):
        if d.startswith("task_") and (base / d / "meta").is_dir():
            jobs.append(dict(fn=_abi_process, task_name=d, task_path=base / d, check_videos=check_videos))
    return jobs


# ═════════════════════════════════════════════════════════════════════════════
#  RH20T — cfg-based, task_index grouping
# ═════════════════════════════════════════════════════════════════════════════

def _rh20t_process(cfg_name, cfg_path, task_index, task_text, episodes, info, rng, check_videos, **_):
    ep = rng.choice(episodes)
    ei = ep["episode_index"]
    features = info.get("features", {})
    views = extract_views(features)
    if not views:
        return None, dict(cfg=cfg_name, task_index=task_index, reason="no views")
    chunk = ei // 1000
    videos = {v: f"videos/chunk-{chunk:03d}/observation.images.{v}/episode_{ei:06d}.mp4" for v in views}
    if check_videos:
        missing = [v for v in views if not (cfg_path / videos[v]).is_file()]
        if len(missing) == len(views):
            return None, dict(cfg=cfg_name, task_index=task_index, reason="all videos missing")
    resolution = None
    for v in views:
        shape = features.get(f"observation.images.{v}", {}).get("shape")
        if shape and len(shape) >= 2:
            resolution = f"{shape[1]}x{shape[0]}"
            break
    ds_dir = str(cfg_path.relative_to(ROOT_DIR))
    return make_record(
        sample_id=f"rh20t-{cfg_name}-{task_index}-{ei}",
        dataset="rh20t", dataset_dir=ds_dir, views=views, videos=videos,
        instruction=task_text,
        duration_sec=round(ep["length"] / info["fps"], 3) if ep.get("length") and info.get("fps") else None,
        fps=info.get("fps"), resolution=resolution,
        robot_type=info.get("robot_type", "unknown"),
    ), None


def collect_rh20t(base: Path, check_videos: bool):
    jobs, skipped = [], []
    cfgs = sorted(d for d in os.listdir(base) if d.startswith("rh20t_cfg") and (base / d / "meta").is_dir())
    for cfg_name in cfgs:
        cfg_path = base / cfg_name
        info_p = cfg_path / "meta" / "info.json"
        if not info_p.exists():
            skipped.append(dict(cfg=cfg_name, reason="missing info.json"))
            continue
        info = json.load(open(info_p, encoding="utf-8"))
        task_map: dict[int, str] = {}
        tp = cfg_path / "meta" / "tasks.jsonl"
        if tp.exists():
            for item in read_jsonl(tp):
                task_map[item["task_index"]] = item.get("task", "")
        eps_by_task: dict[int, list] = defaultdict(list)
        ep_p = cfg_path / "meta" / "episodes.jsonl"
        if ep_p.exists():
            for ep in read_jsonl(ep_p):
                eps_by_task[ep["task_index"]].append(ep)
        for ti in sorted(task_map.keys()):
            eps = eps_by_task.get(ti, [])
            if not eps:
                skipped.append(dict(cfg=cfg_name, task_index=ti, reason="no episodes"))
                continue
            jobs.append(dict(fn=_rh20t_process, cfg_name=cfg_name, cfg_path=cfg_path,
                             task_index=ti, task_text=task_map[ti], episodes=eps, info=info,
                             check_videos=check_videos))
    return jobs, skipped


# ═════════════════════════════════════════════════════════════════════════════
#  DROID — single flat dataset
# ═════════════════════════════════════════════════════════════════════════════
_DROID_VIEW_ORDER = ["exterior_1_left", "exterior_2_left", "wrist_left"]


def _droid_process(task_text, episodes, info, rng, check_videos, base_dir, **_):
    ep = rng.choice(episodes)
    ei = ep["episode_index"]
    features = info.get("features", {})
    views = extract_views(features, _DROID_VIEW_ORDER)
    if not views:
        return None, dict(task=task_text[:80], reason="no views")
    videos = get_video_paths(info, views, ei)
    if check_videos:
        missing = [v for v in views if not (base_dir / videos[v]).is_file()]
        if len(missing) == len(views):
            return None, dict(task=task_text[:80], reason="all videos missing")
    ds_dir = str(base_dir.relative_to(ROOT_DIR))
    return make_record(
        sample_id=f"droid-{ei}",
        dataset="droid_1.0.1", dataset_dir=ds_dir, views=views, videos=videos,
        instruction=task_text,
        duration_sec=round(ep["length"] / info["fps"], 3) if ep.get("length") and info.get("fps") else None,
        fps=info.get("fps"), resolution=get_resolution(features, views),
        robot_type=info.get("robot_type", "unknown"),
    ), None


def collect_droid(base: Path, check_videos: bool, max_tasks: int = 500):
    info = json.load(open(base / "meta" / "info.json", encoding="utf-8"))
    all_eps = read_jsonl(base / "meta" / "episodes.jsonl")
    task_to_eps: dict[str, list] = defaultdict(list)
    skipped_empty = 0
    for ep in all_eps:
        t = (ep.get("tasks") or [""])[0]
        if not t.strip():
            skipped_empty += 1
            continue
        task_to_eps[t].append(ep)
    texts = sorted(task_to_eps.keys())
    if max_tasks and max_tasks < len(texts):
        texts = random.Random(SEED).sample(texts, max_tasks)
        texts.sort()
    print(f"  DROID: {len(all_eps)} episodes, {len(task_to_eps)} unique tasks, "
          f"selected {len(texts)}, skipped {skipped_empty} empty")
    jobs = []
    for t in texts:
        jobs.append(dict(fn=_droid_process, task_text=t, episodes=task_to_eps[t],
                         info=info, check_videos=check_videos, base_dir=base))
    return jobs


# ═════════════════════════════════════════════════════════════════════════════
#  Unified Dispatcher
# ═════════════════════════════════════════════════════════════════════════════

def collect_jobs(dataset_name: str, check_videos: bool, max_tasks: int) -> tuple[list[dict], list[dict]]:
    base = ROOT_DIR / DATASET_DIRS[dataset_name]
    if not base.is_dir():
        raise ValueError(f"Dataset directory does not exist: {base}")

    extra_skipped: list[dict] = []

    if dataset_name == "robomind_v1":
        jobs = collect_robomind_v1(base, check_videos)
    elif dataset_name == "robomind_v2":
        jobs = collect_robomind_v2(base, check_videos)
    elif dataset_name == "galaxea":
        jobs = collect_galaxea(base, check_videos)
    elif dataset_name == "robocoin":
        jobs = collect_simple_task_dirs(base, check_videos, "robocoin", "robocoin", _COIN_VIEW_ORDER)
    elif dataset_name == "rdt":
        jobs = collect_simple_task_dirs(base, check_videos, "rdt", "rdt", _RDT_VIEW_ORDER)
    elif dataset_name == "agibotworld":
        jobs = collect_agibotworld(base, check_videos)
    elif dataset_name == "rh20t":
        jobs, extra_skipped = collect_rh20t(base, check_videos)
    elif dataset_name == "droid":
        jobs = collect_droid(base, check_videos, max_tasks)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name!r}. Available: {', '.join(sorted(DATASET_DIRS))}")

    return jobs, extra_skipped


def run_one_job(job: dict, rng: random.Random):
    fn = job.pop("fn")
    return fn(rng=rng, **job)


def run_sampling(
    dataset_name: str,
    workers: int = 32,
    check_videos: bool = False,
    max_tasks: int = 500,
) -> tuple[list[dict], list[dict]]:
    print(f"\n{'━' * 60}")
    print(f"Sampling [{dataset_name}] -> {DATASET_DIRS[dataset_name]}/")
    print(f"{'━' * 60}")

    jobs, skipped = collect_jobs(dataset_name, check_videos, max_tasks)
    total = len(jobs)
    if total == 0:
        print(f"  No jobs to sample")
        return [], skipped

    print(f"  Total {total} sampling jobs (workers={workers})")

    master_rng = random.Random(SEED)
    seeds = [master_rng.randint(0, 2**63) for _ in range(total)]

    use_tqdm = tqdm is not None
    progress = tqdm(total=total, desc=f"  {dataset_name}", unit="task") if use_tqdm else None
    done_count = 0

    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        fmap = {}
        for idx, job in enumerate(jobs):
            rng = random.Random(seeds[idx])
            fut = pool.submit(run_one_job, dict(job), rng)
            fmap[fut] = idx

        idx_results: dict[int, tuple] = {}
        for fut in as_completed(fmap):
            i = fmap[fut]
            try:
                idx_results[i] = fut.result()
            except Exception as e:
                idx_results[i] = (None, dict(reason=str(e)))
            if progress:
                progress.update(1)
            else:
                done_count += 1
                if done_count % 100 == 0 or done_count == total:
                    print(f"    [{done_count}/{total}]")

    if progress:
        progress.close()

    for i in range(total):
        record, skip = idx_results[i]
        if record:
            results.append(record)
        if skip:
            skipped.append(skip)

    print(f"  → {len(results)} sampled, {len(skipped)} skipped")
    return results, skipped


# ═════════════════════════════════════════════════════════════════════════════
#  main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Unified sampling entry point",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "datasets", nargs="+",
        help="Dataset names. Available:\n  " + "\n  ".join(f"{k:15s} -> {v}" for k, v in DATASET_DIRS.items()) + "\n  all -> all datasets",
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--check-videos", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=500, help="Max DROID sampling tasks (0=all)")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (default: script directory)")
    args = parser.parse_args()

    if "all" in args.datasets:
        dataset_names = list(DATASET_DIRS.keys())
    else:
        for d in args.datasets:
            if d not in DATASET_DIRS:
                raise ValueError(f"Unknown dataset: {d!r}. Available: {', '.join(sorted(DATASET_DIRS))}, all")
        dataset_names = args.datasets

    out_dir = Path(args.out_dir) if args.out_dir else ROOT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    all_skipped: list[dict] = []

    for ds in dataset_names:
        results, skipped = run_sampling(ds, args.workers, args.check_videos, args.max_tasks)
        all_results.extend(results)
        all_skipped.extend(skipped)

    # ── Write JSONL ──
    out_jsonl = out_dir / "all_samples.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in all_results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n[OK] {out_jsonl}  ({len(all_results)} rows)")

    # ── Write CSV ──
    out_csv = out_dir / "all_samples_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "dataset", "robot_type", "fps", "resolution",
                         "views_count", "has_raw_steps", "duration_sec", "instruction_preview"])
        for rec in all_results:
            writer.writerow([
                rec["sample_id"], rec["dataset"], rec["robot_type"],
                rec["fps"], rec["resolution"], len(rec.get("views", [])),
                rec["has_raw_steps"], rec["duration_sec"],
                (rec.get("instruction_raw") or "")[:80],
            ])
    print(f"[OK] {out_csv}")

    # ── Write skipped ──
    out_skip = out_dir / "all_skipped_tasks.json"
    with open(out_skip, "w", encoding="utf-8") as f:
        json.dump(all_skipped, f, indent=2, ensure_ascii=False)
    print(f"[OK] {out_skip}  ({len(all_skipped)} entries)")

    # ── Terminal summary ──
    print(f"\n{'═' * 60}")
    print(f"All done: {len(all_results)} sampled, {len(all_skipped)} skipped")
    ds_cnt = Counter(r["dataset"] for r in all_results)
    rt_cnt = Counter(r["robot_type"] for r in all_results)
    print(f"\nBy dataset:")
    for k, v in ds_cnt.most_common():
        print(f"  {k:25s} {v}")
    print(f"\nBy robot_type:")
    for k, v in rt_cnt.most_common():
        print(f"  {k:25s} {v}")


if __name__ == "__main__":
    main()
