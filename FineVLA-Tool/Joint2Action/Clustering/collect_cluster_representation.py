#!/usr/bin/env python3
"""
从 results_two_stage / results_by_task 中收集所有子数据集的 cluster_representation，
对每个 cluster 的代表性 episode 读取完整元信息（视频路径、instruction、duration、steps 等），
生成与 all_samples.jsonl 相同格式的记录。

输出为扁平 JSON 数组 [...] 。

用法:
    python collect_cluster_representation.py \
        --results_root ./results_two_stage \
        --datasets Galaxea RDT \
        --output cluster_representation_summary.json

python collect_cluster_representation.py --results_root ./results_by_task --datasets BC_Z --output cluster_representation_summary_BC_Z.json --workers 32
python collect_cluster_representation.py --results_root ./results_by_task --datasets RT-1 --output cluster_representation_summary_RT-1.json --workers 32
python collect_cluster_representation.py --results_root ./results_by_task --datasets RH20T-RoboInter --output cluster_representation_summary_RH20T-RoboInter.json
python collect_cluster_representation.py --results_root ./results_by_task --datasets droid_RoboInter --output cluster_representation_summary_droid_RoboInter.json
python collect_cluster_representation.py --results_root ./results_two_stage --datasets Galaxea --output cluster_representation_summary_Galaxea.json 
python collect_cluster_representation.py --results_root ./results_two_stage --datasets RoboCOIN --output cluster_representation_summary_RoboCOIN.json --workers 32
python collect_cluster_representation.py --results_root ./results_two_stage --datasets RoboCOIN_add0130 --output cluster_representation_summary_RoboCOIN_add0130.json --workers 32
python collect_cluster_representation.py --results_root ./results_two_stage --datasets RoboCOIN_add1201 --output cluster_representation_summary_RoboCOIN_add1201.json --workers 32
python collect_cluster_representation.py --results_root ./results_two_stage --datasets RoboMindV1.0 --output cluster_representation_summary_RoboMindV1.0.json --workers 64
python collect_cluster_representation.py --results_root ./results_robomind_v2 --datasets RoboMindV2.0 --output cluster_representation_summary_RoboMindV2.0.json


# 开启并行跑Galaxea
  python collect_cluster_representation.py \
      --results_root ./results_two_stage \
      --datasets Galaxea \
      --output cluster_representation_summary_Galaxea.json \
      --workers 16


python collect_cluster_representation.py \
    --results_root ./results_by_task \
    --datasets RH20T\
    --output cluster_representation_summary_RoboMindV1.0.json \
    --workers 16

"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from utils.sample_all import (
    read_jsonl,
    extract_views,
    get_video_paths,
    get_resolution,
    get_instruction,
    make_record,
    _gal_extract_steps,
    _GAL_QUALITY_LABELS,
    _v1_parse_steps,
    _abi_parse_steps,
)

_DATA_ROOT = "/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21"
ROOT_DIR = Path(_DATA_ROOT)

# config.py dataset_name → (sample_all 的 dataset_label, id_prefix, 处理类型)
DATASET_INFO: dict[str, dict] = {
    "Galaxea":         {"label": "galaxea_open_world", "prefix": "galaxea",      "type": "galaxea"},
    "RDT":             {"label": "rdt",                "prefix": "rdt",           "type": "simple"},
    "RoboCOIN":        {"label": "robocoin",           "prefix": "robocoin",      "type": "robocoin"},
    "RoboCOIN_add0130":{"label": "robocoin",           "prefix": "robocoin",      "type": "robocoin"},
    "RoboCOIN_add1201":{"label": "robocoin",           "prefix": "robocoin",      "type": "robocoin"},
    "Bridge":          {"label": "bridge",             "prefix": "bridge",        "type": "simple"},
    "RT-1":            {"label": "rt1",                "prefix": "rt1",           "type": "simple"},
    "BC_Z":            {"label": "bc_z",               "prefix": "bc_z",          "type": "simple"},
    "droid":           {"label": "droid_1.0.1",        "prefix": "droid",         "type": "simple"},
    "droid_RoboInter": {"label": "droid_robointer",    "prefix": "droid_robointer", "type": "robointer"},
    "egodex":          {"label": "egodex",             "prefix": "egodex",        "type": "simple"},
    "RH20T":           {"label": "rh20t",              "prefix": "rh20t",         "type": "simple"},
    "RH20T-RoboInter": {"label": "rh20t_robointer",    "prefix": "rh20t_robointer", "type": "robointer"},
    "agibotworld":     {"label": "agibotworld",        "prefix": "agibotworld",   "type": "agibotworld"},
    "RoboMindV1.0":    {"label": "robomind_v1.0",      "prefix": "robomindv1",    "type": "robomindv1"},
    "RoboMindV2.0":    {"label": "robomind_v2.0",      "prefix": "robomindv2",    "type": "simple"},
}


class SubdatasetMetaCache:
    """缓存子数据集的 info / episodes / task_map，同一子数据集只读取一次。"""

    def __init__(self):
        self._cache: dict[str, tuple] = {}

    def get(self, subdatasets_root: str):
        """返回 (info, episodes, task_map, error_reason)。"""
        if subdatasets_root in self._cache:
            return self._cache[subdatasets_root]

        meta = Path(subdatasets_root) / "meta"
        fail = lambda reason: (None, None, None, reason)

        info_path = meta / "info.json"
        if not info_path.exists():
            self._cache[subdatasets_root] = fail("missing info.json")
            return self._cache[subdatasets_root]

        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)

        eps_path = meta / "episodes.jsonl"
        if not eps_path.exists():
            self._cache[subdatasets_root] = fail("missing episodes.jsonl")
            return self._cache[subdatasets_root]

        episodes = read_jsonl(eps_path)
        if not episodes:
            self._cache[subdatasets_root] = fail("empty episodes.jsonl")
            return self._cache[subdatasets_root]

        task_map: dict[int, str] = {}
        tp = meta / "tasks.jsonl"
        if tp.exists():
            for item in read_jsonl(tp):
                task_map[item["task_index"]] = item["task"]

        self._cache[subdatasets_root] = (info, episodes, task_map, None)
        return self._cache[subdatasets_root]


def find_episode(episodes: list[dict], episode_index: int) -> dict | None:
    for ep in episodes:
        if ep["episode_index"] == episode_index:
            return ep
    return None


# ═════════════════════════════════════════════════════════════════════════════
#  Galaxea 专用：从 parquet 提取 steps + instruction
# ═════════════════════════════════════════════════════════════════════════════

def _galaxea_episode_extra(
    task_path: Path, info: dict, ep: dict, task_map: dict[int, str],
) -> dict:
    """返回 Galaxea episode 的 steps / instruction / step_source 等额外字段。"""
    ei = ep["episode_index"]
    chunks_size = info.get("chunks_size", 1000)
    data_tpl = info.get("data_path", "")
    chunk = ei // chunks_size
    pq_rel = data_tpl.format(episode_chunk=chunk, episode_index=ei)
    pq_path = task_path / pq_rel

    steps_raw, coarse_text, quality_text = None, None, None
    if pq_path.exists():
        try:
            steps_raw, coarse_text, quality_text = _gal_extract_steps(pq_path, task_map)
        except Exception:
            pass

    instruction = coarse_text
    if not instruction or instruction.strip().lower() in _GAL_QUALITY_LABELS:
        for t in ep.get("tasks", []):
            if t.strip().lower() not in _GAL_QUALITY_LABELS:
                instruction = t
                break
    if instruction and instruction.strip().lower() in _GAL_QUALITY_LABELS:
        instruction = None

    return {
        "instruction": instruction,
        "steps_raw": steps_raw,
        "has_raw_steps": 1 if steps_raw else 0,
        "step_source": "parquet_task_index" if steps_raw else "none",
    }


# ═════════════════════════════════════════════════════════════════════════════
#  AgiBotWorld 专用：从 episode 的 action_config 提取 steps
# ═════════════════════════════════════════════════════════════════════════════

def _agibotworld_episode_extra(task_path: Path, info: dict, ep: dict) -> dict:
    steps_raw = _abi_parse_steps(ep.get("action_config"))
    instruction = get_instruction(task_path / "meta" / "tasks.jsonl", ep)
    return {
        "instruction": instruction,
        "steps_raw": steps_raw,
        "has_raw_steps": 1 if steps_raw else 0,
        "step_source": "action_config" if steps_raw else "none",
    }


# ═════════════════════════════════════════════════════════════════════════════
#  RoboInter 专用：从 parquet 提取 annotation.substask 和 annotation.time_clip
# ═════════════════════════════════════════════════════════════════════════════

def _robointer_episode_extra(
    task_path: Path, info: dict, ep: dict,
) -> dict:
    """返回 RoboInter (RH20T-RoboInter/droid_RoboInter) episode 的 steps / instruction / step_source 等额外字段。"""
    import pandas as pd
    import json as json_module

    ei = ep["episode_index"]
    chunks_size = info.get("chunks_size", 1000)
    data_tpl = info.get("data_path", "")
    chunk = ei // chunks_size
    pq_rel = data_tpl.format(episode_chunk=chunk, episode_index=ei)
    pq_path = task_path / pq_rel

    instruction = None
    steps_raw = None

    if pq_path.exists():
        try:
            # 读取 parquet 文件中的 frame_index, annotation.substask 和 annotation.time_clip
            df = pd.read_parquet(pq_path, columns=["frame_index", "annotation.substask", "annotation.time_clip"])

            if not df.empty and "annotation.substask" in df.columns:
                # 解析 time_clip（存储为字符串，如 "[[0, 286]]"）
                if "annotation.time_clip" in df.columns:
                    time_clip_str = df["annotation.time_clip"].iloc[0]
                    if time_clip_str:
                        try:
                            # 解析 JSON 格式的 time_clip
                            time_clip = json_module.loads(time_clip_str)
                            if time_clip and isinstance(time_clip, list) and len(time_clip) > 0:
                                # 构造 steps_raw，每个 time_clip 段作为一个 step
                                # 从对应时间段的帧中读取 substask 描述
                                steps_raw = []
                                for i, clip in enumerate(time_clip):
                                    if isinstance(clip, list) and len(clip) == 2:
                                        start_frame, end_frame = clip[0], clip[1]
                                        # 获取该时间段内的 substask（读取该段第一帧的 substask）
                                        segment_rows = df[df["frame_index"] == start_frame]
                                        if not segment_rows.empty:
                                            substask = segment_rows.iloc[0]["annotation.substask"]
                                        else:
                                            # 如果找不到精确的frame，使用第一个可用的substask
                                            substask = df["annotation.substask"].iloc[0]

                                        steps_raw.append({
                                            "i": i,
                                            "desc": substask,
                                            "start": start_frame,
                                            "end": end_frame,
                                        })

                                # 使用第一个 substask 作为整体 instruction
                                if steps_raw:
                                    instruction = steps_raw[0]["desc"]
                        except (json_module.JSONDecodeError, ValueError):
                            pass
        except Exception:
            pass

    # 如果从 parquet 读取失败，尝试从 tasks.jsonl 获取 instruction
    if not instruction:
        instruction = get_instruction(task_path / "meta" / "tasks.jsonl", ep)

    return {
        "instruction": instruction,
        "steps_raw": steps_raw,
        "has_raw_steps": 1 if steps_raw else 0,
        "step_source": "parquet.annotation.substask" if steps_raw else "none",
    }


# ═════════════════════════════════════════════════════════════════════════════
#  RoboCOIN 专用：从 parquet 的 subtask_annotation 字段提取 steps
# ═════════════════════════════════════════════════════════════════════════════

def _robocoin_extract_subtask_steps(
    task_path: Path, info: dict, ep: dict
) -> tuple[list[dict] | None, str | None]:
    """
    从 RoboCoin 数据集的 parquet 文件和 subtask_annotations.jsonl 中提取 steps。

    返回:
        (steps_raw, instruction)
        steps_raw: 每个 subtask 的起止帧和描述
        instruction: 第一个非 null subtask 的描述作为整体 instruction
    """
    import pandas as pd

    ei = ep["episode_index"]
    chunks_size = info.get("chunks_size", 1000)
    data_tpl = info.get("data_path", "")
    chunk = ei // chunks_size
    pq_rel = data_tpl.format(episode_chunk=chunk, episode_index=ei)
    pq_path = task_path / pq_rel

    if not pq_path.exists():
        return None, None

    # 1. 读取 subtask_annotations.jsonl
    subtask_file = task_path / "annotations" / "subtask_annotations.jsonl"
    if not subtask_file.exists():
        return None, None

    subtasks = {}
    with open(subtask_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                subtasks[data['subtask_index']] = data['subtask']

    # 2. 读取 parquet 文件中的 subtask_annotation 字段
    try:
        df = pd.read_parquet(pq_path, columns=["subtask_annotation", "frame_index"])
    except Exception:
        return None, None

    if df.empty or "subtask_annotation" not in df.columns:
        return None, None

    # 3. 提取每帧的 subtask_index (使用数组的位置 0)
    subtask_indices = []
    for i in range(len(df)):
        val = df['subtask_annotation'].iloc[i]
        if hasattr(val, '__iter__') and not isinstance(val, str):
            # 数组形式，取第一个元素
            if len(val) > 0:
                subtask_indices.append(int(val[0]))
            else:
                subtask_indices.append(None)
        else:
            # 标量形式
            subtask_indices.append(int(val) if val is not None else None)

    # 4. 识别连续的 subtask 段落
    if not subtask_indices:
        return None, None

    segments = []
    current_idx = subtask_indices[0]
    start_frame = 0

    for i in range(1, len(subtask_indices)):
        if subtask_indices[i] != current_idx:
            # 结束当前段落
            segments.append({
                'subtask_index': current_idx,
                'start': start_frame,
                'end': i - 1,
            })
            current_idx = subtask_indices[i]
            start_frame = i

    # 添加最后一段
    segments.append({
        'subtask_index': current_idx,
        'start': start_frame,
        'end': len(subtask_indices) - 1,
    })

    # 5. 构建 steps_raw，过滤掉 "null" 的段落
    steps_raw = []
    instruction = None

    for seg in segments:
        idx = seg['subtask_index']
        if idx is None or idx not in subtasks:
            continue

        desc = subtasks[idx]
        # 跳过 "null" 描述的段落
        if desc is None or str(desc).lower() == "null":
            continue

        steps_raw.append({
            "i": len(steps_raw),  # 重新编号，从 0 开始
            "desc": desc,
            "start": seg['start'],
            "end": seg['end'],
        })

        # 第一个非 null 的 subtask 作为整体 instruction
        if instruction is None:
            instruction = desc

    # 如果没有有效的 steps，返回 None
    if not steps_raw:
        return None, None

    return steps_raw, instruction


def _robocoin_episode_extra(task_path: Path, info: dict, ep: dict) -> dict:
    """返回 RoboCoin episode 的 steps / instruction / step_source 等额外字段。"""
    steps_raw, _ = _robocoin_extract_subtask_steps(task_path, info, ep)

    # instruction 从 episodes.jsonl 的 tasks 字段读取
    instruction = None
    if ep.get("tasks"):
        tasks_list = ep["tasks"]
        if isinstance(tasks_list, list) and len(tasks_list) > 0:
            instruction = tasks_list[0]
        elif isinstance(tasks_list, str):
            instruction = tasks_list

    # 如果 tasks 字段为空，尝试从 tasks.jsonl 获取
    if not instruction:
        instruction = get_instruction(task_path / "meta" / "tasks.jsonl", ep)

    return {
        "instruction": instruction,
        "steps_raw": steps_raw,
        "has_raw_steps": 1 if steps_raw else 0,
        "step_source": "subtask_annotation" if steps_raw else "none",
    }


# ═════════════════════════════════════════════════════════════════════════════
#  RoboMindV1.0 专用：从 episodes.jsonl 的 action_config_json 字段提取 steps
# ═════════════════════════════════════════════════════════════════════════════

def _robomindv1_episode_extra(task_path: Path, info: dict, ep: dict) -> dict:
    """
    返回 RoboMindV1.0 episode 的 steps / instruction / step_source 等额外字段。

    steps 来自 episodes.jsonl 每行的 action_config_json 字段（JSON 字符串），
    结构为：{"task_summary": "...", "steps": [{"step_description": "...",
              "start_frame": "camera_front_NNNN.jpg",
              "end_frame": "camera_front_NNNN.jpg"}, ...]}
    帧号从文件名中提取（camera_front_NNNN.jpg → int(NNNN)）。
    过滤掉 step_description 为 "[none]" 的段落。
    """
    import re

    # instruction 从 tasks 字段读取
    instruction = None
    if ep.get("tasks"):
        t = ep["tasks"]
        instruction = t[0] if isinstance(t, list) and t else (t if isinstance(t, str) else None)
    if not instruction:
        instruction = get_instruction(task_path / "meta" / "tasks.jsonl", ep)

    # 解析 action_config_json（JSON 字符串）或 action_config（JSON 对象），兼容两种字段名
    cfg_field = "action_config_json" if ep.get("action_config_json") else "action_config"
    raw_cfg = ep.get(cfg_field)
    if not raw_cfg:
        return {"instruction": instruction, "steps_raw": None, "has_raw_steps": 0, "step_source": "none"}

    try:
        cfg = json.loads(raw_cfg) if isinstance(raw_cfg, str) else raw_cfg
    except (json.JSONDecodeError, TypeError):
        return {"instruction": instruction, "steps_raw": None, "has_raw_steps": 0, "step_source": "none"}

    raw_steps = cfg.get("steps")
    if not raw_steps:
        return {"instruction": instruction, "steps_raw": None, "has_raw_steps": 0, "step_source": "none"}

    def _frame_num(filename: str) -> int | None:
        m = re.search(r"(\d+)\.jpg$", filename or "")
        return int(m.group(1)) if m else None

    steps_raw = []
    for step in raw_steps:
        desc = (step.get("step_description") or "").strip()
        # 过滤无意义的 [none] 步骤
        if not desc or desc.lower() == "[none]":
            continue
        start = _frame_num(step.get("start_frame", ""))
        end   = _frame_num(step.get("end_frame",   ""))
        if start is None or end is None:
            continue
        steps_raw.append({"i": len(steps_raw), "desc": desc, "start": start, "end": end})

    if not steps_raw:
        return {"instruction": instruction, "steps_raw": None, "has_raw_steps": 0, "step_source": "none"}

    return {
        "instruction": instruction,
        "steps_raw": steps_raw,
        "has_raw_steps": 1,
        "step_source": cfg_field,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  通用构建
# ═════════════════════════════════════════════════════════════════════════════

def build_episode_record(
    subdatasets_root: str,
    episode_index: int,
    cluster_id: str,
    dataset_name: str,
    ds_info: dict,
    info: dict,
    episodes: list[dict],
    task_map: dict[int, str],
) -> tuple[dict | None, dict | None]:
    """为一个 cluster 代表性 episode 构建完整记录。"""
    task_path = Path(subdatasets_root)
    subdataset_name = task_path.name
    meta = task_path / "meta"

    ep = find_episode(episodes, episode_index)
    if ep is None:
        return None, dict(
            task=dataset_name, subdataset=subdataset_name,
            cluster_id=cluster_id, episode_index=episode_index,
            reason=f"episode_index {episode_index} not found in episodes.jsonl",
        )

    features = info.get("features", {})
    views = extract_views(features)
    if not views:
        return None, dict(
            task=dataset_name, subdataset=subdataset_name,
            cluster_id=cluster_id, episode_index=episode_index,
            reason="no video views in info.json",
        )

    # 对于 RoboCOIN 数据集，过滤掉 fisheye 视角
    if dataset_name in ["RoboCOIN", "RoboCOIN_add0130", "RoboCOIN_add1201"]:
        views = [v for v in views if "fisheye" not in v.lower()]
        if not views:
            return None, dict(
                task=dataset_name, subdataset=subdataset_name,
                cluster_id=cluster_id, episode_index=episode_index,
                reason="no non-fisheye views available",
            )

    videos = get_video_paths(info, views, episode_index)

    try:
        ds_dir = str(task_path.relative_to(ROOT_DIR))
    except ValueError:
        ds_dir = str(task_path)

    duration_sec = None
    if ep.get("length") and info.get("fps"):
        duration_sec = round(ep["length"] / info["fps"], 3)

    process_type = ds_info.get("type", "simple")
    prefix = ds_info["prefix"]
    label = ds_info["label"]

    if process_type == "galaxea":
        extra = _galaxea_episode_extra(task_path, info, ep, task_map)
    elif process_type == "agibotworld":
        extra = _agibotworld_episode_extra(task_path, info, ep)
    elif process_type == "robointer":
        extra = _robointer_episode_extra(task_path, info, ep)
    elif process_type == "robocoin":
        extra = _robocoin_episode_extra(task_path, info, ep)
    elif process_type == "robomindv1":
        extra = _robomindv1_episode_extra(task_path, info, ep)
    else:
        extra = {
            "instruction": get_instruction(meta / "tasks.jsonl", ep),
            "steps_raw": None,
            "has_raw_steps": 0,
            "step_source": "none",
        }

    rec = make_record(
        sample_id=f"{prefix}-{subdataset_name}-{episode_index}",
        dataset=label,
        dataset_dir=ds_dir,
        views=views,
        videos=videos,
        instruction=extra["instruction"],
        duration_sec=duration_sec,
        fps=info.get("fps"),
        resolution=get_resolution(features, views),
        robot_type=info.get("robot_type", "unknown"),
        has_raw_steps=extra["has_raw_steps"],
        step_source=extra["step_source"],
        steps_raw=extra["steps_raw"],
        cluster_id=cluster_id,
    )
    return rec, None


def _process_one_subdataset(
    json_path: str, dataset_name: str, ds_info: dict,
) -> tuple[list[dict], list[dict]]:
    """处理单个子数据集的 cluster_results.json，可在子进程中运行。"""
    sub_name = os.path.basename(os.path.dirname(json_path))
    meta_cache = SubdatasetMetaCache()
    records: list[dict] = []
    skipped: list[dict] = []

    with open(json_path) as f:
        data = json.load(f)

    cluster_repr = data.get("cluster_representation")
    if not cluster_repr:
        skipped.append(dict(task=dataset_name, subdataset=sub_name,
                            reason="no cluster_representation"))
        return records, skipped

    for cluster_id, full_path_or_list in cluster_repr.items():
        # 兼容新格式（list of paths）和旧格式（single path string）
        if isinstance(full_path_or_list, list):
            path_list = full_path_or_list
        else:
            path_list = [full_path_or_list]

        for full_path in path_list:
            parts = full_path.rsplit("/", 1)
            if len(parts) != 2:
                skipped.append(dict(task=dataset_name, subdataset=sub_name,
                                    cluster_id=cluster_id, reason=f"bad path: {full_path}"))
                continue

            subdatasets_root, ep_str = parts
            try:
                episode_index = int(ep_str)
            except ValueError:
                skipped.append(dict(task=dataset_name, subdataset=sub_name,
                                    cluster_id=cluster_id, reason=f"bad episode index: {ep_str}"))
                continue

            info, episodes, task_map, err = meta_cache.get(subdatasets_root)
            if err:
                skipped.append(dict(task=dataset_name, subdataset=sub_name,
                                    cluster_id=cluster_id, reason=err))
                continue

            rec, skip = build_episode_record(
                subdatasets_root, episode_index, cluster_id,
                dataset_name, ds_info, info, episodes, task_map,
            )
            if rec:
                records.append(rec)
            if skip:
                skipped.append(skip)

    return records, skipped


def collect_dataset(
    results_root: str, dataset_name: str, num_workers: int = 8,
) -> tuple[list[dict], list[dict]]:
    """收集一个 dataset 下所有子数据集的 cluster 代表性 episode 信息（多进程）。"""
    dataset_dir = os.path.join(results_root, dataset_name)
    if not os.path.isdir(dataset_dir):
        print(f"[WARN] {dataset_dir} not found, skipping")
        return [], []

    ds_info = DATASET_INFO.get(dataset_name)
    if ds_info is None:
        print(f"[WARN] unknown dataset '{dataset_name}', using defaults")
        ds_info = {"label": dataset_name.lower(), "prefix": dataset_name.lower(), "type": "simple"}

    # 收集所有待处理的 json 路径（递归查找，支持多级目录结构）
    json_paths = []
    dataset_path = Path(dataset_dir)

    # 使用 rglob 递归查找所有 cluster_results.json 文件
    for json_path in dataset_path.rglob("cluster_results.json"):
        if json_path.is_file():
            json_paths.append(str(json_path))

    # 排序以确保处理顺序一致
    json_paths.sort()

    if not json_paths:
        print(f"  No cluster_results.json found in {dataset_dir}")
        return [], []

    print(f"  Found {len(json_paths)} cluster_results.json files")

    records: list[dict] = []
    skipped: list[dict] = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_process_one_subdataset, jp, dataset_name, ds_info): jp
            for jp in json_paths
        }
        pbar = tqdm(as_completed(futures), total=len(json_paths),
                    desc=f"  {dataset_name}", unit="sub", ncols=100)
        for future in pbar:
            jp = futures[future]
            sub_name = os.path.basename(os.path.dirname(jp))
            pbar.set_postfix_str(sub_name[-40:], refresh=True)
            try:
                recs, skips = future.result()
                records.extend(recs)
                skipped.extend(skips)
            except Exception as e:
                print(f"\n  [ERROR] {jp}: {e}")
                skipped.append(dict(task=dataset_name,
                                    subdataset=sub_name,
                                    reason=str(e)))

    return records, skipped


def main():
    p = argparse.ArgumentParser(
        description="Collect cluster representative episodes with full metadata")
    p.add_argument("--results_root", type=str,
                   default="./results_two_stage",
                   help="聚类结果根目录 (default: ./results_two_stage)")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="要处理的数据集名称（默认处理 results_root 下所有目录）")
    p.add_argument("--output", type=str,
                   default="cluster_representation_summary.json",
                   help="输出 JSON 路径")
    p.add_argument("--workers", type=int, default=8,
                   help="并行处理子数据集的进程数 (default: 8)")
    args = p.parse_args()

    if args.datasets:
        dataset_names = args.datasets
    else:
        dataset_names = sorted(
            d for d in os.listdir(args.results_root)
            if os.path.isdir(os.path.join(args.results_root, d))
        )

    all_records: list[dict] = []
    all_skipped: list[dict] = []

    for ds_name in dataset_names:
        print(f"Collecting {ds_name} ...")
        records, skipped = collect_dataset(args.results_root, ds_name, num_workers=args.workers)
        all_records.extend(records)
        all_skipped.extend(skipped)
        print(f"  -> {len(records)} records, {len(skipped)} skipped")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(all_records)} records to {args.output}")

    if all_skipped:
        skip_path = args.output.replace(".json", "_skipped.json")
        with open(skip_path, "w", encoding="utf-8") as f:
            json.dump(all_skipped, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(all_skipped)} skipped entries to {skip_path}")


if __name__ == "__main__":
    main()
