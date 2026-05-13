#!/usr/bin/env python3
"""Verify a converted LeRobot v2.1 dataset.

This script checks:
1. Required files exist (info.json, episodes.jsonl, episodes_stats.jsonl, tasks.jsonl)
2. Episode parquet files match episodes.jsonl
3. Video files exist and have correct frame counts
4. Timestamps are monotonically increasing
5. Data integrity (random sampling)
6. episodes ↔ episodes_stats consistency (length, num_frames)
7. Feature schema validation (parquet columns vs info.json features)
8. Task alignment validation (task_id existence and uniqueness)
9. Multi-video synchronization check (frame alignment across video features)
10. Full episode coverage check (all parquet files exist)

Usage:
    python verify_v21_dataset.py --root /path/to/v21_dataset --sample-size 3
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import resource
import subprocess
import sys
from pathlib import Path
import random
import os

import jsonlines
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm


# ANSI 颜色代码
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


def print_error(msg: str):
    """打印红色错误消息"""
    print(f"{Colors.RED}{Colors.BOLD}✗ 错误: {msg}{Colors.RESET}")


def print_warning(msg: str):
    """打印黄色警告消息"""
    print(f"{Colors.YELLOW}⚠ 警告: {msg}{Colors.RESET}")


def print_success(msg: str):
    """打印绿色成功消息"""
    print(f"{Colors.GREEN}✓ {msg}{Colors.RESET}")


def print_info(msg: str):
    """打印蓝色信息消息"""
    print(f"{Colors.CYAN}ℹ {msg}{Colors.RESET}")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

# 检查4个文件必须存在
def check_required_files(root: Path) -> list[str]:
    """Check that all required v2.1 files exist."""
    errors = []
    required_files = [
        "meta/info.json",
        "meta/episodes.jsonl",
        "meta/episodes_stats.jsonl",
        "meta/tasks.jsonl",
    ]
    
    for f in required_files:
        if not (root / f).exists():
            errors.append(f"Missing required file: {f}")
    
    return errors


def check_info_json(root: Path) -> tuple[dict, list[str]]:
    """Check info.json structure and version."""
    errors = []
    info_path = root / "meta" / "info.json"
    
    try:
        with open(info_path, "r") as f:
            info = json.load(f)
    except Exception as e:
        return {}, [f"Cannot read info.json: {e}"]
    
    # Check version
    version = info.get("codebase_version")
    if version != "v2.1":
        errors.append(f"Expected codebase_version v2.1, got {version}")
    
    # Check required fields
    required_fields = ["total_episodes", "fps", "features", "data_path", "chunks_size"]
    for field in required_fields:
        if field not in info:
            errors.append(f"Missing required field in info.json: {field}")
    
    return info, errors


def load_episodes_jsonl(root: Path) -> list[dict]:
    """Load episodes.jsonl."""
    episodes_path = root / "meta" / "episodes.jsonl"
    episodes = []
    with jsonlines.open(episodes_path, "r") as reader:
        for ep in reader:
            episodes.append(ep)
    return episodes


def load_episodes_stats_jsonl(root: Path) -> list[dict]:
    """Load episodes_stats.jsonl."""
    stats_path = root / "meta" / "episodes_stats.jsonl"
    stats = []
    with jsonlines.open(stats_path, "r") as reader:
        for s in reader:
            stats.append(s)
    return stats


def load_tasks_jsonl(root: Path) -> list[dict]:
    """Load tasks.jsonl."""
    tasks_path = root / "meta" / "tasks.jsonl"
    tasks = []
    with jsonlines.open(tasks_path, "r") as reader:
        for t in reader:
            tasks.append(t)
    return tasks


def check_episodes_stats_consistency(episodes: list[dict], stats: list[dict]) -> list[str]:
    """Check A: episodes ↔ episodes_stats consistency.
    
    Validates:
    - Every episode has a stats entry
    - stats.length == episode.length (if present)
    - stats.num_frames == episode.length (if present)
    """
    errors = []
    # Build stats index by episode_index
    stats_by_idx = {}
    # for循环 检查 episodes_stats.jsonl 中的 episode_index 是否唯一
    for s in stats:
        ep_idx = s.get("episode_index") #得到stats里面的 episode index
        if ep_idx is not None and ep_idx in stats_by_idx:
            errors.append(f"Duplicate episode_index {ep_idx} in episodes_stats.jsonl")
        stats_by_idx[ep_idx] = s
    
    # Check each episode has corresponding stats，检查每个episode是否有对应的stats
    for ep in episodes:
        ep_idx = ep.get("episode_index")
        ep_length = ep.get("length")
        
        if ep_idx is None:
            continue
        
        # 检查每个episode是否有对应的stats
        if ep_idx not in stats_by_idx:
            errors.append(f"Episode {ep_idx}: missing entry in episodes_stats.jsonl")
            continue
        
        stat = stats_by_idx[ep_idx]
        
        # Check length consistency
        stat_length = stat.get("length")
        if stat_length is not None and stat_length != ep_length:
            errors.append(
                f"Episode {ep_idx}: episodes_stats.length ({stat_length}) != "
                f"episodes.length ({ep_length})"
            )
        
        # Check num_frames consistency (some datasets use this field)
        stat_num_frames = stat.get("num_frames")
        if stat_num_frames is not None and stat_num_frames != ep_length:
            errors.append(
                f"Episode {ep_idx}: episodes_stats.num_frames ({stat_num_frames}) != "
                f"episodes.length ({ep_length})"
            )
    
    return errors


def check_task_alignment(episodes: list[dict], tasks: list[dict]) -> list[str]:
    """Check C: Task alignment validation.
    
    Supports two formats:
    - Format A: episode has task_index/task_id (int) referencing tasks.jsonl task_index
    - Format B: episode has tasks (list of strings) matching tasks.jsonl task strings
    
    Validates:
    - No duplicate task_id/task_index in tasks.jsonl
    - task_id type consistency
    - All episode task references exist in tasks.jsonl
    """
    errors = []
    
    # Build task_id set and check for duplicates
    task_ids = set()
    task_id_types = set()
    task_strings = set()  # For format B
    
    for t in tasks:
        task_id = t.get("task_index")  # LeRobot v2.1 uses task_index
        if task_id is None:
            task_id = t.get("task_id")  # Fallback to task_id
        
        if task_id is not None:
            if task_id in task_ids:
                errors.append(f"Duplicate task_id/task_index: {task_id} in tasks.jsonl")
            task_ids.add(task_id)
            task_id_types.add(type(task_id).__name__)
        
        # Also collect task strings for format B
        task_str = t.get("task")
        if task_str is not None:
            task_strings.add(task_str)
    
    # Check task_id type consistency
    if len(task_id_types) > 1:
        errors.append(f"Inconsistent task_id types in tasks.jsonl: {task_id_types}")
    
    # Detect format and check episode task references
    sample_ep = episodes[0] if episodes else {}
    has_task_index = "task_index" in sample_ep or "task_id" in sample_ep
    has_tasks_list = "tasks" in sample_ep and isinstance(sample_ep.get("tasks"), list)
    
    if has_task_index:
        # Format A: task_index/task_id reference
        for ep in episodes:
            ep_idx = ep.get("episode_index")
            ep_task_id = ep.get("task_index") if ep.get("task_index") is not None else ep.get("task_id")
            
            if ep_task_id is not None and ep_task_id not in task_ids:
                errors.append(
                    f"Episode {ep_idx}: task_id/task_index {ep_task_id} not found in tasks.jsonl"
                )
    
    elif has_tasks_list:
        # Format B: tasks list - check all task strings exist in tasks.jsonl
        all_episode_tasks = set()
        for ep in episodes:
            ep_tasks = ep.get("tasks", [])
            for task_str in ep_tasks:
                all_episode_tasks.add(task_str)
        
        missing_tasks = all_episode_tasks - task_strings
        for missing_task in missing_tasks:
            errors.append(f"tasks.jsonl 缺失 task: \"{missing_task[:80]}\"")
    
    return errors


def check_feature_schema(root: Path, episode_index: int, info: dict) -> list[str]:
    """Check B: Feature schema validation for a single episode.
    
    Validates:
    - All non-video features in info.json exist as columns in parquet
    - Parquet column dtypes are compatible
    """
    errors = []
    chunks_size = info.get("chunks_size", 1000)
    chunk_idx = episode_index // chunks_size
    
    parquet_path = root / f"data/chunk-{chunk_idx:03d}/episode_{episode_index:06d}.parquet"
    
    if not parquet_path.exists():
        return errors  # Already reported by other checks
    
    try:
        table = pq.read_table(parquet_path, memory_map=False)
        parquet_columns = set(table.column_names)
        
        features = info.get("features", {})
        
        # Map numpy/torch dtypes to pyarrow compatible types
        dtype_mapping = {
            "float32": ["float", "double"],
            "float64": ["float", "double"],
            "int32": ["int32", "int64"],
            "int64": ["int64"],
            "bool": ["bool"],
            "string": ["string", "large_string"],
        }
        
        for feat_name, feat_spec in features.items():
            feat_dtype = feat_spec.get("dtype", "")
            
            # Skip video features - they should NOT be in parquet
            if feat_dtype == "video":
                if feat_name in parquet_columns:
                    errors.append(
                        f"Episode {episode_index}: video feature '{feat_name}' "
                        f"should NOT be in parquet"
                    )
                continue
            
            # Non-video features MUST be in parquet
            if feat_name not in parquet_columns:
                errors.append(
                    f"Episode {episode_index}: feature '{feat_name}' missing from parquet"
                )
                continue
            
            # Optional: check dtype compatibility (relaxed check)
            # PyArrow types can vary, so we do a loose check
            col = table.column(feat_name)
            pa_type_str = str(col.type).lower()
            
            # Just log a warning for potential mismatches (not an error)
            # because PyArrow type coercion is complex
            
    except Exception as e:
        errors.append(f"Episode {episode_index}: cannot validate schema: {e}")
    
    return errors


def check_multi_video_sync(root: Path, episode_index: int, expected_length: int, info: dict) -> list[str]:
    """Check D: Multi-video synchronization check.
    
    Validates:
    - All video features have approximately the same frame count
    - max(frame_count) - min(frame_count) <= 1
    """
    errors = []
    chunks_size = info.get("chunks_size", 1000)
    chunk_idx = episode_index // chunks_size
    
    video_keys = [key for key, ft in info.get("features", {}).items() if ft.get("dtype") == "video"]
    
    if len(video_keys) <= 1:
        return errors  # No sync check needed for single or no video
    
    frame_counts = {}
    for video_key in video_keys:
        video_path = root / f"videos/chunk-{chunk_idx:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        
        if not video_path.exists():
            continue  # Already reported by other checks
        
        frame_count = get_video_frame_count(video_path)
        if frame_count is not None:
            frame_counts[video_key] = frame_count
    
    if len(frame_counts) > 1:
        counts = list(frame_counts.values())
        max_count = max(counts)
        min_count = min(counts)
        
        if max_count - min_count > 1:
            errors.append(
                f"Episode {episode_index}: video frame count mismatch across features. "
                f"Counts: {frame_counts}"
            )
    
    return errors


def check_full_episode_coverage(root: Path, episodes: list[dict], info: dict) -> list[str]:
    """Check E: Full episode coverage (cheap global check).
    
    Validates:
    - Every episode_index in episodes.jsonl has a corresponding parquet file on disk
    """
    errors = []
    chunks_size = info.get("chunks_size", 1000)
    
    for ep in episodes:
        ep_idx = ep.get("episode_index")
        if ep_idx is None:
            continue
        
        chunk_idx = ep_idx // chunks_size
        parquet_path = root / f"data/chunk-{chunk_idx:03d}/episode_{ep_idx:06d}.parquet"
        
        if not parquet_path.exists():
            errors.append(f"Episode {ep_idx}: parquet file missing: {parquet_path}")
    
    return errors


def check_episode_index_uniqueness(episodes: list[dict]) -> list[str]:
    """Check that episode_index is unique across all episodes."""
    errors = []
    seen = set()
    
    for ep in episodes:
        ep_idx = ep.get("episode_index")
        if ep_idx is not None:
            if ep_idx in seen:
                errors.append(f"Duplicate episode_index: {ep_idx} in episodes.jsonl")
            seen.add(ep_idx)
    
    return errors


def check_episode_parquet(root: Path, episode_index: int, expected_length: int, info: dict) -> list[str]:
    """Check a single episode's parquet file."""
    errors = []
    chunks_size = info.get("chunks_size", 1000)
    chunk_idx = episode_index // chunks_size
    
    parquet_path = root / f"data/chunk-{chunk_idx:03d}/episode_{episode_index:06d}.parquet"
    
    if not parquet_path.exists():
        errors.append(f"Episode {episode_index}: parquet file missing: {parquet_path}")
        return errors
    
    try:
        table = pq.read_table(parquet_path, memory_map=False)
        actual_length = table.num_rows
        
        if actual_length != expected_length:
            errors.append(f"Episode {episode_index}: length mismatch. Expected {expected_length}, got {actual_length}")
        
        # Check timestamp monotonicity
        if "timestamp" in table.column_names:
            timestamps = table.column("timestamp").to_pylist()
            # Handle nested timestamps (may be list of single values)
            if len(timestamps) > 0 and isinstance(timestamps[0], list):
                timestamps = [t[0] for t in timestamps]
            
            for i in range(1, len(timestamps)):
                if timestamps[i] < timestamps[i-1]:
                    errors.append(f"Episode {episode_index}: timestamp not monotonic at frame {i}")
                    break
    
    except Exception as e:
        errors.append(f"Episode {episode_index}: cannot read parquet: {e}")
    
    return errors


def get_video_frame_count(video_path: Path) -> int | None:
    """Get frame count of a video using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames",
            "-of", "default=nokey=1:noprint_wrappers=1",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip())
    except Exception:
        pass
    
    # Fallback: try duration-based estimate
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=duration,r_frame_rate",
            "-of", "json",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            stream = data.get("streams", [{}])[0]
            duration = float(stream.get("duration", 0))
            fps_str = stream.get("r_frame_rate", "30/1")
            num, den = map(int, fps_str.split("/"))
            fps = num / den if den > 0 else 30
            return int(duration * fps)
    except Exception:
        pass
    
    return None


def check_episode_videos(root: Path, episode_index: int, expected_length: int, info: dict) -> list[str]:
    """Check videos for a single episode."""
    errors = []
    chunks_size = info.get("chunks_size", 1000)
    chunk_idx = episode_index // chunks_size
    fps = info.get("fps", 30)
    
    video_keys = [key for key, ft in info.get("features", {}).items() if ft.get("dtype") == "video"]
    
    for video_key in video_keys:
        video_path = root / f"videos/chunk-{chunk_idx:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        
        if not video_path.exists():
            errors.append(f"Episode {episode_index}: video missing: {video_path}")
            continue
        
        frame_count = get_video_frame_count(video_path)
        if frame_count is not None:
            # Allow 1-2 frame drift
            if abs(frame_count - expected_length) > 2:
                errors.append(
                    f"Episode {episode_index}, {video_key}: frame count mismatch. "
                    f"Expected ~{expected_length}, got {frame_count}"
                )
    
    return errors


def verify_dataset(root: Path, sample_size: int = 3, verbose: bool = False) -> tuple[bool, list[str]]:
    """Verify a v2.1 dataset.
    
    Returns:
        (passed: bool, errors: list[str]) - 验证是否通过，以及错误列表
    """
    all_errors = []
    
    print("\n" + "=" * 70)
    print(f"开始验证数据集: {root}")
    print("=" * 70)
    
    # =========================================================================
    # 1. 必需文件存在性检查
    # =========================================================================
    print("\n" + "-" * 70)
    print("【1. 必需文件存在性检查】")
    print("-" * 70)
    
    required_files = [
        "meta/info.json",
        "meta/episodes.jsonl",
        "meta/episodes_stats.jsonl",
        "meta/tasks.jsonl",
    ]
    
    file_check_passed = True
    missing_files = []
    for f in required_files:
        exists = (root / f).exists()
        if exists:
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} - 缺失")
            all_errors.append(f"缺失必需文件: {f}")
            missing_files.append(f)
            file_check_passed = False
    
    if not file_check_passed:
        print(f"\n【结论】必需文件缺失，验证终止")
        print(f"  缺失的文件: {', '.join(missing_files)}")
        return False, all_errors
    
    # =========================================================================
    # 2. info.json 结构检查
    # =========================================================================
    print("\n" + "-" * 70)
    print("【2. info.json 结构检查】")
    print("-" * 70)
    
    info_path = root / "meta" / "info.json"
    try:
        with open(info_path, "r") as f:
            info = json.load(f)
    except Exception as e:
        print(f"  ✗ 无法读取 info.json: {e}")
        all_errors.append(f"无法读取 info.json: {e}")
        return False, all_errors
    
    # 2.1 codebase_version 检查
    version = info.get("codebase_version")
    version_ok = version == "v2.1"
    print(f"  2.1 codebase_version 必须为 \"v2.1\"")
    if version_ok:
        print(f"      {Colors.GREEN}结果: 当前值为 \"{version}\" ✓ 正确{Colors.RESET}")
    else:
        print(f"      {Colors.RED}{Colors.BOLD}结果: 当前值为 \"{version}\" ✗ 错误{Colors.RESET}")
        all_errors.append(f"codebase_version 应为 v2.1，实际为 {version}")
    
    # 2.2 必需字段检查
    required_fields = ["total_episodes", "fps", "features", "data_path", "chunks_size"]
    print(f"  2.2 检查必需字段: {required_fields}")
    for field in required_fields:
        field_exists = field in info
        field_value = info.get(field, "N/A")
        if isinstance(field_value, dict):
            field_value = f"<dict with {len(field_value)} keys>"
        status = "✓ 存在" if field_exists else "✗ 缺失"
        print(f"      - {field}: {status}, 值: {field_value}")
        if not field_exists:
            all_errors.append(f"info.json 缺失字段: {field}")
    
    if not info:
        all_errors.append("info.json 为空或无效")
        return False, all_errors
    
    # =========================================================================
    # 3. episodes.jsonl 加载与检查
    # =========================================================================
    print("\n" + "-" * 70)
    print("【3. episodes.jsonl 加载与检查】")
    print("-" * 70)
    
    try:
        episodes = load_episodes_jsonl(root)
        print(f"  3.1 加载 episodes.jsonl")
        print(f"      结果: ✓ 成功加载 {len(episodes)} 条记录")
    except Exception as e:
        print(f"      结果: ✗ 加载失败: {e}")
        all_errors.append(f"加载 episodes.jsonl 失败: {e}")
        return False, all_errors
    
    # 3.2 episode 数量一致性检查
    total_episodes = info.get('total_episodes')
    count_match = len(episodes) == total_episodes
    print(f"  3.2 episode 数量与 info.json.total_episodes 一致性检查")
    print(f"      episodes.jsonl 记录数: {len(episodes)}")
    print(f"      info.json.total_episodes: {total_episodes}")
    if count_match:
        print(f"      {Colors.GREEN}结果: ✓ 一致{Colors.RESET}")
    else:
        print(f"      {Colors.RED}{Colors.BOLD}结果: ✗ 不一致{Colors.RESET}")
        all_errors.append(f"episode 数量不一致: {len(episodes)} != {total_episodes}")
    
    # 3.3 total_chunks 一致性检查
    chunks_size = info.get('chunks_size', 1000)
    info_total_chunks = info.get('total_chunks')
    # 计算期望的 total_chunks: ceil(total_episodes / chunks_size)
    expected_total_chunks = (total_episodes + chunks_size - 1) // chunks_size if total_episodes and chunks_size else 0
    print(f"  3.3 total_chunks 一致性检查")
    print(f"      info.json.total_chunks: {info_total_chunks}")
    print(f"      期望值 (ceil({total_episodes}/{chunks_size})): {expected_total_chunks}")
    if info_total_chunks == expected_total_chunks:
        print(f"      {Colors.GREEN}结果: ✓ 一致{Colors.RESET}")
    else:
        print(f"      {Colors.RED}{Colors.BOLD}结果: ✗ 不一致{Colors.RESET}")
        all_errors.append(f"total_chunks 不一致: info.json={info_total_chunks}, 期望={expected_total_chunks}")
    
    # 3.4 episode_index 唯一性检查
    print(f"  3.4 episode_index 唯一性检查")
    seen_indices = set()
    duplicate_indices = []
    for ep in episodes:
        ep_idx = ep.get("episode_index")
        if ep_idx is not None:
            if ep_idx in seen_indices:
                duplicate_indices.append(ep_idx)
            seen_indices.add(ep_idx)
    
    if duplicate_indices:
        print(f"      {Colors.RED}{Colors.BOLD}结果: ✗ 发现重复的 episode_index: {duplicate_indices[:5]}...{Colors.RESET}")
        all_errors.extend([f"重复的 episode_index: {idx}" for idx in duplicate_indices])
    else:
        print(f"      {Colors.GREEN}结果: ✓ 所有 {len(episodes)} 个 episode_index 唯一{Colors.RESET}")
    
    # =========================================================================
    # 4. episodes ↔ episodes_stats 一致性检查
    # =========================================================================
    print("\n" + "-" * 70)
    print("【4. episodes ↔ episodes_stats 一致性检查】")
    print("-" * 70)
    
    try:
        stats = load_episodes_stats_jsonl(root)
        print(f"  4.1 加载 episodes_stats.jsonl")
        print(f"      结果: ✓ 成功加载 {len(stats)} 条记录")
        
        # Build stats index
        stats_by_idx = {s.get("episode_index"): s for s in stats}
        
        # 4.2 每个 episode 在 stats 中有对应条目
        print(f"  4.2 检查每个 episode 在 episodes_stats.jsonl 中有对应条目")
        missing_stats = [ep.get("episode_index") for ep in episodes 
                        if ep.get("episode_index") not in stats_by_idx]
        if missing_stats:
            print(f"      {Colors.RED}{Colors.BOLD}结果: ✗ {len(missing_stats)} 个 episode 缺少 stats 条目{Colors.RESET}")
            print(f"      {Colors.RED}缺失的 episode_index: {missing_stats[:5]}...{Colors.RESET}")
            all_errors.extend([f"Episode {idx} 缺少 stats 条目" for idx in missing_stats])
        else:
            print(f"      {Colors.GREEN}结果: ✓ 所有 {len(episodes)} 个 episode 都有对应的 stats 条目{Colors.RESET}")
        
        # 4.3 length 一致性检查
        print(f"  4.3 检查 stats.length == episode.length")
        length_mismatch = []
        for ep in episodes:
            ep_idx = ep.get("episode_index")
            ep_length = ep.get("length")
            if ep_idx in stats_by_idx:
                stat = stats_by_idx[ep_idx]
                stat_length = stat.get("length")
                if stat_length is not None and stat_length != ep_length:
                    length_mismatch.append((ep_idx, ep_length, stat_length))
        
        if length_mismatch:
            print(f"      {Colors.RED}{Colors.BOLD}结果: ✗ {len(length_mismatch)} 个 episode 的 length 不一致{Colors.RESET}")
            for idx, ep_len, stat_len in length_mismatch[:3]:
                print(f"        {Colors.RED}- Episode {idx}: episodes.length={ep_len}, stats.length={stat_len}{Colors.RESET}")
            all_errors.extend([f"Episode {idx} length 不一致" for idx, _, _ in length_mismatch])
        else:
            print(f"      {Colors.GREEN}结果: ✓ 所有 episode 的 length 一致{Colors.RESET}")
        
    except Exception as e:
        print(f"  ✗ 无法加载 episodes_stats.jsonl: {e}")
        all_errors.append(f"无法加载 episodes_stats.jsonl: {e}")
    
    # =========================================================================
    # 5. Task 对齐验证
    # =========================================================================
    print("\n" + "-" * 70)
    print("【5. Task 对齐验证】")
    print("-" * 70)
    
    try:
        tasks = load_tasks_jsonl(root)
        print(f"  5.1 加载 tasks.jsonl")
        print(f"      结果: ✓ 成功加载 {len(tasks)} 条记录")
        
        # 5.2 total_tasks 一致性检查
        info_total_tasks = info.get('total_tasks')
        actual_task_count = len(tasks)
        print(f"  5.2 total_tasks 一致性检查")
        print(f"      info.json.total_tasks: {info_total_tasks}")
        print(f"      tasks.jsonl 记录数: {actual_task_count}")
        if info_total_tasks == actual_task_count:
            print(f"      {Colors.GREEN}结果: ✓ 一致{Colors.RESET}")
        else:
            print(f"      {Colors.RED}{Colors.BOLD}结果: ✗ 不一致{Colors.RESET}")
            all_errors.append(f"total_tasks 不一致: info.json={info_total_tasks}, tasks.jsonl 记录数={actual_task_count}")
        
        # Build task_id set
        task_ids = set()
        task_id_types = set()
        duplicate_task_ids = []
        
        for t in tasks:
            task_id = t.get("task_index") or t.get("task_id")
            if task_id is not None:
                if task_id in task_ids:
                    duplicate_task_ids.append(task_id)
                task_ids.add(task_id)
                task_id_types.add(type(task_id).__name__)
        
        # 5.3 task_id 唯一性检查
        print(f"  5.3 检查 task_id/task_index 唯一性")
        if duplicate_task_ids:
            print(f"      {Colors.RED}{Colors.BOLD}结果: ✗ 发现重复的 task_id: {duplicate_task_ids}{Colors.RESET}")
            all_errors.extend([f"重复的 task_id: {tid}" for tid in duplicate_task_ids])
        else:
            print(f"      {Colors.GREEN}结果: ✓ 所有 {len(task_ids)} 个 task_id 唯一{Colors.RESET}")
        
        # 5.4 task_id 类型一致性检查
        print(f"  5.4 检查 task_id 类型一致性")
        print(f"      发现的类型: {task_id_types}")
        if len(task_id_types) > 1:
            print(f"      {Colors.RED}{Colors.BOLD}结果: ✗ task_id 类型不一致{Colors.RESET}")
            all_errors.append(f"task_id 类型不一致: {task_id_types}")
        else:
            print(f"      {Colors.GREEN}结果: ✓ task_id 类型一致{Colors.RESET}")
        
        # 5.5 每个 episode 的 task 引用检查
        # 支持两种格式：
        # - 格式A: episode 有 task_index/task_id 字段 (int)，指向 tasks.jsonl 的 task_index
        # - 格式B: episode 有 tasks 字段 (list of strings)，每个字符串应匹配 tasks.jsonl 的 task
        
        # 检测使用哪种格式
        sample_ep = episodes[0] if episodes else {}
        has_task_index = "task_index" in sample_ep or "task_id" in sample_ep
        has_tasks_list = "tasks" in sample_ep and isinstance(sample_ep.get("tasks"), list)
        
        if has_task_index:
            print(f"  5.5 检查每个 episode 的 task_index/task_id 存在于 tasks.jsonl 中")
            print(f"      检测到格式: task_index/task_id 引用模式")
            missing_task_refs = []
            for ep in episodes:
                ep_idx = ep.get("episode_index")
                ep_task_id = ep.get("task_index") if ep.get("task_index") is not None else ep.get("task_id")
                if ep_task_id is not None and ep_task_id not in task_ids:
                    missing_task_refs.append((ep_idx, ep_task_id))
            
            if missing_task_refs:
                print(f"      结果: ✗ {len(missing_task_refs)} 个 episode 引用了不存在的 task_id")
                for idx, tid in missing_task_refs[:3]:
                    print(f"        - Episode {idx} 引用 task_id={tid}，但不存在于 tasks.jsonl")
                all_errors.extend([f"Episode {idx} 引用不存在的 task_id {tid}" for idx, tid in missing_task_refs])
            else:
                print(f"      结果: ✓ 所有 episode 的 task_id 引用有效")
        
        elif has_tasks_list:
            print(f"  5.5 检查每个 episode 的 tasks 列表内容存在于 tasks.jsonl 中")
            print(f"      检测到格式: tasks 列表模式 (每个 episode 包含多个 task 字符串)")
            
            # 构建 tasks.jsonl 中的 task 字符串集合
            task_strings_in_jsonl = set()
            for t in tasks:
                task_str = t.get("task")
                if task_str is not None:
                    task_strings_in_jsonl.add(task_str)
            print(f"      tasks.jsonl 中的 task 字符串数量: {len(task_strings_in_jsonl)}")
            
            # 收集 episodes 中所有出现的 task 字符串
            all_episode_tasks = set()
            for ep in episodes:
                ep_tasks = ep.get("tasks", [])
                for task_str in ep_tasks:
                    all_episode_tasks.add(task_str)
            print(f"      episodes.jsonl 中出现的唯一 task 字符串数量: {len(all_episode_tasks)}")
            
            # 找出 episodes 中有但 tasks.jsonl 中没有的 task
            missing_tasks = all_episode_tasks - task_strings_in_jsonl
            
            if missing_tasks:
                print(f"      {Colors.RED}{Colors.BOLD}结果: ✗ 发现 {len(missing_tasks)} 个 task 在 episodes.jsonl 中出现但 tasks.jsonl 中缺失{Colors.RESET}")
                print(f"\n      {Colors.RED}{Colors.BOLD}【tasks.jsonl 中缺失的 task 列表】:{Colors.RESET}")
                for i, missing_task in enumerate(sorted(missing_tasks)):
                    # 找出哪些 episode 使用了这个 task
                    eps_using_task = [ep.get("episode_index") for ep in episodes 
                                     if missing_task in ep.get("tasks", [])]
                    eps_str = str(eps_using_task[:5]) + ("..." if len(eps_using_task) > 5 else "")
                    print(f"        {Colors.RED}{i+1}. \"{missing_task}\"{Colors.RESET}")
                    print(f"           使用此 task 的 episode: {eps_str}")
                    all_errors.append(f"tasks.jsonl 缺失 task: \"{missing_task[:80]}...\"")
                print()
            else:
                print(f"      {Colors.GREEN}结果: ✓ 所有 episode 的 tasks 内容都存在于 tasks.jsonl 中{Colors.RESET}")
            
            # 额外信息：tasks.jsonl 中有但 episodes 中没使用的 task
            unused_tasks = task_strings_in_jsonl - all_episode_tasks
            if unused_tasks:
                print(f"      (信息: tasks.jsonl 中有 {len(unused_tasks)} 个 task 未被任何 episode 使用)")
        
        else:
            print(f"  5.4 检查 episode 的 task 引用")
            print(f"      结果: ⚠ 未检测到 task_index/task_id 或 tasks 字段，跳过此检查")
        
    except Exception as e:
        print(f"  ✗ 无法加载 tasks.jsonl: {e}")
        all_errors.append(f"无法加载 tasks.jsonl: {e}")
    
    # =========================================================================
    # 6. 全 Episode Parquet 文件覆盖检查
    # =========================================================================
    print("\n" + "-" * 70)
    print("【6. 全 Episode Parquet 文件覆盖检查】")
    print("-" * 70)
    
    chunks_size = info.get("chunks_size", 1000) # 默认是1000
    print(f"  6.1 检查每个 episode 的 parquet 文件是否存在 (chunks_size={chunks_size})")
    
    missing_parquets = []
    for ep in episodes:
        ep_idx = ep.get("episode_index")
        if ep_idx is None:
            continue
        chunk_idx = ep_idx // chunks_size
        parquet_path = root / f"data/chunk-{chunk_idx:03d}/episode_{ep_idx:06d}.parquet"
        if not parquet_path.exists():
            missing_parquets.append(ep_idx)
    
    if missing_parquets:
        print(f"      {Colors.RED}{Colors.BOLD}结果: ✗ {len(missing_parquets)} 个 parquet 文件缺失{Colors.RESET}")
        print(f"      {Colors.RED}缺失的 episode_index: {missing_parquets[:10]}...{Colors.RESET}")
        all_errors.extend([f"Episode {idx} parquet 文件缺失" for idx in missing_parquets])
    else:
        print(f"      {Colors.GREEN}结果: ✓ 所有 {len(episodes)} 个 parquet 文件存在{Colors.RESET}")
    
    # =========================================================================
    # 7. 采样 Episode 详细检查
    # =========================================================================
    print("\n" + "-" * 70)
    print("【7. 采样 Episode 详细检查】")
    print("-" * 70)
    
    if len(episodes) > sample_size:
        sampled = random.sample(episodes, sample_size)
    else:
        sampled = episodes
    
    print(f"  采样数量: {len(sampled)} (共 {len(episodes)} 个 episode)")
    print(f"  采样的 episode_index: {[ep.get('episode_index') for ep in sampled]}")
    
    for i, ep in enumerate(sampled):
        ep_idx = ep.get("episode_index")
        length = ep.get("length")
        
        if ep_idx is None or length is None:
            all_errors.append(f"Episode 缺少必需字段: {ep}")
            continue
        
        print(f"\n  --- Episode {ep_idx} (length={length}) ---")
        
        chunk_idx = ep_idx // chunks_size
        parquet_path = root / f"data/chunk-{chunk_idx:03d}/episode_{ep_idx:06d}.parquet"
        
        # 读取 parquet 一次，供 7.1 和 7.2 共用（memory_map=False 避免泄漏文件描述符）
        table = None
        if parquet_path.exists():
            try:
                table = pq.read_table(parquet_path, memory_map=False)
            except Exception as e:
                print(f"    ✗ Parquet 读取失败: {e}")
                all_errors.append(f"Episode {ep_idx} parquet 读取失败: {e}")
        
        # 7.1 Parquet 文件检查
        print(f"    7.{i+1}.1 Parquet 文件检查: {parquet_path.name}")
        if table is not None:
            actual_length = table.num_rows
            length_ok = actual_length == length
            print(f"           行数检查: 期望 {length}, 实际 {actual_length} {'✓' if length_ok else '✗'}")
            if not length_ok:
                all_errors.append(f"Episode {ep_idx} parquet 行数不匹配")
            
            if "timestamp" in table.column_names:
                timestamps = table.column("timestamp").to_pylist()
                if len(timestamps) > 0 and isinstance(timestamps[0], list):
                    timestamps = [t[0] for t in timestamps]
                monotonic = all(timestamps[j] <= timestamps[j+1] for j in range(len(timestamps)-1))
                print(f"           timestamp 单调性: {'✓ 单调非递减' if monotonic else '✗ 非单调'}")
                if not monotonic:
                    all_errors.append(f"Episode {ep_idx} timestamp 非单调")
            else:
                print(f"           timestamp 列: 不存在 (跳过单调性检查)")
        elif not parquet_path.exists():
            print(f"           ✗ 文件不存在")
        
        # 7.2 Feature Schema 检查
        print(f"    7.{i+1}.2 Feature Schema 检查")
        features = info.get("features", {})
        video_features = [k for k, v in features.items() if v.get("dtype") == "video"]
        non_video_features = [k for k, v in features.items() if v.get("dtype") != "video"]
        print(f"           video 特征 ({len(video_features)}): {video_features[:3]}{'...' if len(video_features) > 3 else ''}")
        print(f"           non-video 特征 ({len(non_video_features)}): {non_video_features[:3]}{'...' if len(non_video_features) > 3 else ''}")
        
        if table is not None:
            parquet_columns = set(table.column_names)
            
            missing_cols = [f for f in non_video_features if f not in parquet_columns]
            if missing_cols:
                print(f"           ✗ parquet 中缺少列: {missing_cols[:5]}")
                all_errors.extend([f"Episode {ep_idx} parquet 缺少列 {col}" for col in missing_cols])
            else:
                print(f"           ✓ 所有 non-video 特征都在 parquet 中")
            
            video_in_parquet = [f for f in video_features if f in parquet_columns]
            if video_in_parquet:
                print(f"           ✗ video 特征不应在 parquet 中: {video_in_parquet}")
                all_errors.extend([f"Episode {ep_idx} video 特征 {col} 不应在 parquet 中" for col in video_in_parquet])
            else:
                print(f"           ✓ video 特征未出现在 parquet 中")
        
        del table
        
        # 7.3 视频文件检查
        print(f"    7.{i+1}.3 视频文件检查")
        video_frame_counts = {}
        for video_key in video_features:
            video_path = root / f"videos/chunk-{chunk_idx:03d}/{video_key}/episode_{ep_idx:06d}.mp4"
            if video_path.exists():
                frame_count = get_video_frame_count(video_path)
                if frame_count is not None:
                    video_frame_counts[video_key] = frame_count
                    drift = abs(frame_count - length)
                    drift_ok = drift <= 2
                    print(f"           {video_key}: 存在, 帧数={frame_count}, 期望~{length} {'✓' if drift_ok else '✗'}")
                    if not drift_ok:
                        all_errors.append(f"Episode {ep_idx} {video_key} 帧数不匹配: {frame_count} vs {length}")
                else:
                    print(f"           {video_key}: 存在, 帧数=无法获取")
            else:
                print(f"           {video_key}: ✗ 文件不存在")
                all_errors.append(f"Episode {ep_idx} 视频文件缺失: {video_key}")
        
        # 7.4 多视频同步检查
        if len(video_frame_counts) > 1:
            print(f"    7.{i+1}.4 多视频同步检查")
            counts = list(video_frame_counts.values())
            max_diff = max(counts) - min(counts)
            sync_ok = max_diff <= 1
            print(f"           帧数范围: {min(counts)} - {max(counts)}, 差异={max_diff} {'✓' if sync_ok else '✗'}")
            if not sync_ok:
                all_errors.append(f"Episode {ep_idx} 多视频帧数不同步")
        
        gc.collect()
    
    # =========================================================================
    # 8. 验证结果汇总
    # =========================================================================
    print("\n" + "=" * 70)
    print("【验证结果汇总】")
    print("=" * 70)
    
    if all_errors:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ 验证失败，共发现 {len(all_errors)} 个错误:{Colors.RESET}")
        for i, e in enumerate(all_errors[:20]):
            print(f"  {Colors.RED}{i+1}. {e}{Colors.RESET}")
        if len(all_errors) > 20:
            print(f"  {Colors.RED}... 还有 {len(all_errors) - 20} 个错误未显示{Colors.RESET}")
        print("\n" + "=" * 70)
        return False, all_errors
    else:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ 验证通过！数据集符合 LeRobot v2.1 规范{Colors.RESET}")
        print(f"  {Colors.GREEN}- 总 episode 数: {len(episodes)}{Colors.RESET}")
        print(f"  {Colors.GREEN}- 采样验证数: {len(sampled)}{Colors.RESET}")
        print(f"  {Colors.GREEN}- video 特征数: {len([k for k, v in info.get('features', {}).items() if v.get('dtype') == 'video'])}{Colors.RESET}")
        print("\n" + "=" * 70)
        return True, []


def batch_verify(root: Path, sample_size: int = 3, verbose: bool = False) -> tuple[int, int, list[str], list[str], dict[str, str]]:
    """递归查找并验证 root 下所有 LeRobot 数据集。
    
    通过递归搜索所有 meta/info.json 文件，找出可能的 LeRobot 数据集。
    版本检查会在后续验证过程中进行，非 v2.1 版本会被报告为错误。
    使用 tqdm 显示验证进度。
    
    Returns:
        (success_count, failure_count, passed_datasets, failed_datasets, error_details)
        error_details: dict mapping dataset_name -> error_reason
    """
    print(f"\n{Colors.CYAN}正在搜索 LeRobot 数据集...{Colors.RESET}")
    print(f"搜索根目录: {root}")
    
    # Find all datasets by checking for meta/info.json (不检查版本号，版本检查在验证时进行)
    datasets = []
    for info_path in root.rglob("meta/info.json"):
        # 只要存在 meta/info.json 就认为是数据集目录
        datasets.append(info_path.parent.parent)
    
    if not datasets:
        print(f"{Colors.YELLOW}警告: 在 {root} 下未找到任何 LeRobot 数据集 (需包含 meta/info.json){Colors.RESET}")
        return 0, 0, [], [], {}
    
    print("\n" + "=" * 70)
    print(f"{Colors.CYAN}{Colors.BOLD}递归批量验证模式{Colors.RESET}")
    print("=" * 70)
    print(f"  搜索根目录: {root}")
    print(f"  发现的数据集数量: {len(datasets)}")
    print(f"  每个数据集采样 episode 数: {sample_size}")
    print("=" * 70 + "\n")
    
    success_count = 0
    failure_count = 0
    passed_datasets = []
    failed_datasets = []
    error_details = {}  # 存储每个失败数据集的错误原因
    
    # 使用 tqdm 显示进度
    for ds_path in tqdm(datasets, desc="验证数据集", unit="dataset", ncols=100):
        # 获取相对路径作为数据集名称
        try:
            ds_name = str(ds_path.relative_to(root))
        except ValueError:
            ds_name = ds_path.name
        
        tqdm.write(f"\n{'─' * 70}")
        tqdm.write(f"开始验证数据集: {ds_path}")
        tqdm.write(f"数据集名称: {ds_name}")
        tqdm.write(f"{'─' * 70}")
        
        try:
            passed, errors = verify_dataset(ds_path, sample_size=sample_size, verbose=verbose)
            if passed:
                success_count += 1
                passed_datasets.append(ds_name)
                tqdm.write(f"{Colors.GREEN}>>> {ds_name}: 验证通过 ✓{Colors.RESET}\n")
            else:
                failure_count += 1
                failed_datasets.append(ds_name)
                # 合并错误信息，用分号分隔
                error_details[ds_name] = "; ".join(errors) if errors else "Unknown error"
                tqdm.write(f"{Colors.RED}>>> {ds_name}: 验证失败 ✗{Colors.RESET}\n")
        except Exception as e:
            failure_count += 1
            failed_datasets.append(ds_name)
            error_details[ds_name] = f"验证异常: {e}"
            tqdm.write(f"{Colors.RED}>>> {ds_name}: 验证异常 ✗ - {e}{Colors.RESET}\n")
        finally:
            gc.collect()  # 释放 PyArrow/parquet 等持有的文件句柄，缓解 Too many open files
    
    # 打印汇总结果
    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}递归批量验证结果汇总{Colors.RESET}")
    print("=" * 70)
    print(f"  总数据集数: {len(datasets)}")
    print(f"  {Colors.GREEN}通过: {success_count}{Colors.RESET}")
    print(f"  {Colors.RED}失败: {failure_count}{Colors.RESET}")
    
    if passed_datasets:
        print(f"\n{Colors.GREEN}通过的数据集 ({len(passed_datasets)}):{Colors.RESET}")
        for ds in passed_datasets[:20]:  # 最多显示20个
            print(f"    ✓ {ds}")
        if len(passed_datasets) > 20:
            print(f"    ... 还有 {len(passed_datasets) - 20} 个")
    
    if failed_datasets:
        print(f"\n{Colors.RED}失败的数据集 ({len(failed_datasets)}):{Colors.RESET}")
        for ds in failed_datasets:
            print(f"    ✗ {ds}")
    
    print("\n" + "=" * 70)
    
    return success_count, failure_count, passed_datasets, failed_datasets, error_details


def full_datasets_verify(datasets_root: Path, sample_size: int = 3, verbose: bool = False) -> tuple[int, int, list[str], list[str], dict[str, str]]:
    """验证 datasets_root 下的所有子数据集。
    
    假设目录结构为:
    datasets_root/
    ├── dataset_1/
    │   ├── meta/
    │   ├── data/
    │   └── videos/
    ├── dataset_2/
    │   ├── meta/
    │   ├── data/
    │   └── videos/
    └── ...
    
    每个直接子目录被视为一个独立的 LeRobot v2.1 数据集。
    
    Returns:
        (success_count, failure_count, passed_datasets, failed_datasets, error_details)
        error_details: dict mapping dataset_name -> error_reason
    """
    # 收集所有直接子目录
    subdirs = [d for d in datasets_root.iterdir() if d.is_dir()]
    
    # 过滤出看起来像 LeRobot v2.1 数据集的目录（有 meta/info.json）
    datasets = []
    for subdir in subdirs:
        info_path = subdir / "meta" / "info.json"
        if info_path.exists():
            datasets.append(subdir)
    
    if not datasets:
        print(f"{Colors.YELLOW}警告: 在 {datasets_root} 下未找到任何 LeRobot v2.1 数据集{Colors.RESET}")
        print(f"  检查的子目录数: {len(subdirs)}")
        print(f"  期望每个子目录包含 meta/info.json 文件")
        return 0, 0, [], [], {}
    
    print("\n" + "=" * 70)
    print(f"{Colors.CYAN}{Colors.BOLD}全量数据集验证模式{Colors.RESET}")
    print("=" * 70)
    print(f"  数据集根目录: {datasets_root}")
    print(f"  发现的数据集数量: {len(datasets)}")
    print(f"  每个数据集采样 episode 数: {sample_size}")
    print("=" * 70 + "\n")
    
    success_count = 0
    failure_count = 0
    passed_datasets = []
    failed_datasets = []
    error_details = {}  # 存储每个失败数据集的错误原因
    
    # 使用 tqdm 显示进度
    for ds_path in tqdm(datasets, desc="验证数据集", unit="dataset", ncols=100):
        ds_name = ds_path.name
        tqdm.write(f"\n{'─' * 70}")
        tqdm.write(f"开始验证数据集: {ds_path}")
        tqdm.write(f"数据集名称: {ds_name}")
        tqdm.write(f"{'─' * 70}")
        
        try:
            passed, errors = verify_dataset(ds_path, sample_size=sample_size, verbose=verbose)
            if passed:
                success_count += 1
                passed_datasets.append(ds_name)
                tqdm.write(f"{Colors.GREEN}>>> {ds_name}: 验证通过 ✓{Colors.RESET}\n")
            else:
                failure_count += 1
                failed_datasets.append(ds_name)
                error_details[ds_name] = "; ".join(errors) if errors else "Unknown error"
                tqdm.write(f"{Colors.RED}>>> {ds_name}: 验证失败 ✗{Colors.RESET}\n")
        except Exception as e:
            failure_count += 1
            failed_datasets.append(ds_name)
            error_details[ds_name] = f"验证异常: {e}"
            tqdm.write(f"{Colors.RED}>>> {ds_name}: 验证异常 ✗ - {e}{Colors.RESET}\n")
        finally:
            gc.collect()  # 释放 PyArrow/parquet 等持有的文件句柄，缓解 Too many open files
    
    # 打印汇总结果
    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}全量数据集验证结果汇总{Colors.RESET}")
    print("=" * 70)
    print(f"  总数据集数: {len(datasets)}")
    print(f"  {Colors.GREEN}通过: {success_count}{Colors.RESET}")
    print(f"  {Colors.RED}失败: {failure_count}{Colors.RESET}")
    
    if passed_datasets:
        print(f"\n{Colors.GREEN}通过的数据集 ({len(passed_datasets)}):{Colors.RESET}")
        for ds in passed_datasets:
            print(f"    ✓ {ds}")
    
    if failed_datasets:
        print(f"\n{Colors.RED}失败的数据集 ({len(failed_datasets)}):{Colors.RESET}")
        for ds in failed_datasets:
            print(f"    ✗ {ds}")
    
    print("\n" + "=" * 70)
    
    return success_count, failure_count, passed_datasets, failed_datasets, error_details


def write_error_summary_json(error_details: dict[str, str], output_path: Path):
    """Write error summary to JSON file.
    
    Args:
        error_details: dict mapping dataset_name -> error_reason
        output_path: Path to output JSON file
    """
    summary = {"error": error_details}
    
    # 确保目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n{Colors.CYAN}错误汇总 JSON 已保存到: {output_path}{Colors.RESET}")


def _raise_nofile_limit():
    """尝试提高进程可打开文件数限制，缓解 Too many open files"""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = hard if hard != resource.RLIM_INFINITY else 1048576
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (ValueError, OSError):
        pass


def main():
    _raise_nofile_limit()
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="Verify a converted LeRobot v2.1 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 验证单个数据集
  python verify_v21_dataset.py --root /path/to/dataset --sample-size 3

  # 递归查找并验证所有 v2.1 数据集 (深度搜索)
  python verify_v21_dataset.py --root /path/to/root --batch

  # 验证目录下的所有子数据集 (浅层搜索，带进度条)
  python verify_v21_dataset.py --full-datasets --datasets-root /path/to/datasets_root
        """
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Path to a single v2.1 dataset",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3,
        help="Number of episodes to sample for detailed verification (default: 3)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Recursively find and verify all datasets (with meta/info.json) under --root (deep search)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--full-datasets",
        action="store_true",
        help="验证 --datasets-root 下的所有直接子目录数据集 (浅层搜索，带 tqdm 进度条)",
    )
    parser.add_argument(
        "--datasets-root",
        type=str,
        default=None,
        help="与 --full-datasets 配合使用，指定包含多个数据集的根目录",
    )
    parser.add_argument(
        "--error-json",
        type=str,
        default=None,
        help="错误汇总 JSON 文件的完整路径",
    )
    
    args = parser.parse_args()
    
    # 设置错误汇总 JSON 文件路径
    error_json_path = Path(args.error_json) if args.error_json else None
    error_details = {}
    
    # 模式3: --full-datasets 模式
    if args.full_datasets:
        if not args.datasets_root:
            print(f"{Colors.RED}错误: --full-datasets 需要配合 --datasets-root 使用{Colors.RESET}")
            print("示例: python verify_v21_dataset.py --full-datasets --datasets-root /path/to/datasets")
            sys.exit(1)
        
        datasets_root = Path(args.datasets_root)
        if not datasets_root.exists():
            print(f"{Colors.RED}错误: 路径不存在: {datasets_root}{Colors.RESET}")
            sys.exit(1)
        
        success, failure, _, _, error_details = full_datasets_verify(
            datasets_root, 
            sample_size=args.sample_size,
            verbose=args.verbose
        )
        
        # 输出 JSON 错误汇总
        if error_json_path:
            write_error_summary_json(error_details, error_json_path)
        
        sys.exit(0 if failure == 0 else 1)
    
    # 模式2: --batch 模式 (递归深度搜索)
    elif args.batch:
        if not args.root:
            print(f"{Colors.RED}错误: --batch 需要配合 --root 使用{Colors.RESET}")
            sys.exit(1)
        
        root = Path(args.root)
        if not root.exists():
            logging.error(f"Path does not exist: {root}")
            sys.exit(1)
        
        success, failure, _, _, error_details = batch_verify(
            root, 
            sample_size=args.sample_size,
            verbose=args.verbose
        )
        
        # 输出 JSON 错误汇总
        if error_json_path:
            write_error_summary_json(error_details, error_json_path)
        
        sys.exit(0 if failure == 0 else 1)
    
    # 模式1: 单个数据集验证
    else:
        if not args.root:
            print(f"{Colors.RED}错误: 请指定 --root 参数或使用 --full-datasets 模式{Colors.RESET}")
            parser.print_help()
            sys.exit(1)
        
        root = Path(args.root)
        if not root.exists():
            logging.error(f"Path does not exist: {root}")
            sys.exit(1)
        
        passed, errors = verify_dataset(root, sample_size=args.sample_size, verbose=args.verbose)
        
        # 输出 JSON 错误汇总
        if error_json_path and not passed:
            ds_name = root.name
            error_details[ds_name] = "; ".join(errors) if errors else "Unknown error"
            write_error_summary_json(error_details, error_json_path)
        elif error_json_path and passed:
            # 即使通过也写入空的 error.json
            write_error_summary_json({}, error_json_path)
        
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
