#!/usr/bin/env python3
"""Batch convert all LeRobot v3.0 datasets under a directory tree to v2.1 format.

This script:
1. Recursively finds all v3.0 datasets by checking meta/info.json
2. Converts each to v2.1 format in a mirrored output directory tree
3. Does NOT modify the original input data

Usage:
    python batch_convert_v30_to_v21.py \
        --input-root /path/to/v30_datasets \
        --output-root /path/to/v21_output \
        --num-workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

import jsonlines
import numpy as np
import pyarrow.parquet as pq
import tqdm

# Try to import from lerobot, fallback to local definitions
try:
    from lerobot.datasets.utils import (
        DEFAULT_CHUNK_SIZE,
        DEFAULT_DATA_PATH,
        DEFAULT_VIDEO_PATH,
        EPISODES_DIR,
        LEGACY_EPISODES_PATH,
        LEGACY_EPISODES_STATS_PATH,
        LEGACY_TASKS_PATH,
        load_info,
        load_tasks,
        serialize_dict,
        unflatten_dict,
        write_info,
    )
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    # Define fallback constants
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    EPISODES_DIR = "meta/episodes"
    LEGACY_EPISODES_PATH = "meta/episodes.jsonl"
    LEGACY_EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
    LEGACY_TASKS_PATH = "meta/tasks.jsonl"

V21 = "v2.1"
V30 = "v3.0"

LEGACY_DATA_PATH_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
LEGACY_VIDEO_PATH_TEMPLATE = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
MIN_VIDEO_DURATION = 1e-6
LEGACY_STATS_KEYS = ("mean", "std", "min", "max", "count")


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _to_serializable(value: Any) -> Any:
    """Convert numpy/pyarrow values into standard Python types for JSON dumps."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_serializable(val) for key, val in value.items()}
    return value


def load_info_standalone(root: Path) -> dict:
    """Load info.json without lerobot dependency."""
    info_path = root / "meta" / "info.json"
    with open(info_path, "r") as f:
        return json.load(f)


def write_info_standalone(info: dict, root: Path) -> None:
    """Write info.json without lerobot dependency."""
    info_path = root / "meta" / "info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)


def load_tasks_standalone(root: Path):
    """Load tasks from parquet without lerobot dependency."""
    import pandas as pd
    tasks_path = root / "meta" / "tasks.parquet"
    if tasks_path.exists():
        return pd.read_parquet(tasks_path)
    return None


def unflatten_dict_standalone(flat_dict: dict) -> dict:
    """Convert flattened dict with '/' keys back to nested dict."""
    result = {}
    for key, value in flat_dict.items():
        parts = key.split("/")
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


def serialize_dict_standalone(d: dict) -> dict:
    """Recursively serialize dict values to JSON-compatible types."""
    return _to_serializable(d)


# Use lerobot functions if available, otherwise use standalone versions
if LEROBOT_AVAILABLE:
    _load_info = load_info
    _write_info = write_info
    _load_tasks = load_tasks
    _unflatten_dict = unflatten_dict
    _serialize_dict = serialize_dict
else:
    _load_info = load_info_standalone
    _write_info = write_info_standalone
    _load_tasks = load_tasks_standalone
    _unflatten_dict = unflatten_dict_standalone
    _serialize_dict = serialize_dict_standalone


def find_v30_datasets(root: Path) -> list[Path]:
    """Find all v3.0 datasets under root by checking meta/info.json."""
    datasets = []
    for info_path in root.rglob("meta/info.json"):
        try:
            with open(info_path, "r") as f:
                info = json.load(f)
            if info.get("codebase_version") == V30:
                datasets.append(info_path.parent.parent)
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Could not read {info_path}: {e}")
    return sorted(datasets)


def load_episode_records(root: Path) -> list[dict[str, Any]]:
    """Load the consolidated metadata rows stored in meta/episodes."""
    episodes_dir = root / EPISODES_DIR
    pq_paths = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))
    if not pq_paths:
        raise FileNotFoundError(f"No episode parquet files found in {episodes_dir}.")

    records: list[dict[str, Any]] = []
    for pq_path in pq_paths:
        table = pq.read_table(pq_path)
        records.extend(table.to_pylist())

    records.sort(key=lambda rec: int(rec["episode_index"]))
    return records


def convert_tasks(root: Path, new_root: Path) -> None:
    """Convert tasks parquet to legacy JSONL format."""
    logging.info("Converting tasks parquet to legacy JSONL")
    
    tasks_parquet = root / "meta" / "tasks.parquet"
    if not tasks_parquet.exists():
        logging.warning("No tasks.parquet found, skipping tasks conversion")
        return
    
    import pandas as pd
    tasks_df = pd.read_parquet(tasks_parquet)
    
    out_path = new_root / LEGACY_TASKS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(out_path, mode="w") as writer:
        for idx, row in tasks_df.iterrows():
            task_index = row.get("task_index", idx)
            task = row.get("task", str(idx))
            writer.write({
                "task_index": int(task_index),
                "task": _to_serializable(task),
            })


def convert_info(
    root: Path,
    new_root: Path,
    episode_records: list[dict[str, Any]],
    video_keys: list[str],
) -> None:
    """Convert info.json to v2.1 schema."""
    info = _load_info(root)
    logging.info("Converting info.json metadata to v2.1 schema")

    total_episodes = info.get("total_episodes") or len(episode_records)
    chunks_size = info.get("chunks_size", DEFAULT_CHUNK_SIZE)

    info["codebase_version"] = V21

    # Restore legacy layout templates
    info["data_path"] = LEGACY_DATA_PATH_TEMPLATE
    if info.get("video_path") is not None and len(video_keys) > 0:
        info["video_path"] = LEGACY_VIDEO_PATH_TEMPLATE
    else:
        info["video_path"] = None

    # Remove v3-specific sizing hints
    info.pop("data_files_size_in_mb", None)
    info.pop("video_files_size_in_mb", None)

    # Restore per-feature metadata
    for key, ft in info["features"].items():
        if ft.get("dtype") != "video":
            ft.pop("fps", None)

    info["total_chunks"] = math.ceil(total_episodes / chunks_size) if total_episodes > 0 else 0
    info["total_videos"] = total_episodes * len(video_keys)

    _write_info(info, new_root)


def _group_episodes_by_data_file(
    episode_records: Iterable[dict[str, Any]],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    """Group episodes by their source data file."""
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in episode_records:
        key = (
            int(record["data/chunk_index"]),
            int(record["data/file_index"]),
        )
        grouped[key].append(record)
    return grouped


def convert_data(root: Path, new_root: Path, episode_records: list[dict[str, Any]]) -> None:
    """Convert consolidated parquet files back to per-episode files."""
    from datasets import Dataset
    
    logging.info("Converting consolidated parquet files back to per-episode files")
    grouped = _group_episodes_by_data_file(episode_records)
    chunks_size = DEFAULT_CHUNK_SIZE

    for (chunk_idx, file_idx), records in tqdm.tqdm(grouped.items(), desc="convert data files"):
        source_path = root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        if not source_path.exists():
            raise FileNotFoundError(f"Expected source parquet file not found: {source_path}")

        table = pq.read_table(source_path)
        records = sorted(records, key=lambda rec: int(rec["dataset_from_index"]))
        file_offset = int(records[0]["dataset_from_index"])

        for record in records:
            episode_index = int(record["episode_index"])
            start = int(record["dataset_from_index"]) - file_offset
            stop = int(record["dataset_to_index"]) - file_offset
            length = stop - start

            if length <= 0:
                raise ValueError(
                    f"Invalid episode length: episode_index={episode_index}, length={length}"
                )

            episode_table = table.slice(start, length).to_pandas()

            dest_chunk = episode_index // chunks_size
            dest_path = new_root / LEGACY_DATA_PATH_TEMPLATE.format(
                episode_chunk=dest_chunk,
                episode_index=episode_index,
            )
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            Dataset.from_pandas(episode_table).to_parquet(dest_path)


def _group_episodes_by_video_file(
    episode_records: Iterable[dict[str, Any]],
    video_key: str,
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    """Group episodes by their source video file."""
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    chunk_column = f"videos/{video_key}/chunk_index"
    file_column = f"videos/{video_key}/file_index"

    for record in episode_records:
        if chunk_column not in record or file_column not in record:
            continue
        chunk_idx = record.get(chunk_column)
        file_idx = record.get(file_column)
        if chunk_idx is None or file_idx is None:
            continue
        grouped[(int(chunk_idx), int(file_idx))].append(record)
    return grouped


def _extract_video_segment(
    src: Path,
    dst: Path,
    start: float,
    end: float,
) -> None:
    """Extract a video segment using ffmpeg."""
    duration = max(end - start, MIN_VIDEO_DURATION)
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{start:.6f}",
        "-i", str(src),
        "-t", f"{duration:.6f}",
        "-c", "copy",
        "-avoid_negative_ts", "1",
        "-y",
        str(dst),
    ]

    try:
        subprocess.run(cmd, check=True, timeout=300, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found; it is required for video conversion") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed: {exc.stderr}") from exc


def convert_videos(root: Path, new_root: Path, episode_records: list[dict[str, Any]], video_keys: list[str]) -> None:
    """Convert concatenated MP4 files back to per-episode videos."""
    if len(video_keys) == 0:
        logging.info("No video features detected; skipping video conversion")
        return

    logging.info("Converting concatenated MP4 files back to per-episode videos")
    chunks_size = DEFAULT_CHUNK_SIZE

    for video_key in video_keys:
        grouped = _group_episodes_by_video_file(episode_records, video_key)
        if len(grouped) == 0:
            logging.info(f"No video metadata found for key '{video_key}'; skipping")
            continue

        for (chunk_idx, file_idx), records in tqdm.tqdm(grouped.items(), desc=f"convert videos ({video_key})"):
            src_path = root / DEFAULT_VIDEO_PATH.format(
                video_key=video_key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )
            if not src_path.exists():
                raise FileNotFoundError(f"Expected MP4 file not found: {src_path}")

            records = sorted(records, key=lambda rec: float(rec[f"videos/{video_key}/from_timestamp"]))

            for record in records:
                episode_index = int(record["episode_index"])
                start = float(record[f"videos/{video_key}/from_timestamp"])
                end = float(record[f"videos/{video_key}/to_timestamp"])

                dest_chunk = episode_index // chunks_size
                dest_path = new_root / LEGACY_VIDEO_PATH_TEMPLATE.format(
                    episode_chunk=dest_chunk,
                    video_key=video_key,
                    episode_index=episode_index,
                )

                _extract_video_segment(src_path, dest_path, start=start, end=end)


def convert_episodes_metadata(new_root: Path, episode_records: list[dict[str, Any]]) -> None:
    """Reconstruct legacy episodes and episodes_stats JSONL files."""
    logging.info("Reconstructing legacy episodes and episodes_stats JSONL files")

    episodes_path = new_root / LEGACY_EPISODES_PATH
    stats_path = new_root / LEGACY_EPISODES_STATS_PATH
    episodes_path.parent.mkdir(parents=True, exist_ok=True)

    def _filter_stats(stats: dict[str, Any]) -> dict[str, Any]:
        """Remove v3-only statistics keys."""
        filtered: dict[str, Any] = {}
        for feature, values in stats.items():
            if not isinstance(values, dict):
                continue
            keep = {k: v for k, v in values.items() if k in LEGACY_STATS_KEYS}
            if keep:
                filtered[feature] = keep
        return filtered

    with (
        jsonlines.open(episodes_path, mode="w") as episodes_writer,
        jsonlines.open(stats_path, mode="w") as stats_writer,
    ):
        for record in sorted(episode_records, key=lambda rec: int(rec["episode_index"])):
            legacy_episode = {
                key: value
                for key, value in record.items()
                if not key.startswith("data/")
                and not key.startswith("videos/")
                and not key.startswith("stats/")
                and not key.startswith("meta/")
                and key not in {"dataset_from_index", "dataset_to_index"}
            }

            serializable_episode = {key: _to_serializable(value) for key, value in legacy_episode.items()}
            episodes_writer.write(serializable_episode)

            stats_flat = {key: record[key] for key in record if key.startswith("stats/")}
            stats_nested = _unflatten_dict(stats_flat).get("stats", {})
            stats_serialized = _serialize_dict(_filter_stats(stats_nested))
            stats_writer.write({
                "episode_index": int(record["episode_index"]),
                "stats": stats_serialized,
            })


def convert_single_dataset(src_root: Path, dst_root: Path) -> bool:
    """Convert a single v3.0 dataset to v2.1 format.
    
    Args:
        src_root: Path to the v3.0 dataset
        dst_root: Path where the v2.1 dataset will be created
        
    Returns:
        True if conversion succeeded, False otherwise
    """
    try:
        logging.info(f"Converting: {src_root}")
        logging.info(f"Output to: {dst_root}")
        
        # Create output directory
        dst_root.mkdir(parents=True, exist_ok=True)
        
        # Load episode records
        episode_records = load_episode_records(src_root)
        logging.info(f"Found {len(episode_records)} episodes")
        
        # Get video keys
        info = _load_info(src_root)
        video_keys = [key for key, ft in info["features"].items() if ft.get("dtype") == "video"]
        logging.info(f"Video keys: {video_keys}")
        
        # Convert each component
        convert_info(src_root, dst_root, episode_records, video_keys)
        convert_tasks(src_root, dst_root)
        convert_data(src_root, dst_root, episode_records)
        convert_videos(src_root, dst_root, episode_records, video_keys)
        convert_episodes_metadata(dst_root, episode_records)
        
        logging.info(f"Successfully converted: {src_root}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to convert {src_root}: {e}")
        import traceback
        traceback.print_exc()
        return False


def batch_convert(
    input_root: Path,
    output_root: Path,
    num_workers: int = 1,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Batch convert all v3.0 datasets under input_root.
    
    Args:
        input_root: Root directory containing v3.0 datasets
        output_root: Root directory for v2.1 output
        num_workers: Number of parallel workers (currently sequential due to ffmpeg)
        dry_run: If True, only list datasets without converting
        
    Returns:
        Tuple of (success_count, failure_count)
    """
    # Find all v3.0 datasets
    datasets = find_v30_datasets(input_root)
    logging.info(f"Found {len(datasets)} v3.0 datasets under {input_root}")
    
    if dry_run:
        for ds in datasets:
            rel_path = ds.relative_to(input_root)
            print(f"  - {rel_path}")
        return len(datasets), 0
    
    success_count = 0
    failure_count = 0
    
    for ds_path in datasets:
        # Compute relative path and output path
        rel_path = ds_path.relative_to(input_root)
        out_path = output_root / rel_path
        
        # Skip if output already exists
        if out_path.exists():
            logging.info(f"Skipping (already exists): {out_path}")
            success_count += 1
            continue
        
        # Convert
        if convert_single_dataset(ds_path, out_path):
            success_count += 1
        else:
            failure_count += 1
    
    return success_count, failure_count


def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="Batch convert LeRobot v3.0 datasets to v2.1 format"
    )
    parser.add_argument(
        "--input-root",
        type=str,
        required=True,
        help="Root directory containing v3.0 datasets",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Root directory for v2.1 output (mirrored structure)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, sequential)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List datasets without converting",
    )
    parser.add_argument(
        "--single",
        type=str,
        default=None,
        help="Convert only a single dataset (relative path from input-root)",
    )
    
    args = parser.parse_args()
    
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    
    if not input_root.exists():
        logging.error(f"Input root does not exist: {input_root}")
        sys.exit(1)
    
    if args.single:
        # Convert single dataset
        src = input_root / args.single
        dst = output_root / args.single
        if convert_single_dataset(src, dst):
            logging.info("Conversion completed successfully")
        else:
            logging.error("Conversion failed")
            sys.exit(1)
    else:
        # Batch convert
        success, failure = batch_convert(
            input_root,
            output_root,
            num_workers=args.num_workers,
            dry_run=args.dry_run,
        )
        
        logging.info(f"Conversion complete: {success} succeeded, {failure} failed")
        if failure > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
