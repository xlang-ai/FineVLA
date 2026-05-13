#!/usr/bin/env python3
"""Verify a converted LeRobot v2.1 dataset.

This script checks:
1. Required files exist (info.json, episodes.jsonl, episodes_stats.jsonl, tasks.jsonl)
2. Episode parquet files match episodes.jsonl
3. Video files exist and have correct frame counts
4. Timestamps are monotonically increasing
5. Data integrity (random sampling)

Usage:
    python verify_v21_dataset.py --root /path/to/v21_dataset --sample-size 3
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
import random

import jsonlines
import numpy as np
import pyarrow.parquet as pq


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


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
        table = pq.read_table(parquet_path)
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


def verify_dataset(root: Path, sample_size: int = 3, verbose: bool = False) -> bool:
    """Verify a v2.1 dataset."""
    all_errors = []
    
    logging.info(f"Verifying dataset: {root}")
    
    # Check required files
    errors = check_required_files(root)
    all_errors.extend(errors)
    if errors:
        for e in errors:
            logging.error(e)
        return False
    
    # Check info.json
    info, errors = check_info_json(root)
    all_errors.extend(errors)
    if errors:
        for e in errors:
            logging.error(e)
    
    if not info:
        return False
    
    logging.info(f"  codebase_version: {info.get('codebase_version')}")
    logging.info(f"  total_episodes: {info.get('total_episodes')}")
    logging.info(f"  fps: {info.get('fps')}")
    
    # Load episodes
    try:
        episodes = load_episodes_jsonl(root)
        logging.info(f"  episodes.jsonl entries: {len(episodes)}")
    except Exception as e:
        logging.error(f"Cannot load episodes.jsonl: {e}")
        return False
    
    # Sample episodes for detailed verification
    if len(episodes) > sample_size:
        sampled = random.sample(episodes, sample_size)
    else:
        sampled = episodes
    
    logging.info(f"Verifying {len(sampled)} sampled episodes...")
    
    for ep in sampled:
        ep_idx = ep.get("episode_index")
        length = ep.get("length")
        
        if ep_idx is None or length is None:
            all_errors.append(f"Episode missing required fields: {ep}")
            continue
        
        if verbose:
            logging.info(f"  Checking episode {ep_idx} (length={length})")
        
        # Check parquet
        errors = check_episode_parquet(root, ep_idx, length, info)
        all_errors.extend(errors)
        
        # Check videos
        errors = check_episode_videos(root, ep_idx, length, info)
        all_errors.extend(errors)
    
    # Summary
    if all_errors:
        logging.error(f"Verification FAILED with {len(all_errors)} errors:")
        for e in all_errors[:10]:  # Show first 10
            logging.error(f"  - {e}")
        if len(all_errors) > 10:
            logging.error(f"  ... and {len(all_errors) - 10} more errors")
        return False
    else:
        logging.info("Verification PASSED")
        return True


def batch_verify(root: Path, sample_size: int = 3) -> tuple[int, int]:
    """Verify all v2.1 datasets under root."""
    # Find all datasets by checking for meta/info.json
    datasets = []
    for info_path in root.rglob("meta/info.json"):
        try:
            with open(info_path, "r") as f:
                info = json.load(f)
            if info.get("codebase_version") == "v2.1":
                datasets.append(info_path.parent.parent)
        except Exception:
            pass
    
    logging.info(f"Found {len(datasets)} v2.1 datasets to verify")
    
    success = 0
    failure = 0
    
    for ds in datasets:
        if verify_dataset(ds, sample_size=sample_size):
            success += 1
        else:
            failure += 1
    
    return success, failure


def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="Verify a converted LeRobot v2.1 dataset"
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path to the v2.1 dataset or root directory containing multiple datasets",
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
        help="Verify all v2.1 datasets under root",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    root = Path(args.root)
    
    if not root.exists():
        logging.error(f"Path does not exist: {root}")
        sys.exit(1)
    
    if args.batch:
        success, failure = batch_verify(root, sample_size=args.sample_size)
        logging.info(f"Batch verification: {success} passed, {failure} failed")
        sys.exit(0 if failure == 0 else 1)
    else:
        if verify_dataset(root, sample_size=args.sample_size, verbose=args.verbose):
            sys.exit(0)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
