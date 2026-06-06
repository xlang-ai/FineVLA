#!/usr/bin/env python3
"""
Read *filter_report.json from a dataset, exclude problematic episodes,
then randomly sample a specified number of episodes, generating records in the same
format as cluster_representation_summary.json, and merge them into the output file.

Usage:
    # droid dataset (simple), sample 10k
    python sample_filtered_episodes.py \
        --dataset_root $VLA_DATA_ROOT/droid_1.0.1 \
        --dataset_label droid_1.0.1 \
        --id_prefix droid \
        --process_type simple \
        --sample_n 10000 \
        --output cluster_representation_summary_droid.json

    # droid_RoboInter dataset (extract raw_steps), sample 10k
    python sample_filtered_episodes.py \
        --dataset_root $VLA_DATA_ROOT/droid_RoboInter \
        --dataset_label droid_robointer \
        --id_prefix droid_robointer \
        --process_type robointer \
        --sample_n 10000 \
        --output cluster_representation_summary_droid_robointer.json

    # Other datasets follow the same pattern, just change --dataset_root etc.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

from utils.sample_all import (
    read_jsonl,
    extract_views,
    get_video_paths,
    get_resolution,
    make_record,
)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import pyarrow.parquet as pq_reader
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

import gc
import resource

def _raise_fd_limit():
    """Try to raise the file descriptor limit for the process."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            print(f"File descriptor limit: {soft} -> {hard}")
    except Exception:
        pass

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

SEED = 42


def find_filter_reports(dataset_root: str) -> list[str]:
    """Find all files ending with filter_report.json under dataset_root (non-recursive)."""
    results = []
    for fname in os.listdir(dataset_root):
        if fname.endswith("filter_report.json") and os.path.isfile(os.path.join(dataset_root, fname)):
            results.append(os.path.join(dataset_root, fname))
    return sorted(results)


def extract_problem_episodes(report_path: str) -> set[int]:
    """Extract problematic episode index set from filter_report.json.

    filter_report.json format:
      { "summary": {...}, "<dataset_name>": { "<episode_idx>": [...reasons], ... } }
    Excludes the "summary" key; all episode indices under other keys are problematic.
    """
    with open(report_path, encoding="utf-8") as f:
        data = json.load(f)

    problem_indices = set()
    for key, val in data.items():
        if key == "summary":
            continue
        if isinstance(val, dict):
            for ep_idx_str in val.keys():
                try:
                    problem_indices.add(int(ep_idx_str))
                except ValueError:
                    pass
    return problem_indices


def _get_instruction(task_map: dict[int, str], ep: dict) -> str | None:
    """Get instruction from the pre-loaded task_map, avoiding reading tasks.jsonl every time."""
    # Prefer the episode's own tasks field
    tasks = ep.get("tasks")
    if tasks:
        for t in tasks:
            if t and t.strip():
                return t

    # If no tasks field, look up from task_map
    ti = ep.get("task_index")
    if ti is not None and ti in task_map:
        text = task_map[ti]
        if text and text.strip():
            return text
    return None


def _extract_robointer_steps(dataset_root: Path, info: dict, ep: dict) -> tuple[str | None, list | None, int, str]:
    """Extract raw_steps (annotation.substask + annotation.time_clip) from RoboInter-type dataset parquet files.

    Returns: (instruction, steps_raw, has_raw_steps, step_source)
    """
    if not HAS_PANDAS:
        return None, None, 0, "none"

    ei = ep["episode_index"]
    chunks_size = info.get("chunks_size", 1000)
    data_tpl = info.get("data_path", "")
    chunk = ei // chunks_size
    pq_rel = data_tpl.format(episode_chunk=chunk, episode_index=ei)
    pq_path = dataset_root / pq_rel

    instruction = None
    steps_raw = None

    # Wrap exists check in try-except to avoid crashes on filesystem issues
    try:
        file_exists = pq_path.exists()
    except OSError:
        file_exists = False

    if file_exists:
        try:
            # Use pyarrow to explicitly open/close files, avoiding file descriptor leaks
            if HAS_PYARROW:
                pf = pq_reader.ParquetFile(str(pq_path))
                try:
                    table = pf.read(columns=["frame_index", "annotation.substask", "annotation.time_clip"])
                    df = table.to_pandas()
                finally:
                    # Explicitly close the underlying file handle
                    if hasattr(pf, 'reader') and hasattr(pf.reader, 'close'):
                        pf.reader.close()
                    del pf
            else:
                df = pd.read_parquet(pq_path, columns=["frame_index", "annotation.substask", "annotation.time_clip"])

            if not df.empty and "annotation.substask" in df.columns:
                # Parse time_clip (stored as string, e.g. "[[0, 286]]")
                if "annotation.time_clip" in df.columns:
                    time_clip_str = df["annotation.time_clip"].iloc[0]
                    if time_clip_str:
                        try:
                            # Parse JSON-formatted time_clip
                            time_clip = json.loads(time_clip_str)
                            if time_clip and isinstance(time_clip, list) and len(time_clip) > 0:
                                # Build steps_raw, each time_clip segment as one step
                                steps_raw = []
                                for i, clip in enumerate(time_clip):
                                    if isinstance(clip, list) and len(clip) == 2:
                                        start_frame, end_frame = clip[0], clip[1]
                                        # Get substask for this time segment (read substask at segment's first frame)
                                        segment_rows = df[df["frame_index"] == start_frame]
                                        if not segment_rows.empty:
                                            substask = segment_rows.iloc[0]["annotation.substask"]
                                        else:
                                            # If exact frame not found, use the first available substask
                                            substask = df["annotation.substask"].iloc[0]

                                        steps_raw.append({
                                            "i": i,
                                            "desc": substask,
                                            "start": start_frame,
                                            "end": end_frame,
                                        })

                                # Use the first substask as the overall instruction
                                if steps_raw:
                                    instruction = steps_raw[0]["desc"]
                        except (json.JSONDecodeError, ValueError):
                            pass
            del df
        except Exception:
            pass

    has_raw_steps = 1 if steps_raw else 0
    step_source = "parquet.annotation.substask" if steps_raw else "none"
    return instruction, steps_raw, has_raw_steps, step_source


def build_record_for_episode(
    ep: dict,
    info: dict,
    views: list[str],
    ds_dir: str,
    dataset_label: str,
    id_prefix: str,
    task_map: dict[int, str],
    features: dict,
    dataset_root: Path = None,
    process_type: str = "simple",
) -> dict | None:
    """Build a record in the same format as cluster_representation_summary.json for one episode.

    Args:
        process_type: Dataset processing type
            - "simple": normal dataset, get instruction from tasks.jsonl
            - "robointer": RoboInter type, extract annotation.substask and time_clip from parquet
    """
    ei = ep["episode_index"]

    videos = get_video_paths(info, views, ei)

    duration_sec = None
    if ep.get("length") and info.get("fps"):
        duration_sec = round(ep["length"] / info["fps"], 3)

    # Extract instruction and raw_steps based on process_type
    if process_type == "robointer" and dataset_root:
        instruction_from_steps, steps_raw, has_raw_steps, step_source = _extract_robointer_steps(
            dataset_root, info, ep
        )
        # If extraction from parquet failed, use task_map as fallback
        instruction = instruction_from_steps or _get_instruction(task_map, ep)
    else:
        instruction = _get_instruction(task_map, ep)
        steps_raw = None
        has_raw_steps = 0
        step_source = "none"

    return make_record(
        sample_id=f"{id_prefix}-{ei}",
        dataset=dataset_label,
        dataset_dir=ds_dir,
        views=views,
        videos=videos,
        instruction=instruction,
        duration_sec=duration_sec,
        fps=info.get("fps"),
        resolution=get_resolution(features, views),
        robot_type=info.get("robot_type", "unknown"),
        has_raw_steps=has_raw_steps,
        step_source=step_source,
        steps_raw=steps_raw,
    )


def main():
    p = argparse.ArgumentParser(description="Filter and sample episodes from a dataset")
    p.add_argument("--dataset_root", type=str, required=True,
                   help="Dataset root directory, e.g. .../droid_1.0.1")
    p.add_argument("--dataset_label", type=str, default=None,
                   help="Dataset label (default: directory name)")
    p.add_argument("--id_prefix", type=str, default=None,
                   help="sample_id prefix (default: directory name)")
    p.add_argument("--process_type", type=str, default="simple",
                   choices=["simple", "robointer"],
                   help="Dataset processing type: simple (normal), robointer (RoboInter type, extract annotation.substask)")
    p.add_argument("--sample_n", type=int, default=10000,
                   help="Number of samples (default: 10000)")
    p.add_argument("--output", type=str,
                   default="cluster_representation_summary.json",
                   help="Output JSON path")
    p.add_argument("--append", action="store_true", default=True,
                   help="Append to existing output file if it exists (default behavior)")
    p.add_argument("--no-append", dest="append", action="store_false",
                   help="Overwrite output file instead of appending")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    _raise_fd_limit()

    dataset_root = Path(args.dataset_root).resolve()
    dataset_label = args.dataset_label or dataset_root.name
    id_prefix = args.id_prefix or dataset_root.name
    lerobot_root = Path(os.environ.get("VLA_DATA_ROOT", "/path/to/your/Lerobot_v21"))

    # 1. Find filter_report.json
    reports = find_filter_reports(str(dataset_root))
    if not reports:
        print(f"[ERROR] No *filter_report.json found in {dataset_root}")
        sys.exit(1)
    print(f"Found {len(reports)} filter_report(s):")
    for r in reports:
        print(f"  {r}")

    # 2. Extract problematic episodes
    problem_episodes: set[int] = set()
    for rp in reports:
        problems = extract_problem_episodes(rp)
        print(f"  {Path(rp).name}: {len(problems)} problematic episodes")
        problem_episodes |= problems
    print(f"Total {len(problem_episodes)} problematic episodes (deduplicated)")

    # 3. Read dataset metadata
    info_path = dataset_root / "meta" / "info.json"
    eps_path = dataset_root / "meta" / "episodes.jsonl"
    if not info_path.exists() or not eps_path.exists():
        print(f"[ERROR] Missing meta/info.json or meta/episodes.jsonl")
        sys.exit(1)

    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)
    all_episodes = read_jsonl(eps_path)
    print(f"Total episodes in dataset: {len(all_episodes)}")

    # Read task_map
    task_map: dict[int, str] = {}
    tp = dataset_root / "meta" / "tasks.jsonl"
    if tp.exists():
        for item in read_jsonl(tp):
            task_map[item["task_index"]] = item["task"]

    # 4. Pre-compute common fields
    features = info.get("features", {})
    views = extract_views(features)
    if not views:
        print(f"[ERROR] No video views in dataset")
        sys.exit(1)


    try:
        ds_dir = str(dataset_root.relative_to(lerobot_root))
    except ValueError:
        ds_dir = str(dataset_root)

    # 5. Filter
    valid_episodes = [ep for ep in all_episodes if ep["episode_index"] not in problem_episodes]
    print(f"Remaining after filtering: {len(valid_episodes)} episodes")

    # 6. Group by task
    print("Grouping by task...")
    rng = random.Random(args.seed)
    task_to_eps: dict[str, list[dict]] = defaultdict(list)
    no_task_eps: list[dict] = []
    for ep in valid_episodes:
        instr = _get_instruction(task_map, ep)
        if instr and instr.strip():
            task_to_eps[instr].append(ep)
        else:
            no_task_eps.append(ep)

    print(f"Unique tasks: {len(task_to_eps)}, episodes without task: {len(no_task_eps)} (excluded)")

    # 7. First sample at the task level, then randomly pick one episode per task
    # This avoids needing to select representative episodes for all tasks (reduces memory and file access)
    all_tasks = list(task_to_eps.keys())
    sample_n = min(args.sample_n, len(all_tasks))
    sampled_tasks = rng.sample(all_tasks, sample_n)
    print(f"Sampled {sample_n} from {len(all_tasks)} tasks")

    # Randomly pick one episode per sampled task
    sampled_episodes = []
    for task_text in sampled_tasks:
        eps = task_to_eps[task_text]
        sampled_episodes.append(rng.choice(eps))

    # Sort by episode_index for easier debugging
    sampled_episodes.sort(key=lambda x: x["episode_index"])
    print(f"Selected {len(sampled_episodes)} episodes (one random per task)")

    # 8. Build records (with progress bar)
    records = []
    skipped = 0
    iterator = tqdm(sampled_episodes, desc="Processing episodes") if HAS_TQDM else sampled_episodes
    for idx, ep in enumerate(iterator):
        rec = build_record_for_episode(
            ep, info, views, ds_dir, dataset_label, id_prefix, task_map, features,
            dataset_root=dataset_root, process_type=args.process_type,
        )
        if rec:
            records.append(rec)
        else:
            skipped += 1

        # Run garbage collection every 50 episodes to release lingering file handles
        if (idx + 1) % 50 == 0:
            gc.collect()
    print(f"\nSuccessfully built {len(records)} records, skipped {skipped}")
    if args.process_type == "robointer":
        steps_count = sum(1 for r in records if r.get("has_raw_steps", 0) == 1)
        print(f"  Of which {steps_count} contain raw_steps annotations")

    # 9. Write output (supports append)
    existing = []
    if args.append and os.path.isfile(args.output):
        with open(args.output, encoding="utf-8") as f:
            existing = json.load(f)
        print(f"Existing records: {len(existing)}, append mode")

    all_records = existing + records
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(all_records)} records to {args.output}")
    print(f"  (existing {len(existing)} + new {len(records)})")

    # Final garbage collection
    gc.collect()


if __name__ == "__main__":
    main()
