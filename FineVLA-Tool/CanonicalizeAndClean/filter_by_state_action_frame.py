# -*- coding: utf-8 -*-
"""
Comprehensive filtering script: performs three checks on a single dataset (flat mode)
or a dataset containing sub-datasets (sub-dataset mode).

Three checks:
  1. Too few frames -- episode frame count < config.DATASET_MIN_FRAME[dataset] threshold
  2. Empty task -- episode's tasks field is an empty list or does not exist
  3. L2 anomaly -- range-normalized per-frame L2 distance between state and action trajectories exceeds threshold

Mode auto-detection:
  - Flat mode: <path>/meta/episodes.jsonl exists -> single dataset processing
  - Sub-dataset mode: otherwise recursively find all leaf directories containing meta/episodes.jsonl, sharing modality.json

Usage:
    python filter_by_state_action_frame.py <dataset_path> [options]

Examples:
    # Flat mode
    python filter_by_state_action_frame.py /path/to/Lerobot_v21/BC_Z --force-reconvert

    # Sub-dataset mode (auto-detected)
    python filter_by_state_action_frame.py /path/to/Galaxea-Open-World-Dataset --episodes 3

    # Sub-dataset mode - by robot type
    python filter_by_state_action_frame.py /path/to/RoboMindV2.0/franka --episodes 5

    python filter_by_state_action_frame.py \
      $VLA_DATA_ROOT/RH20T-RoboInter \
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
# Frame Count / Task Filtering
# ============================================================

def _get_min_frame(dataset_key: str):
    """Get min_frame for a dataset key: first try exact match, then match by top-level directory name."""
    if dataset_key in config.DATASET_MIN_FRAME:
        return config.DATASET_MIN_FRAME[dataset_key]
    top_level = dataset_key.split("/")[0]
    return config.DATASET_MIN_FRAME.get(top_level)


def _tasks_empty(tasks) -> bool:
    """Check if tasks is empty: None, empty list, or list of all empty strings."""
    if tasks is None:
        return True
    if isinstance(tasks, list):
        if len(tasks) == 0:
            return True
        if all(isinstance(t, str) and t.strip() == "" for t in tasks):
            return True
    return False


def _find_episode_jsonl(dataset_path: str) -> str | None:
    """Find episodes.jsonl or episode.jsonl under dataset_path/meta/."""
    meta_dir = os.path.join(dataset_path, "meta")
    if not os.path.isdir(meta_dir):
        return None
    for name in config.EPISODE_FILENAMES:
        fp = os.path.join(meta_dir, name)
        if os.path.isfile(fp):
            return fp
    return None


def _load_tasks_jsonl(dataset_path: str) -> dict:
    """Load tasks.jsonl and return a {task_index: task_string} mapping."""
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
    """Parse the task list from an episode object.
    Prefers the inline "tasks" field; falls back to looking up "task_index" in tasks.jsonl if absent.
    """
    tasks = obj.get("tasks")
    if tasks is not None:
        return tasks
    # Look up via task_index
    task_index = obj.get("task_index")
    if task_index is not None and task_map:
        task_text = task_map.get(task_index)
        if task_text is not None:
            return [task_text]
    return None


def check_frame_and_task(dataset_path: str, dataset_key: str, episodes: int = None, quiet: bool = False) -> dict:
    """Check frame count and task, return {ep_id: [reason, ...]}.

    Parameters
    ----------
    episodes : int, optional
        Only check the first N episodes. None means check all.
    """
    result = {}
    min_frame = _get_min_frame(dataset_key)

    ep_file = _find_episode_jsonl(dataset_path)
    if ep_file is None:
        if not quiet:
            print(f"[WARN] episodes.jsonl not found, skipping frame/task check")
        return result

    # Load tasks.jsonl for task_index lookup
    task_map = _load_tasks_jsonl(dataset_path)

    if not quiet:
        print(f"Frame/task check: {ep_file}")
        print(f"  min_frame threshold: {min_frame if min_frame is not None else 'not set (skipping frame check)'}")
        if episodes is not None:
            print(f"  limiting check to first {episodes} episodes")
        if task_map:
            print(f"  loaded tasks.jsonl ({len(task_map)} tasks)")

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

            # If episodes limit is specified, only process the first N
            if episodes is not None and n_total >= episodes:
                break

            n_total += 1
            episode_index = obj.get("episode_index")
            length = obj.get("length", 0)
            tasks = _resolve_episode_tasks(obj, task_map)
            ep_id = str(episode_index)

            reasons = []
            if min_frame is not None and length < min_frame:
                reasons.append(f"frame count is {length}, below threshold {min_frame}")
                n_frame_bad += 1
            if _tasks_empty(tasks):
                reasons.append("task is empty")
                n_task_bad += 1

            if reasons:
                result[ep_id] = reasons

    if not quiet:
        print(f"  total {n_total} episodes, frame count insufficient: {n_frame_bad}, task empty: {n_task_bad}")
    return result


# ============================================================
# Parquet File Integrity Check
# ============================================================

def check_corrupted_parquets(dataset_path: str, episodes: int = None, quiet: bool = False) -> dict:
    """Scan episode parquet files under data/, detecting corrupted files (0-byte, unreadable schema, etc.).
    episodes: only check the first N files. None means all.
    Returns {ep_id: [reason, ...]}.
    """
    import pyarrow.parquet as pq_check

    data_dir = os.path.join(dataset_path, "data")
    if not os.path.isdir(data_dir):
        if not quiet:
            print(f"[WARN] data/ directory not found, skipping parquet integrity check")
        return {}

    parquet_files = sorted(glob_mod.glob(os.path.join(data_dir, "**", "episode_*.parquet"),
                                         recursive=True))
    total = len(parquet_files)
    if episodes is not None and episodes < total:
        parquet_files = parquet_files[:episodes]
    if not quiet:
        print(f"Parquet integrity check: checking {len(parquet_files)}/{total} parquet files")

    result = {}
    n_corrupted = 0
    for pf in parquet_files:
        ep_name = os.path.basename(pf)
        # Extract episode id from filename: episode_000118.parquet -> 118
        ep_id = ep_name.replace("episode_", "").replace(".parquet", "").lstrip("0") or "0"

        reason = None
        try:
            fsize = os.path.getsize(pf)
        except OSError as e:
            reason = f"parquet file inaccessible: {e}"
        else:
            if fsize == 0:
                reason = "parquet file corrupted (0 bytes)"
            else:
                try:
                    pq_check.read_schema(pf)
                except Exception as e:
                    reason = f"parquet file corrupted (schema unreadable): {e}"

        if reason:
            result[ep_id] = [reason]
            n_corrupted += 1

    if not quiet:
        print(f"  corrupted files: {n_corrupted}")
    return result


# ============================================================
# L2 State-Action Filtering
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
    """Normalize slot configuration to list form. Accepts a string or list."""
    if isinstance(slot, str):
        return [slot]
    if isinstance(slot, list):
        return slot
    raise ValueError(f"slot config must be a string or list, got: {type(slot)}")


def _get_compare_slots(dataset_key: str):
    """Return (state_slots, action_slots), both in list form."""
    slot_cfg = config.STATE_ACTION_COMPARE_SLOTS.get(dataset_key)
    if slot_cfg is None:
        return None
    state_slots = _normalize_slot_list(slot_cfg["state_slot"])
    action_slots = _normalize_slot_list(slot_cfg["action_slot"])
    for s in state_slots + action_slots:
        if s in GRIPPER_SLOTS:
            print(f"[ERROR] Config for {dataset_key} uses gripper slot '{s}', skipped.", file=sys.stderr)
            return None
    return state_slots, action_slots


def _ensure_unified_output(dataset_path: str, episodes: int = None, force: bool = False) -> str:
    output_dir = os.path.join(dataset_path, "unified_output")
    meta_path = os.path.join(output_dir, "unified_meta.json")

    if not force:
        if os.path.isdir(output_dir) and os.path.isfile(meta_path):
            parquets = glob_mod.glob(os.path.join(output_dir, "episode_*.parquet"))
            if parquets:
                print(f"Existing unified_output ({len(parquets)} parquets), reusing directly."
                      f" Add --force-reconvert to regenerate.")
                return output_dir

    print(f"Calling convert_unified.py to generate/complete unified_output...")
    convert_script = os.path.join(SCRIPT_DIR, "convert_unified.py")
    cmd = [
        sys.executable, convert_script,
        dataset_path,
        "--output-dir", output_dir,
        "--skip-existing",
    ]
    if episodes is not None:
        cmd += ["--episodes", str(episodes)]
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] convert_unified.py failed:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"convert_unified.py failed (exit code {result.returncode})")
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
    """Extract and concatenate multiple slot ranges from a unified parquet file.

    Parameters
    ----------
    parquet_path : str
    state_ranges : list of (start, end) -- dimension ranges for state slots
    action_ranges : list of (start, end) -- dimension ranges for action slots
    """
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    episode_name = os.path.basename(parquet_path)

    state_all = np.array(df["state_unified"].tolist(), dtype=np.float64)
    action_all = np.array(df["action_unified"].tolist(), dtype=np.float64)
    mask_s_all = np.array(df["mask_state"].tolist(), dtype=bool)
    mask_a_all = np.array(df["mask_action"].tolist(), dtype=bool)
    frame_indices = df["frame_index"].values.astype(int)

    # Concatenate multiple slot ranges
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
        print(f"[WARN] Skipping corrupted unified parquet {parquet_path}: {e}")
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
        print(f"[WARN] Skipping corrupted unified parquet {parquet_path}: {e}")
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
    """Perform L2 check, return (l2_reasons, l2_report_info).
    l2_reasons: {ep_id: [reason, ...]}
    l2_report_info: detailed report dict (statistics, etc.)
    """
    slots = _get_compare_slots(dataset_key)
    if slots is None:
        print(f"Dataset {dataset_key} has no state-action comparison slot configured or config is None, skipping L2 check.")
        return {}, {}

    # Check if modality.json exists
    modality_path = os.path.join(dataset_path, "meta", "modality.json")
    if not os.path.isfile(modality_path):
        print(f"[WARN] Dataset missing meta/modality.json, cannot perform L2 check.")
        print(f"       To perform L2 check, please create the modality.json file first.")
        return {}, {}

    state_slots, action_slots = slots
    state_slot_label = "+".join(state_slots)
    action_slot_label = "+".join(action_slots)
    print(f"\nL2 check: state.{state_slot_label} vs action.{action_slot_label}")
    print(f"  Threshold: {threshold if threshold is not None else 'none (statistics only)'}")

    # Ensure unified_output exists
    output_dir = _ensure_unified_output(dataset_path, episodes, force=force_reconvert)

    # Read unified_meta.json
    slot_layout = _load_unified_meta(output_dir)
    for s in state_slots:
        if s not in slot_layout:
            raise RuntimeError(f"state_slot '{s}' not in slot_layout. Available: {list(slot_layout.keys())}")
    for s in action_slots:
        if s not in slot_layout:
            raise RuntimeError(f"action_slot '{s}' not in slot_layout. Available: {list(slot_layout.keys())}")

    # Build dimension range lists for multiple slots
    state_ranges = [(slot_layout[s]["start"], slot_layout[s]["end"]) for s in state_slots]
    action_ranges = [(slot_layout[s]["start"], slot_layout[s]["end"]) for s in action_slots]

    for s, (start, end) in zip(state_slots, state_ranges):
        print(f"  State slot [{s}]: dims [{start}:{end})")
    for s, (start, end) in zip(action_slots, action_ranges):
        print(f"  Action slot [{s}]: dims [{start}:{end})")

    # Prepare episode list
    parquet_files = _list_unified_parquets(output_dir)
    n_episodes = min(episodes, len(parquet_files)) if episodes is not None else len(parquet_files)
    parquet_files = parquet_files[:n_episodes]
    print(f"  Total {n_episodes} episodes to process")

    # Parallel L2 computation
    num_workers = min(config.NUM_WORKERS, n_episodes)
    print(f"  Parallel workers: {num_workers}")

    worker_fn = partial(_process_episode_worker,
                        state_ranges=state_ranges, action_ranges=action_ranges)

    t_start = time.time()
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(worker_fn, parquet_files)
    t_l2 = time.time() - t_start
    print(f"  L2 parallel computation done, took {t_l2:.1f}s ({n_episodes} episodes, {num_workers} workers)")

    # Summary
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
            print(f"    [{i+1}/{n_episodes}] {r['episode_name']}: no valid frames (mask not satisfied)")

    # Plotting
    if plot:
        print(f"\n  Starting plots...")
        t_plot = time.time()
        for i, pf in enumerate(parquet_files):
            ep_local = process_episode_local(pf, state_ranges, action_ranges)
            png_path = plot_episode(ep_local, state_slot_label, action_slot_label, plot_dir)
            if png_path:
                print(f"    [{i+1}/{n_episodes}] {os.path.basename(pf)} -> {png_path}")
        print(f"  Plotting done, took {time.time() - t_plot:.1f}s")

    # Statistics (filter out inf to avoid skewing mean/percentiles)
    l2_arr = np.array(l2_scores)
    l2_finite = l2_arr[np.isfinite(l2_arr)]
    n_inf = int(np.sum(~np.isfinite(l2_arr)))
    if n_inf > 0:
        print(f"  Note: {n_inf} episodes have L2 = inf (one side stationary while the other is not)")
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
        print(f"  L2 stats: mean={stats['mean_l2']:.4f}, median={stats['median_l2']:.4f}, "
              f"p90={stats['p90_l2']:.4f}, p95={stats['p95_l2']:.4f}, max={stats['max_l2']:.4f}")

    # Collect L2 reasons
    l2_reasons = {}
    for r in results:
        ep_name = r["episode_name"]
        score = r["l2_score"]
        ep_id = ep_name.replace("episode_", "").replace(".parquet", "").lstrip("0") or "0"

        reasons = []
        if r.get("corrupted"):
            reasons.append("parquet file corrupted (unified_output file unreadable)")
        elif r["valid_frames"] == 0 or score is None:
            reasons.append("no valid frames (state/action mask not satisfied)")
        elif np.isnan(score):
            reasons.append(f"too few frames (valid_frames={r['valid_frames']}), cannot compute L2")
        elif threshold is not None and score > threshold:
            reasons.append(f"L2={score:.4f}, exceeds threshold {threshold}")

        if reasons:
            l2_reasons[ep_id] = reasons

    # Detailed report
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
# Summary Statistics (shared by flat and sub-dataset modes)
# ============================================================

def _compute_summary(merged: dict, dataset_key: str, dataset_path: str,
                     threshold, l2_report_info: dict) -> dict:
    """Compute summary statistics from the merged reasons."""
    n_frame_task_only = sum(1 for v in merged.values()
                            if all("frame" in r or "task" in r for r in v))
    n_l2_only = sum(1 for v in merged.values()
                     if all("L2" in r or "no valid frames" in r or "cannot compute" in r for r in v))
    n_both = len(merged) - n_frame_task_only - n_l2_only

    n_frame_bad = sum(1 for v in merged.values()
                      if any("frame" in r for r in v))
    n_task_bad = sum(1 for v in merged.values()
                     if any("task" in r for r in v))
    n_l2_bad = sum(1 for v in merged.values()
                    if any("L2" in r or "no valid frames" in r or "cannot compute" in r for r in v))
    n_corrupted = sum(1 for v in merged.values()
                      if any("parquet file corrupted" in r or "parquet file inaccessible" in r for r in v))

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
    """Merge multiple reasons dicts (frame/task, L2, parquet corruption, etc.)."""
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
# Flat Mode (original logic)
# ============================================================

def _run_flat_mode(dataset_path: str, args):
    """Flat dataset mode: top-level has meta/episodes.jsonl."""
    plot = args.plot or config.PLOT
    dataset_key = _resolve_dataset_key(dataset_path)

    # Determine L2 threshold
    if args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = config.DATASET_L2_THRESHOLD.get(
            dataset_key, config.DEFAULT_L2_THRESHOLD)

    print(f"{'='*60}")
    print(f"Dataset: {dataset_path}")
    print(f"Config key: {dataset_key}")
    print(f"Mode: flat")
    print(f"L2 threshold: {threshold if threshold is not None else 'none (report only)'}")
    print(f"{'='*60}")

    # 1. Parquet integrity check
    print(f"\n--- Step 1: Parquet integrity check ---")
    corrupted_reasons = check_corrupted_parquets(dataset_path, episodes=args.episodes)

    # 2. Frame count / task check
    print(f"\n--- Step 2: Frame count / task check ---")
    frame_task_reasons = check_frame_and_task(dataset_path, dataset_key, episodes=args.episodes)

    # 3. L2 check
    print(f"\n--- Step 3: L2 state-action check ---")
    plot_dir = args.plot_dir or os.path.join(SCRIPT_DIR, "plots", dataset_key)
    l2_reasons, l2_report_info = check_l2(
        dataset_path, dataset_key, args.episodes, threshold,
        args.force_reconvert, plot, plot_dir)

    # 4. Merge
    print(f"\n--- Step 4: Merge results ---")
    merged = _merge_reasons(corrupted_reasons, frame_task_reasons, l2_reasons)
    summary = _compute_summary(merged, dataset_key, dataset_path,
                               threshold, l2_report_info)

    # 5. Write report
    safe_key = dataset_key.replace(os.sep, "_").replace("/", "_")
    filter_report_path = os.path.join(SCRIPT_DIR, f"{safe_key}_filter_report.json")
    report = {
        "summary": summary,
        dataset_key: merged,
    }
    with open(filter_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 6. L2 detailed report
    if l2_report_info:
        l2_detail_path = os.path.join(SCRIPT_DIR, "state_action_diff_report.json")
        l2_report_info["dataset"] = dataset_key
        l2_report_info["dataset_path"] = dataset_path
        with open(l2_detail_path, "w", encoding="utf-8") as f:
            json.dump(l2_report_info, f, ensure_ascii=False, indent=2)
        print(f"L2 detailed report: {l2_detail_path}")

    # 7. Print summary
    bd = summary["breakdown"]
    print(f"\n{'='*60}")
    print(f"Filter report: {filter_report_path}")
    print(f"Total {summary['total_problem_episodes']} problem episodes:")
    print(f"  parquet corrupted: {bd['parquet_corrupted']}")
    print(f"  frame count insufficient: {bd['frame_too_short']}")
    print(f"  task empty: {bd['task_empty']}")
    print(f"  L2 anomaly:  {bd['l2_abnormal']}")
    print(f"  ---")
    print(f"  frame/task only: {summary['frame_task_only']}")
    print(f"  L2 only:        {summary['l2_only']}")
    print(f"  multiple issues: {summary['both']}")
    print(f"{'='*60}")
    if plot:
        print(f"Plot directory: {plot_dir}")


# ============================================================
# Sub-dataset Mode
# ============================================================


def _discover_leaf_datasets(root_path: str) -> list[str]:
    """Recursively find all leaf dataset directories under root_path that contain meta/episodes.jsonl.
    Skips SKIP_DIRS and meta directories.
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
            # Check if this directory is a leaf dataset
            if _find_episode_jsonl(full) is not None:
                leaves.append(full)
            else:
                # Continue recursing
                _walk(full)

    _walk(root_path)
    return leaves



def _process_one_subdataset(leaf_path: str, dataset_key: str, episodes: int,
                            threshold, force_reconvert: bool, plot: bool,
                            plot_dir: str) -> dict:
    """Process a single sub-dataset, return result dict.
    Returns: {"summary": {...}, "episodes": {...}, "l2_scores": [...]}
    """
    leaf_name = os.path.basename(leaf_path)

    # Parquet integrity check
    corrupted_reasons = check_corrupted_parquets(leaf_path, episodes=episodes, quiet=True)

    # Frame count / task check
    frame_task_reasons = check_frame_and_task(leaf_path, dataset_key, episodes=episodes, quiet=True)

    # L2 check
    l2_reasons, l2_report_info = check_l2(
        leaf_path, dataset_key, episodes, threshold,
        force_reconvert, plot, plot_dir)

    # Merge
    merged = _merge_reasons(corrupted_reasons, frame_task_reasons, l2_reasons)

    # Get total episode count
    ep_file = _find_episode_jsonl(leaf_path)
    total_episodes = 0
    if ep_file:
        with open(ep_file, "r", encoding="utf-8") as f:
            total_episodes = sum(1 for line in f if line.strip())

    # Statistics
    summary = _compute_summary(merged, leaf_name, leaf_path, threshold, l2_report_info)
    summary["total_episodes"] = total_episodes

    # Collect all L2 scores for global statistics
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
    """Sub-dataset mode: recursively find leaf datasets, process each, then aggregate."""
    plot = args.plot or config.PLOT
    dataset_key = _resolve_dataset_key(dataset_path)

    # Determine L2 threshold
    if args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = config.DATASET_L2_THRESHOLD.get(
            dataset_key, config.DEFAULT_L2_THRESHOLD)

    # Discover leaf datasets
    leaves = _discover_leaf_datasets(dataset_path)
    if not leaves:
        print(f"[ERROR] No sub-datasets containing meta/episodes.jsonl found under {dataset_path}.", file=sys.stderr)
        sys.exit(1)

    # Limit sub-dataset count (for testing)
    total_leaves = len(leaves)
    if args.max_subsets_num is not None and args.max_subsets_num < total_leaves:
        leaves = leaves[:args.max_subsets_num]

    print(f"{'='*60}")
    print(f"Dataset: {dataset_path}")
    print(f"Config key: {dataset_key}")
    print(f"Mode: sub-dataset")
    print(f"Found {total_leaves} sub-datasets" +
          (f", processing first {len(leaves)} this run" if len(leaves) < total_leaves else ""))
    print(f"L2 threshold: {threshold if threshold is not None else 'none (report only)'}")
    print(f"{'='*60}")

    # Process sub-datasets one by one
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

        # Check if the leaf dataset has modality.json
        leaf_modality = os.path.join(leaf_path, "meta", "modality.json")
        if not os.path.isfile(leaf_modality):
            print(f"  [SKIP] Missing meta/modality.json, skipping")
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

            # Brief progress
            print(f"  -> {n_prob} problem episodes"
                  f" (corrupted:{sub_bd['parquet_corrupted']}, frames:{sub_bd['frame_too_short']},"
                  f" task:{sub_bd['task_empty']}, L2:{sub_bd['l2_abnormal']})"
                  f"  total {sub_summary.get('total_episodes', '?')} episodes")

        except Exception as e:
            print(f"  [ERROR] Processing failed: {e}")
            n_failed += 1
            continue

    # Global L2 statistics
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

    # Global summary
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

    # Write report
    filter_report_path = os.path.join(SCRIPT_DIR, f"{dataset_key}_filter_report.json")
    report = {
        "summary": global_summary,
        "subdatasets": subdatasets_results,
    }
    with open(filter_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Print global summary
    print(f"\n{'='*60}")
    print(f"Filter report: {filter_report_path}")
    print(f"Sub-datasets: {n_processed} succeeded, {n_failed} failed (total {len(leaves)})")
    print(f"Total episodes: {total_episodes_global}")
    print(f"Total {total_problem_eps} problem episodes:")
    print(f"  parquet corrupted: {total_corrupted}")
    print(f"  frame count insufficient: {total_frame_bad}")
    print(f"  task empty: {total_task_bad}")
    print(f"  L2 anomaly:  {total_l2_bad}")
    if global_l2_stats:
        print(f"  L2 global stats: mean={global_l2_stats['mean_l2']:.4f}, "
              f"median={global_l2_stats['median_l2']:.4f}, "
              f"p90={global_l2_stats['p90_l2']:.4f}, "
              f"p95={global_l2_stats['p95_l2']:.4f}, "
              f"max={global_l2_stats['max_l2']:.4f}")
    print(f"{'='*60}")


# ============================================================
# main: Auto-detect Mode
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive filtering: frame count + task + L2(state, action), outputs {dataset}_filter_report.json. "
                    "Auto-detects flat/sub-dataset mode.")
    parser.add_argument("dataset_path", help="LeRobot v2.1 dataset path (flat or parent directory containing sub-datasets)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of episodes to process (default: all). In sub-dataset mode, limits per sub-dataset")
    parser.add_argument("--threshold", type=float, default=None,
                        help="L2 distance threshold; exceeding marks as anomalous (overrides the dataset's config threshold)")
    parser.add_argument("--plot", action="store_true",
                        help="Output state vs action comparison plots for each episode")
    parser.add_argument("--plot-dir", type=str, default=None,
                        help="Plot output directory (default: Filter/plots/<dataset_key>/)")
    parser.add_argument("--force-reconvert", action="store_true",
                        help="Force regenerate unified_output (incremental: skips existing parquets, only fills in missing ones)")
    parser.add_argument("--max-subsets-num", type=int, default=None,
                        help="Max number of sub-datasets to process in sub-dataset mode (default: all). For quick testing")
    args = parser.parse_args()

    dataset_path = args.dataset_path.rstrip("/")

    # Auto-detect mode
    if _find_episode_jsonl(dataset_path) is not None:
        # Flat mode
        _run_flat_mode(dataset_path, args)
    else:
        # Sub-dataset mode
        _run_subdataset_mode(dataset_path, args)


if __name__ == "__main__":
    main()
