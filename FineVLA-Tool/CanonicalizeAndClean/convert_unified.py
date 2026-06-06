"""
General-purpose script: reads a LeRobot v2.1 dataset and calls UnifyJointAction to convert
state/action into a unified 80-dimensional representation.

============================================================
Two Running Modes
============================================================

1. Dry-run mode (--dry-run)
   - Only reads data, performs conversion, and prints result summary
   - Does not write any files or modify original data
   - Purpose: quickly verify whether modality.json configuration is correct and conversion results are reasonable

2. Normal mode (without --dry-run)
   - Reads data, performs conversion, and prints result summary
   - Saves conversion results as .parquet files to the directory specified by --output-dir
   - Also generates unified_meta.json recording vector layout and target types
   - If --output-dir is not specified, defaults to <dataset_path>/unified_output/
   - Does not modify original parquet data

============================================================
Target Representation Types (hardcoded)
============================================================

The unified target types are fixed in the script as:
  - joint state:  abs_joint   (absolute joint angles)
  - eef state:    abs_rotvec  (absolute pose, rotation as rotation vector)
  - joint action: abs_joint   (absolute joint angles)
  - eef action:   abs_rotvec  (absolute pose, rotation as rotation vector)
To modify, edit the TARGET_* constants in main().

============================================================
Usage
============================================================

    python convert_unified.py <dataset_path> [--episodes N] [--dry-run] [--output-dir DIR]

Examples:
    # dry-run: test the first 2 episodes, only view results without writing files
    python convert_unified.py /path/to/dataset --episodes 2 --dry-run

    # normal mode: convert the first 5 episodes, output to a specified directory
    python convert_unified.py /path/to/dataset --episodes 5 --output-dir /tmp/unified_output

    # normal mode: convert all episodes, output to unified_output/ under the dataset
    python convert_unified.py /path/to/dataset --episodes 9999
"""

import sys
import os
import argparse
import glob
import json
import multiprocessing
from functools import partial
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))
from utils.UnifyJointAction import (
    UnifiedStateActionTransform,
    UNIFIED_STATE_ACTION_DIM,
    UNIFIED_STATE_ACTION_INDICES,
)


def load_modality(dataset_path: str) -> dict:
    modality_path = os.path.join(dataset_path, "meta", "modality.json")
    if not os.path.exists(modality_path):
        raise FileNotFoundError(f"modality.json not found: {modality_path}")
    with open(modality_path, encoding="utf-8") as f:
        return json.load(f)


def load_info(dataset_path: str) -> dict:
    """Read meta/info.json and return the parsed dict."""
    info_path = os.path.join(dataset_path, "meta", "info.json")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"info.json not found: {info_path}")
    with open(info_path, encoding="utf-8") as f:
        return json.load(f)


def list_episodes(dataset_path: str) -> list[str]:
    """List all episode parquet files in the dataset.
    Prefers computing paths from info.json; falls back to filesystem scan if files are missing (non-contiguous numbering).
    """
    info = load_info(dataset_path)
    total_episodes = info["total_episodes"]
    chunks_size = info.get("chunks_size", 1000)
    data_path_template = info.get("data_path")

    # First try generating paths by contiguous numbering
    files = []
    for ep_idx in range(total_episodes):
        chunk_idx = ep_idx // chunks_size
        if data_path_template:
            rel_path = data_path_template.format(episode_chunk=chunk_idx, episode_index=ep_idx)
        else:
            rel_path = f"data/chunk-{chunk_idx:03d}/episode_{ep_idx:06d}.parquet"
        files.append(os.path.join(dataset_path, rel_path))

    # Check if the first few files exist; if any are missing, fall back to filesystem scan
    if files and not os.path.isfile(files[min(1, len(files) - 1)]):
        files = sorted(glob.glob(os.path.join(dataset_path, "data", "**", "episode_*.parquet"),
                                 recursive=True))

    if not files:
        raise FileNotFoundError(f"No episodes found in {dataset_path} (total_episodes=0)")
    return files


# Read data from a parquet file and map it to the format required by UnifyJointAction according to modality_cfg.
def extract_episode_data(parquet_path: str, modality_cfg: dict) -> tuple[dict, np.ndarray]:
    """
    Read data from a parquet file and map it to the format required by UnifyJointAction
    according to modality_cfg.

    Returns:
        data dict: keys are "state.<unified_name>" / "action.<unified_name>", values are (N, D) np.ndarray
        frame_index: (N,) int64 ndarray
    """
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    N = len(df)
    data = {}

    frame_index = df["frame_index"].values.astype(np.int64)

    for category in ["state", "action"]:
        cfg = modality_cfg.get(category, {})
        for unified_name, field_info in cfg.items():
            original_key = field_info["original_key"]
            if original_key not in df.columns:
                raise KeyError(f"Column '{original_key}' not found in parquet. "
                               f"Available: {list(df.columns)}")

            col = df[original_key]
            first_val = col.iloc[0]

            if isinstance(first_val, np.ndarray):
                arr = np.stack(col.values).astype(np.float32)
            elif isinstance(first_val, (float, int, np.floating, np.integer)):
                arr = col.values.astype(np.float32).reshape(N, 1)
            elif isinstance(first_val, list):
                arr = np.array(col.tolist(), dtype=np.float32)
            else:
                raise TypeError(f"Unsupported type for '{original_key}': {type(first_val)}")

            if arr.ndim == 1:
                arr = arr.reshape(N, 1)

            # Support extracting a subset from a large array by indices list or start/end slice
            indices = field_info.get("indices")
            col_start = field_info.get("start")
            col_end = field_info.get("end")
            if indices is not None:
                arr = arr[:, indices]
            elif col_start is not None and col_end is not None:
                arr = arr[:, col_start:col_end]

            expected_dim = field_info.get("dim")
            # Verify that the dim in modality_cfg is correct and matches the action constraints
            if expected_dim is not None and arr.shape[1] != expected_dim:
                raise ValueError(
                    f"{category}.{unified_name}: expected dim={expected_dim}, "
                    f"got {arr.shape[1]} from '{original_key}'"
                )

            data[f"{category}.{unified_name}"] = arr

    return data, frame_index


def build_apply_to(modality_cfg: dict) -> list[str]:
    """Build the apply_to list: list all state keys first, then all action keys."""
    keys = []
    for unified_name in modality_cfg.get("state", {}):
        keys.append(f"state.{unified_name}")
    for unified_name in modality_cfg.get("action", {}):
        keys.append(f"action.{unified_name}")
    return keys


def print_result(result: dict, episode_path: str):
    """Print a summary of the conversion result, randomly sampling one frame to show both state and action."""
    ep_name = os.path.basename(episode_path)
    unified_s = result["state.unified"]
    unified_a = result["action.unified"]
    mask_s = result["mask.state"]
    mask_a = result["mask.action"]

    N = unified_s.shape[0]
    sample_idx = np.random.randint(0, N)

    print(f"\n{'='*70}")
    print(f"  Episode: {ep_name}  |  frames={N}  |  sample frame={sample_idx}")
    print(f"{'='*70}")

    print(f"\n  State unified: shape={unified_s.shape}, dtype={unified_s.dtype}")
    print(f"  Action unified: shape={unified_a.shape}, dtype={unified_a.dtype}")

    print(f"\n  {'Slot':15s} | {'Range':8s} | {'S dims':6s} | {'A dims':6s} | {'State[sample]':50s} | {'Action[sample]'}")
    print(f"  {'-'*130}")
    for name, (lo, hi) in UNIFIED_STATE_ACTION_INDICES.items():
        s_active = int(mask_s[sample_idx, lo:hi].sum())
        a_active = int(mask_a[sample_idx, lo:hi].sum())
        s_sample = "(empty)"
        a_sample = "(empty)"
        if s_active > 0:
            s_sample = np.array2string(unified_s[sample_idx, lo:lo+s_active], precision=4, suppress_small=True, max_line_width=200)
        if a_active > 0:
            a_sample = np.array2string(unified_a[sample_idx, lo:lo+a_active], precision=4, suppress_small=True, max_line_width=200)
        print(f"  {name:15s} | [{lo:2d}:{hi:2d}] | {s_active:6d} | {a_active:6d} | {s_sample:50s} | {a_sample}")


def save_result(result: dict, frame_index: np.ndarray, output_dir: str, episode_name: str):
    """Save conversion results as a parquet file, including frame_index and unified vectors."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, episode_name)

    table = pa.table({
        "frame_index": frame_index,
        "state_unified": [row.tolist() for row in result["state.unified"]],
        "action_unified": [row.tolist() for row in result["action.unified"]],
        "mask_state": [row.tolist() for row in result["mask.state"]],
        "mask_action": [row.tolist() for row in result["mask.action"]],
    })
    pq.write_table(table, out_path)
    return out_path


def save_unified_meta(output_dir: str):
    """Save unified vector metadata description to output_dir/unified_meta.json."""
    meta = {
        "description": "Unified state/action representation metadata",
        "unified_dim": UNIFIED_STATE_ACTION_DIM,
        "target_types": {
            "joint_state": TARGET_JOINT_STATE_TYPE,
            "eef_state": TARGET_EEF_STATE_TYPE,
            "joint_action": TARGET_JOINT_ACTION_TYPE,
            "eef_action": TARGET_EEF_ACTION_TYPE,
        },
        "slot_layout": {
            name: {
                "start": lo,
                "end": hi,
                "dim": hi - lo,
            }
            for name, (lo, hi) in UNIFIED_STATE_ACTION_INDICES.items()
        },
        "columns": {
            "frame_index": "int64, original frame index from the dataset",
            "state_unified": f"list<float32>[{UNIFIED_STATE_ACTION_DIM}], unified state vector",
            "action_unified": f"list<float32>[{UNIFIED_STATE_ACTION_DIM}], unified action vector",
            "mask_state": f"list<bool>[{UNIFIED_STATE_ACTION_DIM}], True where state has valid data",
            "mask_action": f"list<bool>[{UNIFIED_STATE_ACTION_DIM}], True where action has valid data",
        },
    }
    os.makedirs(output_dir, exist_ok=True)
    meta_path = os.path.join(output_dir, "unified_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)
    return meta_path


# ---- Parallel worker --------------------------------------------------------
# Module-level global variables, automatically inherited by child processes in fork mode
_g_transform = None
_g_modality_cfg = None
_g_output_dir = None


def _convert_one_episode(ep_path: str) -> str | None:
    """Worker: read -> convert -> save a single episode, return the saved path.
    Skips corrupted files (e.g., 0-byte parquet) and returns None.
    """
    ep_name = os.path.basename(ep_path)
    # Skip 0-byte files
    try:
        fsize = os.path.getsize(ep_path)
    except OSError:
        fsize = -1
    if fsize <= 0:
        print(f"[WARN] Skipping corrupted file ({fsize} bytes): {ep_path}")
        return None
    try:
        data, frame_index = extract_episode_data(ep_path, _g_modality_cfg)
        result = _g_transform.apply(data)
        saved_path = save_result(result, frame_index, _g_output_dir, ep_name)
        return saved_path
    except Exception as e:
        print(f"[WARN] Skipping failed episode {ep_path}: {e}")
        return None


# config: target type and dataset path
TARGET_JOINT_STATE_TYPE = "abs_joint"
TARGET_EEF_STATE_TYPE = "abs_quat"
TARGET_JOINT_ACTION_TYPE = "abs_joint"
TARGET_EEF_ACTION_TYPE = "abs_quat"


"""
Test:
python convert_unified.py "$VLA_DATA_ROOT/BC_Z" --episodes 1 --dry-run \
    --output-dir ./output
"""
def main():
    parser = argparse.ArgumentParser(description="Convert dataset to unified state/action representation")
    parser.add_argument("dataset_path", help="Path to the LeRobot v2.1 dataset root")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes to process (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Only print results, do not save files")
    parser.add_argument("--output-dir", default=None, help="Output directory for unified parquet files")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip episodes whose parquet already exists in output-dir")
    args = parser.parse_args()

    dataset_path = args.dataset_path.rstrip("/")
    modality_cfg = load_modality(dataset_path)
    info = load_info(dataset_path)
    total_episodes = info["total_episodes"]
    episode_files = list_episodes(dataset_path)
    n_episodes = min(args.episodes, total_episodes) if args.episodes is not None else total_episodes

    print(f"Dataset: {dataset_path}")
    print(f"Total episodes (from info.json): {total_episodes}")
    print(f"Episodes to process: {n_episodes}")
    print(f"Mode: {'dry-run (only print, no file output)' if args.dry_run else 'normal (save .parquet files)'}")
    print(f"Target types: joint_state={TARGET_JOINT_STATE_TYPE}, eef_state={TARGET_EEF_STATE_TYPE}, "
          f"joint_action={TARGET_JOINT_ACTION_TYPE}, eef_action={TARGET_EEF_ACTION_TYPE}")
    
    #build_apply_to : list all "state key" and "action key" in modality_cfg
    apply_to = build_apply_to(modality_cfg)
    print(f"Apply to keys: {apply_to}") # print orginial action and state keys

    
    modality_path = os.path.join(dataset_path, "meta", "modality.json")
    transform = UnifiedStateActionTransform(
        apply_to=apply_to,
        modality_path=modality_path,
        target_joint_state_type=TARGET_JOINT_STATE_TYPE,
        target_eef_state_type=TARGET_EEF_STATE_TYPE,
        target_joint_action_type=TARGET_JOINT_ACTION_TYPE,
        target_eef_action_type=TARGET_EEF_ACTION_TYPE,
    )

    if args.output_dir is None and not args.dry_run:
        args.output_dir = os.path.join(dataset_path, "unified_output")
        print(f"No output dir specified, using: {args.output_dir}")

    if not args.dry_run:
        meta_path = save_unified_meta(args.output_dir)
        print(f"Saved unified_meta.json to: {meta_path}")

    # Filter out already-existing episodes
    todo_paths = []
    skipped = 0
    for i in range(n_episodes):
        ep_path = episode_files[i]
        ep_name = os.path.basename(ep_path)
        if args.skip_existing and args.output_dir:
            out_path = os.path.join(args.output_dir, ep_name)
            if os.path.isfile(out_path):
                skipped += 1
                continue
        todo_paths.append(ep_path)

    print(f"To process: {len(todo_paths)}, skipped (already exist): {skipped}")

    if args.dry_run:
        # Dry-run mode: sequential, print one by one
        for ep_path in todo_paths:
            data, frame_index = extract_episode_data(ep_path, modality_cfg)
            result = transform.apply(data)
            print_result(result, ep_path)
        print(f"\nDone (dry-run). Processed {len(todo_paths)} episodes, skipped {skipped}.")
    else:
        # Normal mode: multiprocessing parallel
        global _g_transform, _g_modality_cfg, _g_output_dir
        _g_transform = transform
        _g_modality_cfg = modality_cfg
        _g_output_dir = args.output_dir

        try:
            from config import NUM_WORKERS
        except (ImportError, AttributeError):
            NUM_WORKERS = 48

        num_workers = min(NUM_WORKERS, len(todo_paths)) if todo_paths else 1
        print(f"Using multiprocessing.Pool with {num_workers} workers ...")

        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.imap_unordered(_convert_one_episode, todo_paths, chunksize=16)
            done = 0
            failed = 0
            for saved_path in results:
                done += 1
                if saved_path is None:
                    failed += 1
                if done % 500 == 0 or done == len(todo_paths):
                    print(f"  Progress: {done}/{len(todo_paths)} episodes saved"
                          + (f" ({failed} failed)" if failed else ""))

        print(f"\nDone. Processed {len(todo_paths)} episodes, skipped {skipped} (already exist)"
              + (f", {failed} failed (corrupted/unreadable)" if failed else "") + ".")


if __name__ == "__main__":
    main()
