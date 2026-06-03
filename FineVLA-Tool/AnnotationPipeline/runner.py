"""
Parallel pipeline runner.

process_single_trajectory  — runs all stages for one sample (stage-agnostic loop)
process_trajectories_parallel — orchestrates multiprocessing over many samples

Output is JSONL (one result per line, appended).  On restart, already-completed
sample_ids are detected and skipped automatically (resumable).
"""

import json
import multiprocessing as mp
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set

from tqdm import tqdm

from config import (
    DEFAULT_MODEL,
    DEFAULT_MAX_FRAMES,
    DEFAULT_BATCH_SIZE,
    logger,
)
from data_types import (
    TrajectoryItem,
    StageResult,
    ProcessingResult,
)
from video_utils import _ensure_imports
from api_client import create_api_client
from stages import STAGE_REGISTRY
from dataset_configs import get_config as get_dataset_config


# Per-process API client (created once per worker, reused across samples)
_process_client = None


def _get_client(api_key: str = None, base_url: str = None):
    """Return the per-process API client, creating it on first call."""
    global _process_client
    if _process_client is None:
        _ensure_imports()
        _process_client = create_api_client(api_key=api_key, base_url=base_url)
    return _process_client


def _worker_init(api_key: str = None, base_url: str = None):
    """Initialize worker process: lazy imports + pre-create API client."""
    _ensure_imports()
    _get_client(api_key=api_key, base_url=base_url)


# =============================================================================
# Single-trajectory processing (stage-agnostic loop)
# =============================================================================

def process_single_trajectory(
    trajectory_dict: Dict[str, Any],
    idx: int = 0,
    model: str = DEFAULT_MODEL,
    api_key: str = None,
    base_url: str = None,
    max_frames: int = DEFAULT_MAX_FRAMES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    stage_params: Dict[str, Dict] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Process a single trajectory through all pipeline stages.

    Uses a per-process API client (created once by _worker_init or on first
    call via _get_client) to avoid creating a new connection per sample.
    """
    start_time = time.time()
    stage_params = stage_params or {}

    trajectory = TrajectoryItem.from_dict(trajectory_dict)

    if verbose:
        logger.info(f"Processing: {trajectory.task} - {trajectory.initial_instruction[:50]}...")

    config = get_dataset_config(trajectory_dict.get("dataset", ""))
    stage_defs = config.get_active_stages(trajectory_dict)

    def _make_result(status: str, error_detail: str = "") -> Dict[str, Any]:
        """Build a full error result for early-exit / exception cases."""
        sr = {
            sd.name: StageResult(stage_name=sd.name, error=error_detail or status)
            for sd in stage_defs
        }
        out = ProcessingResult(
            trajectory=trajectory, stage_results=sr,
            processing_time=time.time() - start_time, model=model,
        ).to_output_dict()
        out["_task_status"] = {"status": status, "error_detail": error_detail, "idx": idx}
        return out

    if not trajectory.video_path:
        return _make_result("MISSING_VIDEO_PATH", "Missing videoPath field")

    if not trajectory.initial_instruction:
        return _make_result("MISSING_INSTRUCTION", "Missing initialInstruction field")

    try:
        client = _get_client(api_key=api_key, base_url=base_url)
        stage_views = trajectory_dict.get("stage_views", {})
        stage_results: Dict[str, StageResult] = {}

        # Cache opened video readers across stages to avoid redundant I/O
        video_reader_cache: Dict[str, Any] = {}

        try:
            for sd in stage_defs:
                if sd.depends_on:
                    dep = stage_results.get(sd.depends_on)
                    if not dep or not dep.success:
                        stage_results[sd.name] = StageResult(
                            stage_name=sd.name,
                            error=f"Skipped: dependency '{sd.depends_on}' failed",
                        )
                        continue

                fn = STAGE_REGISTRY.get(sd.fn_name)
                if fn is None:
                    stage_results[sd.name] = StageResult(
                        stage_name=sd.name,
                        error=f"Unknown stage function '{sd.fn_name}'",
                    )
                    continue

                video_path = (
                    stage_views.get(sd.name, {}).get("video_path")
                    or trajectory.video_path
                )

                params = stage_params.get(sd.name, {})
                fps = params.get("target_fps", sd.default_fps)
                ratio = params.get("sample_ratio", sd.default_sample_ratio)

                prompt_kwargs: Dict[str, str] = {}
                override = config.get_prompt_override(sd.name, trajectory_dict)
                if override:
                    prompt_kwargs["system_prompt"] = override[0]
                    prompt_kwargs["user_prompt_template"] = override[1]

                if verbose:
                    view_name = stage_views.get(sd.name, {}).get("view", "")
                    logger.info(f"  Stage '{sd.name}': view={view_name or 'default'}, fps={fps}")

                result = fn(
                    client,
                    video_path,
                    trajectory.initial_instruction,
                    stage_results,
                    model=model,
                    target_fps=fps,
                    max_frames=max_frames,
                    batch_size=batch_size,
                    sample_ratio=ratio,
                    trajectory_dict=trajectory_dict,
                    video_reader_cache=video_reader_cache,
                    **prompt_kwargs,
                )
                stage_results[sd.name] = result
        finally:
            for _vr_tuple in video_reader_cache.values():
                vr = _vr_tuple[0]
                if vr is not None:
                    del vr
            video_reader_cache.clear()

        processing_time = time.time() - start_time
        proc_result = ProcessingResult(
            trajectory=trajectory,
            stage_results=stage_results,
            processing_time=processing_time,
            model=model,
        )

        if verbose:
            logger.info(f"  {'SUCCESS' if proc_result.success else 'FAILED'} in {processing_time:.1f}s")

        out = proc_result.to_output_dict()
        if proc_result.success:
            out["_task_status"] = {"status": "success", "error_detail": "", "idx": idx}
        else:
            error_detail = next(
                (r.error for r in stage_results.values() if r.error), "Unknown error",
            )
            error_type = error_detail.split(":")[0] if ":" in error_detail else "UNKNOWN_ERROR"
            out["_task_status"] = {"status": error_type, "error_detail": error_detail, "idx": idx}
        return out

    except Exception as e:
        logger.error(f"Error processing trajectory: {e}")
        error_str = str(e)
        error_type = "EXCEPTION"
        if "FileNotFoundError" in error_str or "not found" in error_str.lower():
            error_type = "FILE_NOT_FOUND"
        elif "API" in error_str or "429" in error_str or "rate" in error_str.lower():
            error_type = "API_ERROR"
        return _make_result(error_type, error_str)


# =============================================================================
# JSONL I/O + Resume
# =============================================================================

def load_completed_ids(output_path: str) -> Set[str]:
    """Scan existing JSONL output and return sample_ids that already succeeded."""
    completed: Set[str] = set()
    if not os.path.exists(output_path):
        return completed
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                status = r.get("_task_status", {}).get("status", "")
                sid = r.get("sample_id", "")
                if status == "success" and sid:
                    completed.add(sid)
            except json.JSONDecodeError:
                continue
    return completed


def _append_result_jsonl(result: Dict, output_path: str):
    """Append a single result as one JSONL line."""
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


# =============================================================================
# Parallel orchestrator
# =============================================================================

def process_trajectories_parallel(
    trajectories: List[Dict[str, Any]],
    output_path: str,
    model: str = DEFAULT_MODEL,
    api_key: str = None,
    base_url: str = None,
    num_workers: int = None,
    max_frames: int = DEFAULT_MAX_FRAMES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    stage_params: Dict[str, Dict] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Process multiple trajectories in parallel using multiprocessing.

    Results are appended to ``output_path`` as JSONL (one line per sample).
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    num_workers = min(num_workers, len(trajectories))
    total = len(trajectories)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Processing {total} trajectories with {num_workers} workers")

    process_kwargs = {
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "max_frames": max_frames,
        "batch_size": batch_size,
        "stage_params": stage_params,
        "verbose": verbose,
    }

    task_statuses: List[Dict] = []
    success_count = 0
    start_time = time.time()

    def _collect(idx: int, result: Dict):
        nonlocal success_count
        _append_result_jsonl(result, output_path)
        status_info = result.get("_task_status", {"status": "UNKNOWN_ERROR", "error_detail": "", "idx": idx})
        task_statuses.append(status_info)
        if status_info["status"] == "success":
            success_count += 1

    if num_workers == 1:
        _worker_init(api_key=api_key, base_url=base_url)
        for i, traj in enumerate(tqdm(trajectories, desc="Processing", unit="traj")):
            _collect(i, process_single_trajectory(traj, idx=i, **process_kwargs))
    else:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_worker_init,
            initargs=(api_key, base_url),
        ) as executor:
            futures = {
                executor.submit(process_single_trajectory, traj, idx=i, **process_kwargs): i
                for i, traj in enumerate(trajectories)
            }
            for future in tqdm(as_completed(futures), total=total, desc="Processing", unit="traj"):
                idx = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "processingMetadata": {"success": False},
                        "_task_status": {"status": "EXECUTOR_ERROR", "error_detail": str(e), "idx": idx},
                    }
                _collect(idx, result)

    total_time = time.time() - start_time
    return print_statistics_summary(
        task_statuses=task_statuses,
        total_time=total_time,
        output_path=output_path,
        num_workers=num_workers,
    )


# =============================================================================
# Statistics
# =============================================================================

def print_statistics_summary(
    task_statuses: List[Dict],
    total_time: float,
    output_path: str,
    num_workers: int,
) -> Dict[str, Any]:
    """Print execution statistics summary."""
    total = len(task_statuses)
    status_counter = Counter(s["status"] for s in task_statuses)
    success_count = status_counter.get("success", 0)
    failure_count = total - success_count
    success_rate = (success_count / total * 100) if total > 0 else 0

    summary = {
        "total_trajectories": total,
        "successes": success_count,
        "failures": failure_count,
        "success_rate": round(success_rate, 2),
        "total_time_seconds": round(total_time, 2),
        "avg_time_per_trajectory": round(total_time / total, 2) if total > 0 else 0,
        "output_path": output_path,
        "num_workers": num_workers,
        "error_breakdown": dict(status_counter),
    }

    avg = summary["avg_time_per_trajectory"]
    print(f"\n--- Results: {success_count}/{total} succeeded ({success_rate:.1f}%), "
          f"{total_time:.1f}s total, {avg:.1f}s/task, {num_workers} workers ---")
    print(f"Output: {output_path}")
    if failure_count > 0:
        for stype, cnt in sorted(status_counter.items(), key=lambda x: -x[1]):
            if stype != "success":
                print(f"  {stype}: {cnt}")

    return summary
