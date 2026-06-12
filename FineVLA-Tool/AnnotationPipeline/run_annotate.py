#!/usr/bin/env python3
"""
Entry point: reads JSONL directly and runs the annotation pipeline.

Usage:
    python run_annotate.py \\
        --input /path/to/input.jsonl \\
        --output /path/to/results.jsonl \\
        --mode annotate --num_workers 8

    # Process only specific datasets
    python run_annotate.py \\
        --input /path/to/input.jsonl \\
        --output /path/to/results.jsonl \\
        --dataset droid_1.0.1,rdt

    # Dry-run: check loading & view selection, no API calls
    python run_annotate.py \\
        --input /path/to/input.jsonl \\
        --dry-run
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from config import (
    VIDEO_BASE_DIR,
    DEFAULT_MODEL,
    DEFAULT_BASE_URL,
    DEFAULT_MAX_FRAMES,
    DEFAULT_BATCH_SIZE,
    logger,
)
from dataset_configs import get_config as get_dataset_config, has_dedicated_config
from runner import process_trajectories_parallel, load_completed_ids


# =========================================================================
# JSONL loading + view selection
# =========================================================================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def has_raw_steps(sample: Dict[str, Any]) -> bool:
    """Return True when a sample carries pre-annotated step boundaries."""
    return bool(sample.get("has_raw_steps") or sample.get("steps_raw"))


def prepare_samples(
    samples: List[Dict[str, Any]],
    video_base_dir: str = "",
) -> List[Dict[str, Any]]:
    """
    For each raw JSONL sample, compute per-stage view selection and absolute
    video paths via the dataset's config (get_config always returns a config;
    DefaultConfig is used for datasets without a dedicated subclass).

    Writes into each sample dict:
      - stage_views  : {stage_name: {"view": ..., "video_path": ...}, ...}
      - video_path   : first stage's path (backward compat with TrajectoryItem)
      - selected_view: first stage's view name (backward compat)
      - view_type    : ViewType string
    """
    prepared = []
    for s in samples:
        ds_config = get_dataset_config(s.get("dataset", ""))
        stage_views = ds_config.resolve_stage_views(s, video_base_dir=video_base_dir)

        s["stage_views"] = stage_views

        first_stage = next(iter(stage_views), None)
        s["video_path"] = stage_views[first_stage]["video_path"] if first_stage else ""
        s["selected_view"] = stage_views[first_stage]["view"] if first_stage else ""
        s["view_type"] = ds_config.classify_view(s)
        prepared.append(s)
    return prepared


# =========================================================================
# CLI
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Annotate robot manipulation videos from JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--input", required=True, help="Input JSONL file")
    p.add_argument("--output", default=None, help="Output JSON (required unless --dry-run)")
    p.add_argument("--video-base-dir", default=VIDEO_BASE_DIR, help="Video root directory")

    # Filtering
    p.add_argument(
        "--mode", default="all", choices=["all", "annotate", "review"],
        help="'annotate' = no subtask, 'review' = has subtask, 'all' = everything",
    )
    p.add_argument(
        "--dataset", default=None,
        help="Comma-separated dataset names to include (e.g. droid_1.0.1,rdt)",
    )
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)

    # Pipeline
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--max_frames", type=int, default=DEFAULT_MAX_FRAMES)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)

    # Per-stage overrides (generic, repeatable)
    p.add_argument(
        "--stage-fps", action="append", default=[], metavar="STAGE=FPS",
        help="Override target_fps for a stage, e.g. analysis=4.0 (repeatable)",
    )
    p.add_argument(
        "--stage-ratio", action="append", default=[], metavar="STAGE=RATIO",
        help="Override sample_ratio for a stage, e.g. refinement=0.5 (repeatable)",
    )

    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--base_url", default=DEFAULT_BASE_URL)
    p.add_argument("--quiet", action="store_true")

    p.add_argument("--dry-run", action="store_true", help="Only load & preview, no API calls")

    return p.parse_args()


def _print_config_summary(samples: List[Dict[str, Any]], stage_params: Dict[str, Dict] = None):
    """Print a config-centric summary grouped by dataset."""
    from collections import defaultdict
    stage_params = stage_params or {}

    ds_groups: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        ds_groups[s["dataset"]].append(s)

    print(f"\n  {len(samples)} samples across {len(ds_groups)} datasets")

    all_stage_names: set = set()
    all_video_paths: List[str] = []

    for ds, grp in sorted(ds_groups.items(), key=lambda kv: -len(kv[1])):
        cfg = get_dataset_config(ds)
        stage_parts = []
        for sd in cfg.stages:
            fps = stage_params.get(sd.name, {}).get("target_fps", sd.default_fps)
            dep = f", after={sd.depends_on}" if sd.depends_on else ""
            stage_parts.append(f"{sd.name}(fps={fps}{dep})")
            all_stage_names.add(sd.name)
        print(f"\n  {ds} ({len(grp)} samples): {' -> '.join(stage_parts)}")

        view_parts = []
        for sd in cfg.stages:
            vc = Counter(s.get("stage_views", {}).get(sd.name, {}).get("view", "") or "skip" for s in grp)
            view_parts.append(f"{sd.name}:{','.join(f'{v}({c})' for v, c in vc.most_common())}")
        print(f"    views: {' | '.join(view_parts)}")

        for s in grp:
            for sv in s.get("stage_views", {}).values():
                if sv.get("video_path"):
                    all_video_paths.append(sv["video_path"])

    unique = list(dict.fromkeys(all_video_paths))[:30]
    found = sum(1 for p in unique if os.path.isfile(p))
    print(f"\n  Video spot-check: {found}/{len(unique)} exist\n")

    return all_stage_names


def main():
    args = parse_args()

    # ── Load JSONL ──
    logger.info(f"Loading JSONL: {args.input}")
    samples = load_jsonl(args.input)
    logger.info(f"Loaded {len(samples)} raw samples")

    # ── Filter by mode ──
    if args.mode == "annotate":
        samples = [s for s in samples if not has_raw_steps(s)]
    elif args.mode == "review":
        samples = [s for s in samples if has_raw_steps(s)]
    logger.info(f"After mode={args.mode} filter: {len(samples)} samples")

    # ── Filter by dataset ──
    if args.dataset:
        ds_set = {d.strip() for d in args.dataset.split(",")}
        samples = [s for s in samples if s.get("dataset") in ds_set]
        logger.info(f"Filtered to datasets {ds_set}: {len(samples)} samples")

    if not samples:
        logger.error("No samples after filtering. Check --mode / --dataset.")
        sys.exit(1)

    # ── Compute view & video_path ──
    samples = prepare_samples(samples, video_base_dir=args.video_base_dir)

    # ── Slice ──
    end = args.end if args.end is not None else len(samples)
    samples = samples[args.start:end]
    logger.info(f"Processing [{args.start}:{end}], total: {len(samples)}")

    if not samples:
        logger.error("No samples after slicing.")
        sys.exit(1)

    # ── Parse stage overrides early (needed by summary + pipeline) ──
    stage_params: Dict[str, Dict] = {}

    def _set_param(stage: str, key: str, value):
        stage_params.setdefault(stage, {})[key] = value

    for item in args.stage_fps:
        if "=" not in item:
            logger.error(f"Invalid --stage-fps format: '{item}', expected STAGE=FPS")
            sys.exit(1)
        stage, val = item.split("=", 1)
        _set_param(stage.strip(), "target_fps", float(val))

    for item in args.stage_ratio:
        if "=" not in item:
            logger.error(f"Invalid --stage-ratio format: '{item}', expected STAGE=RATIO")
            sys.exit(1)
        stage, val = item.split("=", 1)
        ratio_val = float(val)
        if not (0 < ratio_val <= 1):
            logger.error(f"--stage-ratio {stage}={ratio_val} must be in (0, 1]")
            sys.exit(1)
        _set_param(stage.strip(), "sample_ratio", ratio_val)

    # ── Config-centric summary (with effective fps/ratio) ──
    all_stage_names = _print_config_summary(samples, stage_params=stage_params)

    if stage_params:
        logger.info(f"Stage params: {stage_params}")
        unknown = set(stage_params.keys()) - all_stage_names
        if unknown:
            logger.warning(
                f"--stage-fps/--stage-ratio reference unknown stages: {unknown}  "
                f"(known stages: {sorted(all_stage_names)})"
            )

    # ── Dry-run stops here ──
    if args.dry_run:
        print("Dry-run complete. No API calls made.")
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)
            print(f"Prepared data saved to {args.output}")
        sys.exit(0)

    # ── Run pipeline ──
    if not args.output:
        logger.error("--output is required when not using --dry-run")
        sys.exit(1)

    # Normalise output path to .jsonl (append-friendly format)
    output_path = args.output
    if output_path.endswith(".json"):
        output_path = output_path[:-5] + ".jsonl"
        logger.info(f"Output path adjusted to JSONL: {output_path}")

    # ── Resume: skip already-completed samples ──
    completed_ids = load_completed_ids(output_path)
    if completed_ids:
        before = len(samples)
        samples = [s for s in samples if s.get("sample_id", "") not in completed_ids]
        logger.info(f"Resuming: {len(completed_ids)} already completed, "
                     f"skipped {before - len(samples)}, {len(samples)} remaining")

    if not samples:
        logger.info("All samples already completed. Nothing to do.")
        sys.exit(0)

    summary = process_trajectories_parallel(
        trajectories=samples,
        output_path=output_path,
        model=args.model,
        base_url=args.base_url,
        num_workers=args.num_workers,
        max_frames=args.max_frames,
        batch_size=args.batch_size,
        stage_params=stage_params,
        verbose=not args.quiet,
    )

    sys.exit(0 if summary["failures"] == 0 else 1)


if __name__ == "__main__":
    main()
