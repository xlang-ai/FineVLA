# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

VLA_Annotation is a monorepo for VLA (Vision-Language-Action) robotics dataset annotation tooling. The **Filter** module (current working directory) handles quality filtering and unified representation conversion for LeRobot v2.1 datasets stored under a shared data root.

## Project Structure

The repo has several independent top-level modules:

- **Filter/** — Episode quality filtering and state/action unification (current focus)
- **Format_Transfer/** — Dataset format conversion (LeRobot version upgrades via `any4lerobot`)
- **Joint2Action/** — Joint-to-action conversion, clustering, and visualization (FastAPI + HTML frontend)
- **Gold_Trajectory/** — Gold trajectory review UI (FastAPI backend + vanilla JS frontend)
- **Visualization/** — Dataset visualization tools
- **Calculation/** — Dataset statistics computation

## Filter Module Architecture

The Filter pipeline operates on LeRobot v2.1 datasets located at `DATA_ROOT` (configured in `config.py`).

### Pipeline Flow

```
Raw dataset parquets + modality.json
        │
        ▼
  convert_unified.py          ← Step 1: convert to 80-dim unified representation
  (calls UnifyJointAction)       outputs: <dataset>/unified_output/*.parquet + unified_meta.json
        │
        ▼
  filter_by_state_action_frame.py  ← Step 2: combined filtering (frames + task + L2)
  (calls cal_distance.py)            outputs: {dataset_key}_filter_report.json
```

### Core Scripts

- **`config.py`** — Central configuration: `DATASET_MIN_FRAME`, `STATE_ACTION_COMPARE_SLOTS`, `DATASET_L2_THRESHOLD` (per-dataset), `DEFAULT_L2_THRESHOLD`, `NUM_WORKERS`, `DATA_ROOT`, `SKIP_DIRS`
- **`filter_by_state_action_frame.py`** — **Main entry point.** Combined filter: (1) frame count, (2) empty task (handles `None`/`[]`/`[""]`), (3) L2 state-action divergence. L2 threshold priority: `--threshold` CLI > `DATASET_L2_THRESHOLD[key]` > `DEFAULT_L2_THRESHOLD`
- **`convert_unified.py`** — Converts raw parquets into unified 80-dim representation. Supports `--dry-run`, `--skip-existing`, multiprocessing
- **`cal_distance.py`** — Range-normalized per-frame L2 distance: `compute_vla_l2_score` (score) and `compute_episode_similarity` (score + arrays for plotting)

### Key Utility

- **`utils/UnifyJointAction.py`** — `UnifiedStateActionTransform`: maps heterogeneous state/action into 80-dim vector with mask. Supported EEF rotation types: `abs_quat` (xyzw), `abs_wxyz`, `abs_euler` (xyz), `abs_rotvec`. Coordinate frames: `abs`/`delta`/`rel`. Joints: `abs_joint`

### Unified 80-dim Vector Layout

Defined in `UNIFIED_STATE_ACTION_INDICES` (UnifyJointAction.py):
- `[0:7]` left_joint, `[7:16]` left_eef, `[16:17]` left_gripper, `[17:29]` left_hand
- `[29:36]` right_joint, `[36:45]` right_eef, `[45:46]` right_gripper, `[46:58]` right_hand
- `[58:80]` reserved

## Common Commands

```bash
# Main combined filter (frame + task + L2) for a single dataset
python filter_by_state_action_frame.py /path/to/Lerobot_v21/BC_Z

# With options: limit episodes, custom threshold, force re-convert, enable plots
python filter_by_state_action_frame.py /path/to/Lerobot_v21/BC_Z \
    --episodes 100 --threshold 1.5 --force-reconvert --plot

# Convert dataset to unified representation (standalone, dry-run)
python convert_unified.py /path/to/Lerobot_v21/BC_Z --episodes 2 --dry-run

# Convert and save unified parquet files
python convert_unified.py /path/to/Lerobot_v21/BC_Z --episodes 5 --output-dir /tmp/unified_output
```

## Dataset Root & Conventions

- Dataset root: `/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21`
- Each dataset has `meta/` containing `info.json`, `modality.json`, and `episodes.jsonl` (or `episode.jsonl`)
- When traversing datasets, always skip `video`, `videos`, and `data` directories
- The `modality.json` describes original state/action field mappings with `original_key`, `type`, `dim`, `start`, `end`
- modality.json key names **must** match `UNIFIED_STATE_ACTION_INDICES` (e.g. `left_joint`, `right_eef`, not custom names)
- Unified outputs go to `<dataset>/unified_output/` (note: xvla shared filesystem has write issues, use `--output-dir /tmp/...` as workaround)

## Dataset Status (as of 2026-03-07)

### Flat datasets (top-level meta, ready for filtering)

| Dataset | Episodes | modality.json | State type | Action type | L2 slot | L2 threshold | Notes |
|---------|----------|--------------|------------|-------------|----------|---------------|-------|
| BC_Z | 39,350 | ✅ | abs_euler(right_eef) + gripper | delta_euler(right_eef) + gripper | right_eef | 1.0 | min_frame=41 |
| Bridge | 53,192 | ✅ | abs_euler(right_eef) + gripper | delta_euler(right_eef) + gripper | right_eef | 0.5 | min_frame=10; **27% task=[""]**; L2 mean=0.067 |
| RT-1 | 87,212 | ✅ (rewritten) | abs_quat_xyzw(right_eef) + gripper | delta_euler(right_eef) + gripper | right_eef | 0.5 | L2 mean=0.139 (control latency); 8 empty tasks |
| droid_1.0.1 | 95,600 | ✅ (created) | abs_euler(eef) + abs_joint(7) + gripper | abs_euler(eef) + abs_joint(7) + gripper | right_eef | 0.5 | **22% empty tasks**; L2 mean=0.034 |
| egodex_train_robot_yhq | 314,839 | ✅ (created) | abs_euler(L/R eef, **camera frame**) + gripper | abs_euler(L/R eef, **world frame**) + gripper | **None** (skip) | — | State/action in different coord frames, L2 not comparable |
| xvla-soft-fold_franka_v3_franka | 1,542 | ✅ (rewritten) | abs_joint(L/R joint+hand) | abs_joint(L/R joint) + gripper | right_joint | — | L2 mean=0.055; shared FS write error, use /tmp |

### Sub-directory datasets (no top-level meta, need per-sub processing)

| Dataset | Sub-datasets | Have modality.json | Notes |
|---------|-------------|-------------------|-------|
| Galaxea-Open-World-Dataset | 227 | 1 out of 227 | Only Adjust_The_Air_Conditioner_Temperature has modality |
| RDT-yhq | 296 | 0 | All missing modality |
| RH20T-fjy | 7 | 0 | Sub-datasets: rh20t_cfg1 etc. |
| RoboCOIN | 297 | 0 | AIRBOT_MMK2 based robots |
| RoboCOIN_add0130 | 29 | 0 | |
| RoboCOIN_add1201 | 112 | 0 | |
| RoboCOIN_annotations_backup | 0 | 0 | Empty/backup only |
| RoboMindV1.0 | 0 | 0 | Possibly deeper nesting |
| RoboMindV2.0 | 0 | 0 | Possibly deeper nesting |
| agibotworld_hyy | 214 | 0 | Sub-datasets: task_327 etc. |

**Total: ~1182 sub-datasets, almost all missing modality.json.** Need batch modality template generation per robot type.

## Key Configuration Patterns

- To add a new dataset: add entries to `DATASET_MIN_FRAME`, `STATE_ACTION_COMPARE_SLOTS`, and `DATASET_L2_THRESHOLD` in `config.py`
- `STATE_ACTION_COMPARE_SLOTS` slot names must match `unified_meta.json` keys; gripper slots are forbidden
- L2 threshold tuning: run with threshold=None first, then set based on p90/p95 from statistics output
- `--threshold` CLI arg overrides per-dataset config for one-off testing

## Known Issues

- `_tasks_empty()` now handles `[""]` (empty-string tasks) — this was a bug fixed during Bridge testing
- xvla shared filesystem (`cpfs`) has write errors for large parquets — workaround: `--output-dir /tmp/...`
- `filter_by_state_action_frame.py` only processes one dataset path at a time; no batch mode for sub-directory datasets yet
- egodex state (camera frame) vs action (world frame) mismatch makes L2 comparison meaningless

## Language and Style

- Code comments and UI text are primarily in Chinese
- Python with numpy, pyarrow, scipy, pydantic, matplotlib dependencies
