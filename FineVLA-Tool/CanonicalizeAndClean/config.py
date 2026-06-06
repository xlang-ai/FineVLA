# -*- coding: utf-8 -*-
"""
Dataset frame-count filtering configuration.
Dataset name -> minimum frame threshold; episodes below this value are considered "too few frames" and logged.
None means no frame-count filtering is applied; only episodes with "empty task" are recorded.

  python filter_by_state_action_frame.py \
      $VLA_DATA_ROOT/RH20T-RoboInter \
      --episodes 10 --force-reconvert
"""
# 1. Filter out episodes with too few frames
# Dataset name (matching directory name under Lerobot_v21 or top-level directory name) -> minimum allowed frames
# If a dataset has multiple sub-directory levels, the script first looks up by full relative path, then by top-level name
import os


DATASET_MIN_FRAME = {
    "BC_Z": 41,
    "Bridge": 10,
    "Galaxea-Open-World-Dataset": 50,
    "RT-1": 12,
    "RDT-yhq": 30,
    "droid_1.0.1": 40,
    "droid_RoboInter": 40,
    "RH20T-RoboInter": 20,
    "RH20T-fjy/rh20t_cfg1": 20,
    "RH20T-fjy/rh20t_cfg2": 20,
    "RH20T-fjy/rh20t_cfg3": 20,
    "RH20T-fjy/rh20t_cfg4": 20,
    "RH20T-fjy/rh20t_cfg5": 20,
    "RH20T-fjy/rh20t_cfg6": 20,
    "RH20T-fjy/rh20t_cfg7": 20,
    "RoboCOIN": 60,
    "RoboCOIN_add0130": 60,
    "RoboCOIN_add1201": 60,
    "RoboMindV1.0": 30,
    "RoboMindV2.0": 30,
    "agibotworld_hyy": None,
    "xvla-soft-fold_franka_v3_franka": None,
}




# 2. Filter out episodes with excessively large state-action differences
# ============================================================
# State-Action Difference Filtering Configuration
# ============================================================
# Dataset name -> {state_slot, action_slot} or None (skip)
# Slot names must match keys in slot_layout from unified_meta.json.
# left_gripper / right_gripper are not allowed.
# state_slot / action_slot accept a string (single slot) or a list (multiple slots concatenated, for dual-arm).
# List example: {"state_slot": ["left_eef", "right_eef"], "action_slot": ["left_eef", "right_eef"]}
STATE_ACTION_COMPARE_SLOTS = {
    "BC_Z": {"state_slot": "right_eef", "action_slot": "right_eef"},
    "Bridge": {"state_slot": "right_eef", "action_slot": "right_eef"},
    "RT-1": {"state_slot": "right_eef", "action_slot": "right_eef"},
    "droid_1.0.1": {"state_slot": "right_eef", "action_slot": "right_eef"},
    "droid_RoboInter": {"state_slot": "right_eef", "action_slot": "right_eef"},
    "Galaxea-Open-World-Dataset": {"state_slot": "right_joint", "action_slot": "right_joint"},
    "RDT-yhq": {"state_slot": "right_joint", "action_slot": "right_joint"},
    # RH20T-fjy configs have different dimensions; configured by sub-path
    "RH20T-RoboInter": {"state_slot": "right_eef", "action_slot": "right_eef"},
    "RH20T-fjy/rh20t_cfg1": {"state_slot": "right_joint", "action_slot": "right_joint"},
    "RH20T-fjy/rh20t_cfg2": {"state_slot": "right_joint", "action_slot": "right_joint"},
    "RH20T-fjy/rh20t_cfg3": {"state_slot": "right_joint", "action_slot": "right_joint"},
    "RH20T-fjy/rh20t_cfg4": {"state_slot": "right_joint", "action_slot": "right_joint"},
    "RH20T-fjy/rh20t_cfg5": {"state_slot": "right_joint", "action_slot": "right_joint"},
    "RH20T-fjy/rh20t_cfg6": {"state_slot": "right_joint", "action_slot": "right_joint"},
    "RH20T-fjy/rh20t_cfg7": {"state_slot": "right_joint", "action_slot": "right_joint"},

    # RoboCOIN dual-arm: compare both left and right eef simultaneously
    "RoboCOIN": {"state_slot": ["left_eef", "right_eef"], "action_slot": ["left_eef", "right_eef"]},
    "RoboCOIN_add0130": {"state_slot": ["left_eef", "right_eef"], "action_slot": ["left_eef", "right_eef"]},
    "RoboCOIN_add1201": {"state_slot": ["left_eef", "right_eef"], "action_slot": ["left_eef", "right_eef"]},

    # RoboMindV1.0 and RoboMindV2.0: use left_joint+right_joint for L2 comparison (all robot types have joint)
    # Single-arm robots (franka_1rgb/3rgb, ur_1rgb, etc.) have left_joint zeroed in unified, which does not affect comparison, so all can be included
    "RoboMindV1.0": {"state_slot": ["left_joint", "right_joint"], "action_slot": ["left_joint", "right_joint"]},
    "RoboMindV2.0": {"state_slot": ["left_joint", "right_joint"], "action_slot": ["left_joint", "right_joint"]},
    "agibotworld_hyy": None,
    "xvla-soft-fold_franka_v3_franka": {"state_slot": "right_joint", "action_slot": "right_joint"},
}

# 3. Per-dataset L2 distance threshold (range-normalized per-frame L2)
# ============================================================
# Episodes with L2 score exceeding the dataset threshold are flagged as anomalous
# Set to None to skip threshold filtering and only output the report
# Recommended: first run without a threshold, then determine based on p90/p95 statistics in the report
DATASET_L2_THRESHOLD = {
    "BC_Z": 1.0,
    "Bridge": 0.5,
    "Galaxea-Open-World-Dataset": 0.5,
    "RDT-yhq": 0.3,
    "droid_1.0.1": 0.5,
    "droid_RoboInter": 0.5,
    "RT-1": 0.5,
    "RH20T-RoboInter": 0.2,
    "RH20T-fjy/rh20t_cfg1": 0.2,
    "RH20T-fjy/rh20t_cfg2": 0.2,
    "RH20T-fjy/rh20t_cfg3": 0.2,
    "RH20T-fjy/rh20t_cfg4": 0.2,
    "RH20T-fjy/rh20t_cfg5": 0.2,
    "RH20T-fjy/rh20t_cfg6": 0.2,
    "RH20T-fjy/rh20t_cfg7": 0.2,
    "RoboCOIN": 0.2,
    "RoboCOIN_add0130": 0.2,
    "RoboCOIN_add1201": 0.2,
    "RoboMindV1.0": 1,
    "RoboMindV2.0": 0.5,
    "agibotworld_hyy": None,
    "xvla-soft-fold_franka_v3_franka": None,
}

# Global default threshold: used when a dataset is not configured in DATASET_L2_THRESHOLD
# Set to None to skip threshold filtering for unconfigured datasets
DEFAULT_L2_THRESHOLD = 0.2
PLOT = False
NUM_WORKERS = 96

# Data root directory (absolute path to Lerobot_v21)
DATA_ROOT = os.environ.get("VLA_DATA_ROOT", "/path/to/your/Lerobot_v21")

# Directory names to skip when searching for meta (do not descend into these directories)
SKIP_DIRS = {"video", "videos", "data", "unified_output"}

# Episode metadata filenames (tried in priority order)
EPISODE_FILENAMES = ["episodes.jsonl", "episode.jsonl"]


"""

python convert_unified.py "$VLA_DATA_ROOT/BC_Z" --episodes 1 --dry-run \
    --output-dir "./output"

python filter_by_state_action_frame.py $VLA_DATA_ROOT/droid_RoboInter --force-reconvert --episode 100
"""