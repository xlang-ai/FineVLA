# -*- coding: utf-8 -*-
"""
数据集帧数过滤配置。
数据集名称 -> 最小帧数阈值，低于此值的 episode 视为「帧数过少」并记录。
None 表示暂不按帧数过滤，仅统计并记录「task 为空」的 episode。

  python filter_by_state_action_frame.py \
      $VLA_DATA_ROOT/RH20T-RoboInter \
      --episodes 10 --force-reconvert
"""
# 1.筛选掉帧数过少的episode
# 数据集名称（与 Lerobot_v21 下目录名或「顶层目录名」对应）-> 最小允许帧数
# 若某数据集有多层子目录，脚本会先用完整相对路径查找，再用顶层名查找
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




# 2. 筛选掉State-Action 差值过大的episode
# ============================================================
# State-Action 差值筛选配置
# ============================================================
# 数据集名称 -> {state_slot, action_slot} 或 None（跳过）
# slot 名必须与 unified_meta.json 中 slot_layout 的键一致。
# 禁止使用 left_gripper / right_gripper。
# state_slot / action_slot 支持字符串（单 slot）或列表（多 slot 拼接，用于双臂）。
# 列表示例: {"state_slot": ["left_eef", "right_eef"], "action_slot": ["left_eef", "right_eef"]}
STATE_ACTION_COMPARE_SLOTS = {
    "BC_Z": {"state_slot": "right_eef", "action_slot": "right_eef"},
    "Bridge": {"state_slot": "right_eef", "action_slot": "right_eef"},
    "RT-1": {"state_slot": "right_eef", "action_slot": "right_eef"},
    "droid_1.0.1": {"state_slot": "right_eef", "action_slot": "right_eef"},
    "droid_RoboInter": {"state_slot": "right_eef", "action_slot": "right_eef"},
    "Galaxea-Open-World-Dataset": {"state_slot": "right_joint", "action_slot": "right_joint"},
    "RDT-yhq": {"state_slot": "right_joint", "action_slot": "right_joint"},
    # RH20T-fjy 各cfg维度不同，按子路径配置
    "RH20T-RoboInter": {"state_slot": "right_eef", "action_slot": "right_eef"},
    "RH20T-fjy/rh20t_cfg1": {"state_slot": "right_joint", "action_slot": "right_joint"},
    "RH20T-fjy/rh20t_cfg2": {"state_slot": "right_joint", "action_slot": "right_joint"},
    "RH20T-fjy/rh20t_cfg3": {"state_slot": "right_joint", "action_slot": "right_joint"},
    "RH20T-fjy/rh20t_cfg4": {"state_slot": "right_joint", "action_slot": "right_joint"},
    "RH20T-fjy/rh20t_cfg5": {"state_slot": "right_joint", "action_slot": "right_joint"},
    "RH20T-fjy/rh20t_cfg6": {"state_slot": "right_joint", "action_slot": "right_joint"},
    "RH20T-fjy/rh20t_cfg7": {"state_slot": "right_joint", "action_slot": "right_joint"},

    # RoboCOIN 双臂: 同时比较左右 eef
    "RoboCOIN": {"state_slot": ["left_eef", "right_eef"], "action_slot": ["left_eef", "right_eef"]},
    "RoboCOIN_add0130": {"state_slot": ["left_eef", "right_eef"], "action_slot": ["left_eef", "right_eef"]},
    "RoboCOIN_add1201": {"state_slot": ["left_eef", "right_eef"], "action_slot": ["left_eef", "right_eef"]},

    # RoboMindV1.0 和 RoboMindV2.0: 用 left_joint+right_joint 做 L2 比较（所有 robot type 都有 joint）
    # 单臂 robot（franka_1rgb/3rgb, ur_1rgb 等）的 left_joint 在 unified 中为零，不影响比较 所以本质上可以把所有的都加进来计算
    "RoboMindV1.0": {"state_slot": ["left_joint", "right_joint"], "action_slot": ["left_joint", "right_joint"]},
    "RoboMindV2.0": {"state_slot": ["left_joint", "right_joint"], "action_slot": ["left_joint", "right_joint"]},
    "agibotworld_hyy": None,
    "xvla-soft-fold_franka_v3_franka": {"state_slot": "right_joint", "action_slot": "right_joint"},
}

# 3. 每个数据集的 L2 距离阈值（range-normalized per-frame L2）
# ============================================================
# episode 的 L2 score 超过该数据集阈值则标记为异常
# 设为 None 表示不做阈值筛选，仅输出报告
# 建议先跑一遍不设阈值，根据报告中的 p90/p95 统计值来确定
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

# 全局默认阈值：当数据集未在 DATASET_L2_THRESHOLD 中配置时使用
# 设为 None 表示未配置的数据集不做阈值筛选
DEFAULT_L2_THRESHOLD = 0.2
PLOT = False
NUM_WORKERS = 96

# 数据根目录（Lerobot_v21 的绝对路径）
DATA_ROOT = os.environ.get("VLA_DATA_ROOT", "/path/to/your/Lerobot_v21")

# 查找 meta 时需避开的目录名（不进入这些目录向下找）
SKIP_DIRS = {"video", "videos", "data", "unified_output"}

# episode 元数据文件名（按优先级尝试）
EPISODE_FILENAMES = ["episodes.jsonl", "episode.jsonl"]


"""

python convert_unified.py "$VLA_DATA_ROOT/BC_Z" --episodes 1 --dry-run \
    --output-dir "./output"

python filter_by_state_action_frame.py $VLA_DATA_ROOT/droid_RoboInter --force-reconvert --episode 100
"""