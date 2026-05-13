"""
通用脚本：读取 LeRobot v2.1 数据集，调用 UnifyJointAction 将 state/action 转为统一 80 维表示。

============================================================
两种运行模式
============================================================

1. dry-run 模式 (--dry-run)
   - 只读取数据、执行转换、打印结果摘要
   - 不写入任何文件，不修改原始数据
   - 用途：快速验证 modality.json 配置是否正确、转换结果是否合理

2. 正常模式 (不加 --dry-run)
   - 读取数据、执行转换、打印结果摘要
   - 将转换结果保存为 .parquet 文件到 --output-dir 指定的目录
   - 同时生成 unified_meta.json 记录向量布局和 target 类型
   - 如果未指定 --output-dir，默认保存到 <dataset_path>/unified_output/
   - 不修改原始 parquet 数据

============================================================
目标表示类型（hardcoded）
============================================================

统一后的目标类型在脚本中固定为：
  - joint state:  abs_joint   (绝对关节角)
  - eef state:    abs_rotvec  (绝对位姿，旋转用旋转向量)
  - joint action: abs_joint   (绝对关节角)
  - eef action:   abs_rotvec  (绝对位姿，旋转用旋转向量)
如需修改，直接编辑 main() 中 TARGET_* 常量。

============================================================
用法
============================================================

    python convert_unified.py <dataset_path> [--episodes N] [--dry-run] [--output-dir DIR]

示例:
    # dry-run: 测试前 2 个 episode，只看结果不写文件
    python convert_unified.py /path/to/dataset --episodes 2 --dry-run

    # 正常模式: 转换前 5 个 episode，输出到指定目录
    python convert_unified.py /path/to/dataset --episodes 5 --output-dir /tmp/unified_output

    # 正常模式: 转换全部 episode，输出到数据集下的 unified_output/
    python convert_unified.py /path/to/dataset --episodes 9999
"""

import sys
import os
import argparse
import json
import glob
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


def list_episodes(dataset_path: str) -> list[str]:
    """按文件名排序返回所有 episode parquet 文件路径。"""
    pattern = os.path.join(dataset_path, "data", "chunk-*", "episode_*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files found matching: {pattern}")
    return files


# 从parquet文件中读取数据，按照modality_cfg映射成UnifyJointAction需要的格式。
def extract_episode_data(parquet_path: str, modality_cfg: dict) -> tuple[dict, np.ndarray]:
    """
    从 parquet 文件中读取数据，按照 modality_cfg 映射成 UnifyJointAction 需要的格式。

    Returns:
        data dict: keys 为 "state.<unified_name>" / "action.<unified_name>"，values 为 (N, D) np.ndarray
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

            # 支持从一个大数组中切片取子集 (start/end 索引)
            col_start = field_info.get("start")
            col_end = field_info.get("end")
            if col_start is not None and col_end is not None:
                arr = arr[:, col_start:col_end]

            expected_dim = field_info.get("dim")
            # 验证一下modality_cfg中的dim是否正确，是否和action 中的约束匹配
            if expected_dim is not None and arr.shape[1] != expected_dim:
                raise ValueError(
                    f"{category}.{unified_name}: expected dim={expected_dim}, "
                    f"got {arr.shape[1]} from '{original_key}'"
                )

            data[f"{category}.{unified_name}"] = arr

    return data, frame_index


def build_apply_to(modality_cfg: dict) -> list[str]:
    """构建 apply_to 列表：先列所有 state key，再列所有 action key。"""
    keys = []
    for unified_name in modality_cfg.get("state", {}):
        keys.append(f"state.{unified_name}")
    for unified_name in modality_cfg.get("action", {}):
        keys.append(f"action.{unified_name}")
    return keys


def print_result(result: dict, episode_path: str):
    """打印转换结果的摘要信息。"""
    ep_name = os.path.basename(episode_path)
    unified_s = result["state.unified"]
    unified_a = result["action.unified"]
    mask_s = result["mask.state"]
    mask_a = result["mask.action"]

    print(f"\n{'='*70}")
    print(f"  Episode: {ep_name}  |  frames={unified_s.shape[0]}")
    print(f"{'='*70}")

    print(f"\n  State unified: shape={unified_s.shape}, dtype={unified_s.dtype}")
    print(f"  Action unified: shape={unified_a.shape}, dtype={unified_a.dtype}")

    print(f"\n  {'Slot':15s} | {'Range':8s} | {'State dims':10s} | {'Action dims':11s} | {'State[0] sample'}")
    print(f"  {'-'*75}")
    for name, (lo, hi) in UNIFIED_STATE_ACTION_INDICES.items():
        s_active = int(mask_s[0, lo:hi].sum())
        a_active = int(mask_a[0, lo:hi].sum())
        sample = ""
        if s_active > 0:
            vals = unified_s[0, lo:lo+s_active]
            sample = np.array2string(vals, precision=4, suppress_small=True, max_line_width=200)
        else:
            sample = "(empty)"
        print(f"  {name:15s} | [{lo:2d}:{hi:2d}] | {s_active:10d} | {a_active:11d} | {sample}")


def save_result(result: dict, frame_index: np.ndarray, output_dir: str, episode_name: str):
    """将转换结果保存为 parquet 文件，包含 frame_index 和统一向量。"""
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
    """保存统一向量的元数据说明到 output_dir/unified_meta.json。"""
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


# config: target type and dataset path
TARGET_JOINT_STATE_TYPE = "abs_joint"
TARGET_EEF_STATE_TYPE = "abs_quat"
TARGET_JOINT_ACTION_TYPE = "abs_joint"
TARGET_EEF_ACTION_TYPE = "abs_quat"


"""
测试：
python convert_unified.py "/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21/Galaxea-Open-World-Dataset/Adjust_The_Air_Conditioner_Temperature_20250711_006" --episodes 1 --dry-run
"""
def main():
    parser = argparse.ArgumentParser(description="Convert dataset to unified state/action representation")
    parser.add_argument("dataset_path", help="Path to the LeRobot v2.1 dataset root")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to process (default: 1)")
    parser.add_argument("--dry-run", action="store_true", help="Only print results, do not save files")
    parser.add_argument("--output-dir", default=None, help="Output directory for unified parquet files")
    args = parser.parse_args()

    dataset_path = args.dataset_path.rstrip("/")
    modality_cfg = load_modality(dataset_path)
    episode_files = list_episodes(dataset_path)
    n_episodes = min(args.episodes, len(episode_files))

    print(f"Dataset: {dataset_path}")
    print(f"Total episodes available: {len(episode_files)}")
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

    for i in range(n_episodes):
        ep_path = episode_files[i]
        data, frame_index = extract_episode_data(ep_path, modality_cfg)
        result = transform.apply(data)
        print_result(result, ep_path)

        if not args.dry_run:
            ep_name = os.path.basename(ep_path)
            saved_path = save_result(result, frame_index, args.output_dir, ep_name)
            print(f"  Saved to: {saved_path}")

    print(f"\nDone. Processed {n_episodes} episodes.")


if __name__ == "__main__":
    main()
