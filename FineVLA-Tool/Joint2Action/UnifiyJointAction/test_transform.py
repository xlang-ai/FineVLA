"""
测试 UnifiedStateActionTransform 的正确性。
模拟一个双臂机器人场景：
  - left_joint (7 DOF) + left_eef (xyz+rotvec=6D) + left_gripper (1D)
  - right_joint (7 DOF) + right_eef (xyz+rotvec=6D) + right_gripper (1D)

测试 3 种场景：
  1. abs→abs: 原始action是abs，目标也是abs，数值应完全保留
  2. abs→rel: 原始action是abs，目标是rel（相对s_0），验证相减逻辑
  3. delta→abs: 原始action是delta，目标是abs，验证累加逻辑
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import json
import copy

from utils.UnifyJointAction import UnifiedStateActionTransform, UNIFIED_STATE_ACTION_DIM, UNIFIED_STATE_ACTION_INDICES


def make_test_data(N=4, seed=42):
    """构造 N 个时间步的模拟双臂机器人数据。"""
    rng = np.random.RandomState(seed)

    left_joint_s  = rng.randn(N, 7).astype(np.float32)
    right_joint_s = rng.randn(N, 7).astype(np.float32)
    left_joint_a  = left_joint_s + rng.randn(N, 7).astype(np.float32) * 0.1
    right_joint_a = right_joint_s + rng.randn(N, 7).astype(np.float32) * 0.1

    left_eef_pos_s  = rng.randn(N, 3).astype(np.float32)
    left_eef_rot_s  = R.random(N, random_state=rng)
    left_eef_s = np.concatenate([left_eef_pos_s, left_eef_rot_s.as_rotvec().astype(np.float32)], axis=-1)

    right_eef_pos_s = rng.randn(N, 3).astype(np.float32)
    right_eef_rot_s = R.random(N, random_state=rng)
    right_eef_s = np.concatenate([right_eef_pos_s, right_eef_rot_s.as_rotvec().astype(np.float32)], axis=-1)

    left_eef_pos_a  = left_eef_pos_s + rng.randn(N, 3).astype(np.float32) * 0.05
    left_eef_rot_a  = R.random(N, random_state=rng)
    left_eef_a = np.concatenate([left_eef_pos_a, left_eef_rot_a.as_rotvec().astype(np.float32)], axis=-1)

    right_eef_pos_a = right_eef_pos_s + rng.randn(N, 3).astype(np.float32) * 0.05
    right_eef_rot_a = R.random(N, random_state=rng)
    right_eef_a = np.concatenate([right_eef_pos_a, right_eef_rot_a.as_rotvec().astype(np.float32)], axis=-1)

    left_grip_s  = rng.rand(N, 1).astype(np.float32)
    right_grip_s = rng.rand(N, 1).astype(np.float32)
    left_grip_a  = rng.rand(N, 1).astype(np.float32)
    right_grip_a = rng.rand(N, 1).astype(np.float32)

    data = {
        "state.left_joint":   left_joint_s,
        "state.left_eef":     left_eef_s,
        "state.left_gripper": left_grip_s,
        "state.right_joint":  right_joint_s,
        "state.right_eef":    right_eef_s,
        "state.right_gripper": right_grip_s,
        "action.left_joint":  left_joint_a,
        "action.left_eef":    left_eef_a,
        "action.left_gripper": left_grip_a,
        "action.right_joint": right_joint_a,
        "action.right_eef":   right_eef_a,
        "action.right_gripper": right_grip_a,
    }
    return data


APPLY_TO = [
    "state.left_joint", "state.left_eef", "state.left_gripper",
    "state.right_joint", "state.right_eef", "state.right_gripper",
    "action.left_joint", "action.left_eef", "action.left_gripper",
    "action.right_joint", "action.right_eef", "action.right_gripper",
]
MODALITY_PATH = Path(__file__).parent / "test_modality_config.json"


def print_unified_layout(unified: np.ndarray, mask: np.ndarray, label: str):
    """可视化 unified 向量中各 slot 的填充情况。"""
    print(f"\n{'='*60}")
    print(f"  {label}  shape={unified.shape}")
    print(f"{'='*60}")
    for name, (lo, hi) in UNIFIED_STATE_ACTION_INDICES.items():
        slot = unified[:, lo:hi]
        slot_mask = mask[:, lo:hi]
        active = slot_mask.any()
        if active:
            active_dims = int(slot_mask[0].sum())
            print(f"  [{lo:2d}:{hi:2d}] {name:15s} | 有效维度={active_dims} | 第0帧值={slot[0, :active_dims].round(4)}")
        else:
            print(f"  [{lo:2d}:{hi:2d}] {name:15s} | (空)")


def test_abs_to_abs():
    """测试1: abs → abs，原始和目标都是绝对表示，数值应直接保留。"""
    print("\n" + "#"*60)
    print("  测试1: abs_joint / abs_rotvec → abs_joint / abs_rotvec")
    print("#"*60)

    data = make_test_data()
    orig_left_joint_s = data["state.left_joint"].copy()
    orig_left_joint_a = data["action.left_joint"].copy()
    orig_left_eef_s = data["state.left_eef"].copy()
    orig_left_eef_a = data["action.left_eef"].copy()
    orig_left_grip_a = data["action.left_gripper"].copy()

    transform = UnifiedStateActionTransform(
        apply_to=APPLY_TO,
        modality_path=MODALITY_PATH,
        target_joint_state_type="abs_joint",
        target_eef_state_type="abs_rotvec",
        target_joint_action_type="abs_joint",
        target_eef_action_type="abs_rotvec",
    )
    result = transform.apply(data)

    unified_s = result["state.unified"]
    unified_a = result["action.unified"]
    mask_s = result["mask.state"]
    mask_a = result["mask.action"]

    print_unified_layout(unified_s, mask_s, "Unified State")
    print_unified_layout(unified_a, mask_a, "Unified Action")

    # 验证: joint 部分数值应完全一致
    lo, hi = UNIFIED_STATE_ACTION_INDICES["left_joint"]
    np.testing.assert_allclose(unified_s[:, lo:lo+7], orig_left_joint_s, atol=1e-5,
                               err_msg="left_joint state 不一致")
    np.testing.assert_allclose(unified_a[:, lo:lo+7], orig_left_joint_a, atol=1e-5,
                               err_msg="left_joint action 不一致")

    # 验证: eef 部分 (abs_rotvec → abs_rotvec) 数值应一致
    lo, hi = UNIFIED_STATE_ACTION_INDICES["left_eef"]
    np.testing.assert_allclose(unified_s[:, lo:lo+6], orig_left_eef_s, atol=1e-5,
                               err_msg="left_eef state 不一致")
    np.testing.assert_allclose(unified_a[:, lo:lo+6], orig_left_eef_a, atol=1e-5,
                               err_msg="left_eef action 不一致")

    # 验证: gripper
    lo, hi = UNIFIED_STATE_ACTION_INDICES["left_gripper"]
    np.testing.assert_allclose(unified_a[:, lo:lo+1], orig_left_grip_a, atol=1e-5,
                               err_msg="left_gripper action 不一致")

    # 验证: mask 正确
    assert mask_s[:, UNIFIED_STATE_ACTION_INDICES["left_hand"][0]:UNIFIED_STATE_ACTION_INDICES["left_hand"][1]].sum() == 0, \
        "left_hand 未使用但 mask 非零"

    print("\n  [PASS] abs → abs 全部验证通过!")


def test_abs_to_rel():
    """测试2: abs → rel，action 应变成相对第一帧 state 的偏移。"""
    print("\n" + "#"*60)
    print("  测试2: abs → rel_joint / rel_rotvec")
    print("#"*60)

    data = make_test_data()
    orig_left_joint_s = data["state.left_joint"].copy()
    orig_left_joint_a = data["action.left_joint"].copy()
    orig_left_eef_s = data["state.left_eef"].copy()
    orig_left_eef_a = data["action.left_eef"].copy()

    transform = UnifiedStateActionTransform(
        apply_to=APPLY_TO,
        modality_path=MODALITY_PATH,
        target_joint_state_type="abs_joint",
        target_eef_state_type="abs_rotvec",
        target_joint_action_type="rel_joint",
        target_eef_action_type="rel_rotvec",
    )
    result = transform.apply(data)

    unified_a = result["action.unified"]

    # 验证 joint: rel = abs_action - abs_state[0]
    lo, _ = UNIFIED_STATE_ACTION_INDICES["left_joint"]
    expected_joint_rel = orig_left_joint_a - orig_left_joint_s[0]
    np.testing.assert_allclose(unified_a[:, lo:lo+7], expected_joint_rel, atol=1e-5,
                               err_msg="left_joint rel action 不一致")

    # 验证 eef: rel 位置 = R_s0^{-1} @ (pos_a - pos_s0)
    lo, _ = UNIFIED_STATE_ACTION_INDICES["left_eef"]
    pos_s = orig_left_eef_s[:, :3]
    rot_s = R.from_rotvec(orig_left_eef_s[:, 3:])
    pos_a = orig_left_eef_a[:, :3]
    rot_a = R.from_rotvec(orig_left_eef_a[:, 3:])

    expected_pos = rot_s[0].inv().apply(pos_a - pos_s[0])
    expected_rot = rot_s[0].inv() * rot_a
    expected_eef_rel = np.concatenate([expected_pos, expected_rot.as_rotvec()], axis=-1)

    np.testing.assert_allclose(unified_a[:, lo:lo+6], expected_eef_rel, atol=1e-4,
                               err_msg="left_eef rel action 不一致")

    print_unified_layout(result["action.unified"], result["mask.action"], "Unified Action (rel)")
    print("\n  [PASS] abs → rel 全部验证通过!")


def test_delta_to_abs():
    """测试3: delta → abs，需要修改 modality config 中 action type 为 delta。"""
    print("\n" + "#"*60)
    print("  测试3: delta_joint / delta_rotvec → abs_joint / abs_rotvec")
    print("#"*60)

    # 创建一个 action type 为 delta 的临时 config
    with open(MODALITY_PATH) as f:
        cfg = json.load(f)
    for key in cfg["action"]:
        if "eef" in key:
            cfg["action"][key]["type"] = "delta_rotvec"
        else:
            cfg["action"][key]["type"] = "delta_joint"

    delta_config_path = Path(__file__).parent / "test_modality_config_delta.json"
    with open(delta_config_path, "w") as f:
        json.dump(cfg, f, indent=4)

    data = make_test_data()
    orig_left_joint_s = data["state.left_joint"].copy()
    orig_right_joint_s = data["state.right_joint"].copy()
    orig_left_eef_s = data["state.left_eef"].copy()

    # 构造 delta action: delta = abs_action - abs_state (对于 joint)
    # 先记录原始的 abs action
    abs_left_joint_a = data["action.left_joint"].copy()
    abs_right_joint_a = data["action.right_joint"].copy()

    # 把 data 中的 action 改成 delta 形式
    data["action.left_joint"] = abs_left_joint_a - orig_left_joint_s
    data["action.right_joint"] = abs_right_joint_a - orig_right_joint_s

    # eef delta: 位置 delta = pos_a - pos_s, 旋转 delta = rot_a * rot_s^{-1}
    for side in ["left", "right"]:
        s_key = f"state.{side}_eef"
        a_key = f"action.{side}_eef"
        s_arr = data[s_key].copy()
        a_arr = data[a_key].copy()
        pos_s = s_arr[:, :3]
        rot_s = R.from_rotvec(s_arr[:, 3:])
        pos_a = a_arr[:, :3]
        rot_a = R.from_rotvec(a_arr[:, 3:])
        delta_pos = pos_a - pos_s
        delta_rot = rot_a * rot_s.inv()
        data[a_key] = np.concatenate([delta_pos, delta_rot.as_rotvec()], axis=-1).astype(np.float32)

    # 记录 delta 后的原始 abs action 用于验证
    expected_left_joint_abs = abs_left_joint_a

    transform = UnifiedStateActionTransform(
        apply_to=APPLY_TO,
        modality_path=delta_config_path,
        target_joint_state_type="abs_joint",
        target_eef_state_type="abs_rotvec",
        target_joint_action_type="abs_joint",
        target_eef_action_type="abs_rotvec",
    )
    result = transform.apply(data)
    unified_a = result["action.unified"]

    # 验证: delta + state 应还原回 abs
    lo, _ = UNIFIED_STATE_ACTION_INDICES["left_joint"]
    np.testing.assert_allclose(unified_a[:, lo:lo+7], expected_left_joint_abs, atol=1e-5,
                               err_msg="delta→abs left_joint 还原失败")

    print_unified_layout(result["action.unified"], result["mask.action"], "Unified Action (delta→abs)")
    print("\n  [PASS] delta → abs 全部验证通过!")

    delta_config_path.unlink()


def test_rotation_format_conversion():
    """测试4: 旋转格式转换 rotvec → euler → quat，验证数值一致性。"""
    print("\n" + "#"*60)
    print("  测试4: 旋转格式转换 abs_rotvec → abs_euler (state), abs_quat (action)")
    print("#"*60)

    data = make_test_data()
    orig_left_eef_s = data["state.left_eef"].copy()
    orig_left_eef_a = data["action.left_eef"].copy()

    transform = UnifiedStateActionTransform(
        apply_to=APPLY_TO,
        modality_path=MODALITY_PATH,
        target_joint_state_type="abs_joint",
        target_eef_state_type="abs_euler",
        target_joint_action_type="abs_joint",
        target_eef_action_type="abs_quat",
    )
    result = transform.apply(data)

    unified_s = result["state.unified"]
    unified_a = result["action.unified"]

    # 验证 state eef: rotvec → euler
    lo, _ = UNIFIED_STATE_ACTION_INDICES["left_eef"]
    pos_s = orig_left_eef_s[:, :3]
    rot_s = R.from_rotvec(orig_left_eef_s[:, 3:])
    expected_s = np.concatenate([pos_s, rot_s.as_euler("xyz")], axis=-1)
    np.testing.assert_allclose(unified_s[:, lo:lo+6], expected_s, atol=1e-4,
                               err_msg="rotvec→euler state 不一致")

    # 验证 action eef: rotvec → quat (abs→abs，xyz+xyzw=7维)
    pos_a = orig_left_eef_a[:, :3]
    rot_a = R.from_rotvec(orig_left_eef_a[:, 3:])
    expected_a = np.concatenate([pos_a, rot_a.as_quat()], axis=-1)
    np.testing.assert_allclose(unified_a[:, lo:lo+7], expected_a, atol=1e-4,
                               err_msg="rotvec→quat action 不一致")

    print(f"  state left_eef slot [{lo}:{lo+6}]: xyz+euler (6D)")
    print(f"    第0帧: {unified_s[0, lo:lo+6].round(4)}")
    print(f"  action left_eef slot [{lo}:{lo+7}]: xyz+quat (7D)")
    print(f"    第0帧: {unified_a[0, lo:lo+7].round(4)}")
    print("\n  [PASS] 旋转格式转换验证通过!")


def test_mask_correctness():
    """测试5: 验证 mask 的正确性——有数据的 slot 为 True，空 slot 为 False。"""
    print("\n" + "#"*60)
    print("  测试5: mask 正确性验证")
    print("#"*60)

    data = make_test_data()
    transform = UnifiedStateActionTransform(
        apply_to=APPLY_TO,
        modality_path=MODALITY_PATH,
        target_joint_state_type="abs_joint",
        target_eef_state_type="abs_rotvec",
        target_joint_action_type="abs_joint",
        target_eef_action_type="abs_rotvec",
    )
    result = transform.apply(data)
    mask_s = result["mask.state"]
    mask_a = result["mask.action"]

    expected_active = {
        "left_joint": 7,
        "left_eef": 6,
        "left_gripper": 1,
        "left_hand": 0,
        "right_joint": 7,
        "right_eef": 6,
        "right_gripper": 1,
        "right_hand": 0,
        "reserved": 0,
    }

    print(f"  {'slot':15s} | {'范围':8s} | {'预期':4s} | {'state mask':10s} | {'action mask':10s} | 状态")
    print(f"  {'-'*70}")
    all_ok = True
    for name, (lo, hi) in UNIFIED_STATE_ACTION_INDICES.items():
        exp = expected_active[name]
        s_active = int(mask_s[0, lo:hi].sum())
        a_active = int(mask_a[0, lo:hi].sum())
        ok = s_active == exp and a_active == exp
        status = "OK" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  {name:15s} | [{lo:2d}:{hi:2d}] | {exp:4d} | {s_active:10d} | {a_active:10d} | {status}")

    assert all_ok, "mask 检查失败"
    print("\n  [PASS] mask 全部验证通过!")


if __name__ == "__main__":
    print("="*60)
    print("  UnifiedStateActionTransform 测试")
    print("="*60)

    test_abs_to_abs()
    test_abs_to_rel()
    test_delta_to_abs()
    test_rotation_format_conversion()
    test_mask_correctness()

    print("\n" + "="*60)
    print("  全部 5 个测试通过!")
    print("="*60)
