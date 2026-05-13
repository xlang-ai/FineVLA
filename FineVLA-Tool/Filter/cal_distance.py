# -*- coding: utf-8 -*-
"""
距离计算模块。
使用 range-norm + per-frame L2 度量 state 轨迹与 action 轨迹之间的相似性。

action 已统一为绝对位置表示，因此直接逐帧比较 state(t) 与 action(t)。

计算步骤：
1. 静止检测：state 静止但 action 不静止（或反过来），直接判定异常
2. Range 归一化：用 state+action 联合的每维 (max - min) 作为分母
3. 逐帧计算归一化后的 L2 距离，取均值作为 score

score 含义：每帧每维的平均偏差占活动范围的比例。
例如 score=0.01 表示平均偏差是活动范围的 1%。

注意：函数名和字段名均使用 l2 前缀，表示 range-normalized per-frame L2 距离。
"""

import numpy as np

# 静止轨迹检测阈值：如果某条轨迹所有维度的 std 都低于此值，视为静止
STATIC_STD_THRESHOLD = 1e-3

# 最小活动范围阈值：range 低于此值的维度视为不活动，不参与距离计算
# 避免几乎不动的维度上微小噪声被放大主导结果
MIN_ACTIVE_RANGE = 0.05

# 双方静止阈值：如果某维度 state 和 action 的 std 都低于此值，
# 即使 range >= MIN_ACTIVE_RANGE 也视为不活动（排除恒定偏移干扰）
BOTH_STATIC_STD_THRESHOLD = 1e-3


def get_active_mask(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """返回活动维度的 bool mask。

    同时满足以下两个条件的维度才被视为活动：
    1. 联合 range >= MIN_ACTIVE_RANGE
    2. state 和 action 不同时静止（至少一方 std >= BOTH_STATIC_STD_THRESHOLD）
    """
    combined = np.concatenate([state, action], axis=0)
    dim_range = combined.max(axis=0) - combined.min(axis=0)
    range_ok = dim_range >= MIN_ACTIVE_RANGE

    state_std = state.std(axis=0)
    action_std = action.std(axis=0)
    both_static = (state_std < BOTH_STATIC_STD_THRESHOLD) & (action_std < BOTH_STATIC_STD_THRESHOLD)

    return range_ok & ~both_static


def normalize_by_range(state: np.ndarray, action: np.ndarray,
                       active_mask: np.ndarray = None) -> tuple:
    """用 state 和 action 联合的每维活动范围做归一化。

    只对 active_mask 为 True 的维度做归一化，其余维度置零。

    Parameters
    ----------
    state : np.ndarray, shape (T, D)
    action : np.ndarray, shape (T, D)
    active_mask : np.ndarray, shape (D,), bool. None 表示全部参与。

    Returns
    -------
    (state_norm, action_norm) : tuple of np.ndarray, each shape (T, D_active)
    """
    if active_mask is not None:
        state = state[:, active_mask]
        action = action[:, active_mask]

    combined = np.concatenate([state, action], axis=0)  # (2T, D)
    dim_range = combined.max(axis=0) - combined.min(axis=0)
    dim_range = np.where(dim_range < 1e-6, 1.0, dim_range)
    return state / dim_range, action / dim_range


def is_static(trajectory: np.ndarray) -> bool:
    """判断轨迹是否静止（所有维度 std 都低于阈值）。"""
    return bool(np.all(trajectory.std(axis=0) < STATIC_STD_THRESHOLD))


def compute_vla_l2_score(state: np.ndarray, action: np.ndarray) -> float:
    """计算 state 与 action 的 range-norm per-frame L2 均值。

    步骤：
    0. 静止检测：一方静止另一方不静止 → 返回 inf
    1. 用联合活动范围做归一化
    2. 逐帧计算归一化 L2 距离
    3. 取所有帧的均值

    Parameters
    ----------
    state : np.ndarray, shape (T, D)
    action : np.ndarray, shape (T, D)

    Returns
    -------
    float
        range-normalized per-frame L2 均值。
        轨迹过短（T < 2）时返回 NaN。
        一方静止另一方不静止时返回 inf。
    """
    T = state.shape[0]
    if T < 2:
        return float("nan")

    state_static = is_static(state)
    action_static = is_static(action)
    if state_static != action_static:
        return float("inf")

    active = get_active_mask(state, action)
    if not np.any(active):
        return 0.0  # 所有维度都不活动，state 和 action 都静止

    s_norm, a_norm = normalize_by_range(state, action, active)
    per_frame_l2 = np.sqrt(np.sum((s_norm - a_norm) ** 2, axis=1))
    return float(per_frame_l2.mean())


def compute_episode_similarity(state: np.ndarray, action: np.ndarray) -> dict:
    """计算单个 episode 的相似性，同时返回中间结果（用于绘图）。

    Parameters
    ----------
    state : np.ndarray, shape (T, D)
    action : np.ndarray, shape (T, D)

    Returns
    -------
    dict
        l2_score : float — range-normalized per-frame L2 均值
        state_norm : np.ndarray (T, D_active) — 归一化后的 state（仅活动维度）
        action_norm : np.ndarray (T, D_active) — 归一化后的 action（仅活动维度）
    """
    T = state.shape[0]
    if T < 2:
        return {
            "l2_score": float("nan"),
            "state_norm": state,
            "action_norm": action,
        }

    state_static = is_static(state)
    action_static = is_static(action)
    active = get_active_mask(state, action)

    if state_static != action_static:
        s_norm, a_norm = normalize_by_range(state, action, active)
        return {
            "l2_score": float("inf"),
            "state_norm": s_norm,
            "action_norm": a_norm,
        }

    if not np.any(active):
        return {
            "l2_score": 0.0,
            "state_norm": state,
            "action_norm": action,
        }

    s_norm, a_norm = normalize_by_range(state, action, active)
    per_frame_l2 = np.sqrt(np.sum((s_norm - a_norm) ** 2, axis=1))
    score = float(per_frame_l2.mean())

    return {
        "l2_score": score,
        "state_norm": s_norm,
        "action_norm": a_norm,
    }
