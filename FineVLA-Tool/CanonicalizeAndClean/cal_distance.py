# -*- coding: utf-8 -*-
"""
Distance computation module.
Measures similarity between state and action trajectories using range-normalization + per-frame L2.

Actions have been unified to absolute position representation, so state(t) and action(t) are compared frame by frame.

Computation steps:
1. Stationarity detection: if state is stationary but action is not (or vice versa), flag as anomalous directly
2. Range normalization: use the per-dimension (max - min) of the combined state+action as the denominator
3. Compute the normalized L2 distance per frame, take the mean as the score

Score interpretation: the average per-frame per-dimension deviation as a proportion of the active range.
For example, score=0.01 means the average deviation is 1% of the active range.

Note: function names and field names use the l2 prefix, denoting range-normalized per-frame L2 distance.
"""

import numpy as np

# Stationary trajectory detection threshold: if std of all dimensions is below this value, consider it stationary
STATIC_STD_THRESHOLD = 1e-3

# Minimum active range threshold: dimensions with range below this value are considered inactive and excluded from distance computation
# Prevents tiny noise in nearly-static dimensions from being amplified and dominating the result
MIN_ACTIVE_RANGE = 0.05

# Both-stationary threshold: if std of both state and action for a dimension is below this value,
# treat it as inactive even if range >= MIN_ACTIVE_RANGE (excludes constant-offset interference)
BOTH_STATIC_STD_THRESHOLD = 1e-3


def get_active_mask(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Return a boolean mask indicating active dimensions.

    A dimension is considered active only if both conditions are met:
    1. Combined range >= MIN_ACTIVE_RANGE
    2. State and action are not both stationary (at least one has std >= BOTH_STATIC_STD_THRESHOLD)
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
    """Normalize using the per-dimension active range of the combined state and action.

    Only dimensions where active_mask is True are normalized; the rest are zeroed out.

    Parameters
    ----------
    state : np.ndarray, shape (T, D)
    action : np.ndarray, shape (T, D)
    active_mask : np.ndarray, shape (D,), bool. None means all dimensions participate.

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
    """Determine whether a trajectory is stationary (std of all dimensions below threshold)."""
    return bool(np.all(trajectory.std(axis=0) < STATIC_STD_THRESHOLD))


def compute_vla_l2_score(state: np.ndarray, action: np.ndarray) -> float:
    """Compute the range-normalized per-frame L2 mean between state and action.

    Steps:
    0. Stationarity detection: one side stationary while the other is not -> return inf
    1. Normalize using the combined active range
    2. Compute normalized L2 distance per frame
    3. Take the mean across all frames

    Parameters
    ----------
    state : np.ndarray, shape (T, D)
    action : np.ndarray, shape (T, D)

    Returns
    -------
    float
        Range-normalized per-frame L2 mean.
        Returns NaN if trajectory is too short (T < 2).
        Returns inf if one side is stationary while the other is not.
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
        return 0.0  # All dimensions are inactive; both state and action are stationary

    s_norm, a_norm = normalize_by_range(state, action, active)
    per_frame_l2 = np.sqrt(np.sum((s_norm - a_norm) ** 2, axis=1))
    return float(per_frame_l2.mean())


def compute_episode_similarity(state: np.ndarray, action: np.ndarray) -> dict:
    """Compute similarity for a single episode, also returning intermediate results (for plotting).

    Parameters
    ----------
    state : np.ndarray, shape (T, D)
    action : np.ndarray, shape (T, D)

    Returns
    -------
    dict
        l2_score : float -- range-normalized per-frame L2 mean
        state_norm : np.ndarray (T, D_active) -- normalized state (active dimensions only)
        action_norm : np.ndarray (T, D_active) -- normalized action (active dimensions only)
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
