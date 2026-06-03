"""
基于 DTW 的轨迹相似度计算。

支持两种帧代价模式：
  - EEF 模式 (rot_type = "quaternion" | "euler"):
      向量 [x,y,z, qx,qy,qz,qw, gripper] (8D)
      代价 = w_pos * L2(pos) + w_rot * geodesic(quat) + w_grip * |grip|
  - Joint 模式 (rot_type = "none"):
      向量 [joints..., gripper] (D 可变)
      代价 = w_pos * L2(joints) + w_grip * |grip|  (w_rot 不使用)
"""

from __future__ import annotations

import numpy as np
from numba import njit


# ──────────────────────── 单帧代价函数 ────────────────────────

@njit(cache=True)
def _quat_geodesic(q1: np.ndarray, q2: np.ndarray) -> float:
    """四元数测地线角距离（弧度）。q 与 -q 表示同一姿态。"""
    dot = q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]
    dot = abs(dot)
    if dot > 1.0:
        dot = 1.0
    return 2.0 * np.arccos(dot)


@njit(cache=True)
def frame_cost_eef(
    a: np.ndarray,
    b: np.ndarray,
    w_pos: float = 1.0,
    w_rot: float = 1.0,
    w_grip: float = 0.03,
) -> float:
    """
    EEF 模式帧代价。
    a, b: 长度 8 的向量 [x,y,z, qx,qy,qz,qw, gripper]
    """
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    d_pos = np.sqrt(dx*dx + dy*dy + dz*dz)

    d_rot = _quat_geodesic(a[3:7], b[3:7])

    d_grip = abs(a[7] - b[7])

    return w_pos * d_pos + w_rot * d_rot + w_grip * d_grip


@njit(cache=True)
def frame_cost_joint(
    a: np.ndarray,
    b: np.ndarray,
    w_pos: float = 1.0,
    w_grip: float = 0.03,
) -> float:
    """
    Joint 模式帧代价。
    a, b: [joints..., gripper]，最后一个元素是 gripper。
    w_pos 作用于 joints 的 L2 距离。
    """
    D = a.shape[0]
    sq_sum = 0.0
    for k in range(D - 1):
        diff = a[k] - b[k]
        sq_sum += diff * diff
    d_joint = np.sqrt(sq_sum)

    d_grip = abs(a[D - 1] - b[D - 1])

    return w_pos * d_joint + w_grip * d_grip


# ──────────────────────── DTW 核心 (EEF) ────────────────────────

@njit(cache=True)
def _dtw_cost_matrix_eef(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    w_pos: float,
    w_rot: float,
    w_grip: float,
) -> np.ndarray:
    N = seq_a.shape[0]
    M = seq_b.shape[0]
    dp = np.full((N + 1, M + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost = frame_cost_eef(seq_a[i - 1], seq_b[j - 1], w_pos, w_rot, w_grip)
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return dp


@njit(cache=True)
def _dtw_with_window_eef(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    w_pos: float,
    w_rot: float,
    w_grip: float,
    window: int,
) -> float:
    N = seq_a.shape[0]
    M = seq_b.shape[0]
    w = max(window, abs(N - M))
    dp = np.full((N + 1, M + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, N + 1):
        j_start = max(1, i - w)
        j_end = min(M, i + w) + 1
        for j in range(j_start, j_end):
            cost = frame_cost_eef(seq_a[i - 1], seq_b[j - 1], w_pos, w_rot, w_grip)
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return dp[N, M]


# ──────────────────────── DTW 核心 (Joint) ────────────────────────

@njit(cache=True)
def _dtw_cost_matrix_joint(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    w_pos: float,
    w_grip: float,
) -> np.ndarray:
    N = seq_a.shape[0]
    M = seq_b.shape[0]
    dp = np.full((N + 1, M + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost = frame_cost_joint(seq_a[i - 1], seq_b[j - 1], w_pos, w_grip)
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return dp


@njit(cache=True)
def _dtw_with_window_joint(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    w_pos: float,
    w_grip: float,
    window: int,
) -> float:
    N = seq_a.shape[0]
    M = seq_b.shape[0]
    w = max(window, abs(N - M))
    dp = np.full((N + 1, M + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, N + 1):
        j_start = max(1, i - w)
        j_end = min(M, i + w) + 1
        for j in range(j_start, j_end):
            cost = frame_cost_joint(seq_a[i - 1], seq_b[j - 1], w_pos, w_grip)
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return dp[N, M]


# ──────────────────────── 路径回溯 ────────────────────────

@njit(cache=True)
def _backtrace_length(dp: np.ndarray) -> int:
    """回溯 DTW 路径以获取路径长度（用于归一化）。"""
    i = dp.shape[0] - 1
    j = dp.shape[1] - 1
    length = 0
    while i > 0 or j > 0:
        length += 1
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            candidates = (dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1])
            argmin = 0
            if candidates[1] < candidates[argmin]:
                argmin = 1
            if candidates[2] < candidates[argmin]:
                argmin = 2
            if argmin == 0:
                i -= 1
                j -= 1
            elif argmin == 1:
                i -= 1
            else:
                j -= 1
    return length


# ──────────────────────── 路径回溯（完整路径） ────────────────────────

def _backtrace_path(dp: np.ndarray) -> list[tuple[int, int]]:
    """回溯 DTW 得到完整对齐路径 [(i,j), ...]，0-indexed。"""
    i = dp.shape[0] - 1
    j = dp.shape[1] - 1
    path = []
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            vals = (dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1])
            best = 0
            if vals[1] < vals[best]:
                best = 1
            if vals[2] < vals[best]:
                best = 2
            if best == 0:
                i -= 1; j -= 1
            elif best == 1:
                i -= 1
            else:
                j -= 1
    path.reverse()
    return path


# ──────────────────────── 两阶段 DTW ────────────────────────

def two_stage_dtw_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    w_pos: float = 1.0,
    w_rot: float = 1.0,
    w_grip: float = 1.0,
    window: int | None = None,
    rot_type: str = "quaternion",
    grip_threshold: float = 0.01,
) -> dict:
    """
    两阶段 DTW 距离：

    阶段 1：仅用 pos+rot 做 DTW 对齐 (w_grip=0)，得到对齐路径
            D_eef = eef_raw_cost / path_length

    阶段 2：沿阶段 1 的对齐路径，累计 gripper 差异
            D_grip = grip_total_cost / path_length

    最终距离：D = D_eef + w_grip × D_grip

    Returns
    -------
    dict 包含所有中间量，方便分析
    """
    seq_a = np.ascontiguousarray(seq_a, dtype=np.float64)
    seq_b = np.ascontiguousarray(seq_b, dtype=np.float64)
    N, M = seq_a.shape[0], seq_b.shape[0]

    is_joint = (rot_type == "none")
    grip_dim = seq_a.shape[1] - 1 if is_joint else 7

    # ── 阶段 1: EEF-only DTW (gripper weight = 0) ──
    if window is not None:
        if is_joint:
            eef_raw = _dtw_with_window_joint(seq_a, seq_b, w_pos, 0.0, window)
        else:
            eef_raw = _dtw_with_window_eef(seq_a, seq_b, w_pos, w_rot, 0.0, window)
        path_len = N + M  # approximate for windowed
        path = None
        d_eef = eef_raw / path_len
    else:
        if is_joint:
            dp = _dtw_cost_matrix_joint(seq_a, seq_b, w_pos, 0.0)
        else:
            dp = _dtw_cost_matrix_eef(seq_a, seq_b, w_pos, w_rot, 0.0)
        eef_raw = float(dp[N, M])
        path = _backtrace_path(dp)
        path_len = len(path)
        d_eef = eef_raw / path_len

    # ── 阶段 2: 沿对齐路径统计 gripper 差异 ──
    grip_total = 0.0
    grip_diff_frames = 0

    if path is not None:
        for (pi, pj) in path:
            diff = abs(float(seq_a[pi, grip_dim]) - float(seq_b[pj, grip_dim]))
            grip_total += diff
            if diff > grip_threshold:
                grip_diff_frames += 1

    d_grip = grip_total / path_len if path_len > 0 else 0.0
    grip_mismatch_ratio = grip_diff_frames / path_len if path_len > 0 else 0.0

    # ── 最终距离 ──
    d_total = d_eef + w_grip * d_grip

    return {
        "d_total": float(d_total),
        "d_eef": float(d_eef),
        "d_grip": float(d_grip),
        "eef_raw_cost": float(eef_raw),
        "grip_raw_cost": float(grip_total),
        "path_length": path_len,
        "grip_diff_frames": grip_diff_frames,
        "grip_mismatch_ratio": float(grip_mismatch_ratio),
        "w_pos": w_pos, "w_rot": w_rot, "w_grip": w_grip,
        "seq_a_length": N,
        "seq_b_length": M,
    }


# ──────────────────────── 统一接口 ────────────────────────

def dtw_distance(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    w_pos: float = 1.0,
    w_rot: float = 1.0,
    w_grip: float = 1.0,
    normalize: bool = True,
    window: int | None = None,
    rot_type: str = "quaternion",
    two_stage: bool = False,
) -> float:
    """
    计算两条轨迹的 DTW 距离。

    Parameters
    ----------
    two_stage : True 时使用两阶段 DTW（EEF 对齐 + gripper 沿路径统计）
    """
    if two_stage:
        result = two_stage_dtw_distance(
            seq_a, seq_b, w_pos, w_rot, w_grip,
            window=window, rot_type=rot_type,
        )
        return result["d_total"]

    seq_a = np.ascontiguousarray(seq_a, dtype=np.float64)
    seq_b = np.ascontiguousarray(seq_b, dtype=np.float64)

    is_joint = (rot_type == "none")

    if window is not None:
        if is_joint:
            raw = _dtw_with_window_joint(seq_a, seq_b, w_pos, w_grip, window)
        else:
            raw = _dtw_with_window_eef(seq_a, seq_b, w_pos, w_rot, w_grip, window)
        if normalize:
            raw /= (seq_a.shape[0] + seq_b.shape[0])
        return float(raw)

    if is_joint:
        dp = _dtw_cost_matrix_joint(seq_a, seq_b, w_pos, w_grip)
    else:
        dp = _dtw_cost_matrix_eef(seq_a, seq_b, w_pos, w_rot, w_grip)

    raw = dp[seq_a.shape[0], seq_b.shape[0]]

    if normalize:
        path_len = _backtrace_length(dp)
        raw /= path_len

    return float(raw)


# ──────────────────────── 批量距离矩阵 ────────────────────────

def _dtw_worker(args):
    """顶层函数，供 multiprocessing 序列化调用。"""
    i, j, sa, sb, wp, wr, wg, norm, win, rtype, ts = args
    return i, j, dtw_distance(sa, sb, wp, wr, wg, norm, win, rtype, ts)


def compute_distance_matrix(
    trajectories: list,
    w_pos: float = 1.0,
    w_rot: float = 1.0,
    w_grip: float = 1.0,
    normalize: bool = True,
    window: int | None = None,
    n_jobs: int = 1,
    rot_type: str = "quaternion",
    two_stage: bool = False,
) -> np.ndarray:
    """
    计算 N 条轨迹的两两 DTW 距离矩阵。

    Parameters
    ----------
    two_stage : True 时使用两阶段 DTW
    """
    N = len(trajectories)
    dist_matrix = np.zeros((N, N), dtype=np.float64)

    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    total = len(pairs)

    if n_jobs > 1:
        from multiprocessing import Pool

        seqs = [t.combined for t in trajectories]
        tasks = [
            (i, j, seqs[i], seqs[j], w_pos, w_rot, w_grip, normalize, window, rot_type, two_stage)
            for i, j in pairs
        ]
        with Pool(n_jobs) as pool:
            for idx, (i, j, d) in enumerate(pool.imap_unordered(_dtw_worker, tasks, chunksize=4)):
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
                if (idx + 1) % 50 == 0:
                    print(f"  [{idx+1}/{total}] pairs computed")
    else:
        for idx, (i, j) in enumerate(pairs):
            d = dtw_distance(
                trajectories[i].combined,
                trajectories[j].combined,
                w_pos, w_rot, w_grip, normalize, window, rot_type, two_stage,
            )
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
            if (idx + 1) % 50 == 0:
                print(f"  [{idx+1}/{total}] pairs computed")

    return dist_matrix
