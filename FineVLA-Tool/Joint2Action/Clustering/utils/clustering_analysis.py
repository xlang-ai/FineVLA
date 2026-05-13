"""
基于 DTW 距离矩阵的轨迹聚类与可视化。
支持层次聚类、K-Medoids，以及多种可视化方式。
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS


# ──────────────────────── 自动选 k ────────────────────────

def auto_select_k(
    linkage_matrix: np.ndarray,
    max_k: int = 10,
    min_k: int = 2,
) -> tuple[int, dict]:
    """
    基于 dendrogram 合并高度的跳跃自动选择最佳 k。

    定义
    ----
    merge_height(k) = Z[N-k, 2]
        把 k 个簇合并为 k-1 个所需的距离。

    abs_gap(k)  = merge_height(k) - merge_height(k+1)
        "从 k 合并到 k-1 比从 k+1 合并到 k 贵多少"，越大说明 k 是好的分界。

    rel_gap(k)  = abs_gap(k) / merge_height(k+1)     (> 0)
        归一化到合并尺度，使不同量纲的数据集可比较。

    选 k 策略
    ---------
    1. 计算每个 k 的 abs_gap 和 rel_gap
    2. 对 rel_gap 排序，选最大的 rel_gap 对应的 k
    3. 置信度 = best_rel_gap / second_rel_gap（>= 2 视为高置信）

    Returns
    -------
    best_k : 推荐的 cluster 数
    info : 诊断信息
    """
    N = linkage_matrix.shape[0] + 1
    max_k = min(max_k, N - 1)
    if max_k < min_k:
        return min_k, {"note": "too few samples for auto-k"}

    merge_dists = linkage_matrix[:, 2]

    # merge_height(k) = 把 k 个簇合并到 k-1 个的代价
    merge_heights = {}
    for k in range(min_k, max_k + 1):
        idx = N - k
        if 0 <= idx < len(merge_dists):
            merge_heights[k] = float(merge_dists[idx])

    # abs_gap(k) = merge_height(k) - merge_height(k+1)
    abs_gaps = {}
    rel_gaps = {}
    for k in merge_heights:
        if (k + 1) not in merge_heights:
            continue
        ag = merge_heights[k] - merge_heights[k + 1]
        abs_gaps[k] = ag
        denom = merge_heights[k + 1]
        rel_gaps[k] = ag / denom if denom > 1e-12 else float("inf")

    if not rel_gaps:
        return min_k, {"note": "cannot compute gaps"}

    # 按 rel_gap 排序选 best_k
    ranked = sorted(rel_gaps.items(), key=lambda x: x[1], reverse=True)
    best_k = ranked[0][0]
    best_rel = ranked[0][1]
    second_rel = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence_ratio = best_rel / second_rel if second_rel > 1e-12 else float("inf")

    info = {
        "merge_heights": merge_heights,
        "abs_gaps": abs_gaps,
        "rel_gaps": {k: round(v, 4) for k, v in rel_gaps.items()},
        "ranked_k": [k for k, _ in ranked],
        "best_rel_gap": round(best_rel, 4),
        "second_rel_gap": round(second_rel, 4),
        "confidence_ratio": round(confidence_ratio, 2),
        "confident": confidence_ratio >= 2.0,
    }
    return best_k, info


# ──────────────────────── 聚类方法 ────────────────────────

def _build_linkage(dist_matrix: np.ndarray, method: str) -> np.ndarray:
    """构建 linkage 矩阵。"""
    condensed = squareform(dist_matrix, checks=False)
    if method == "ward":
        mds = MDS(n_components=min(10, len(dist_matrix) - 1),
                   dissimilarity="precomputed", random_state=42, normalized_stress="auto")
        embedding = mds.fit_transform(dist_matrix)
        from scipy.cluster.hierarchy import linkage as _linkage
        return _linkage(embedding, method="ward")
    return linkage(condensed, method=method)


def hierarchical_clustering(
    dist_matrix: np.ndarray,
    n_clusters: int | None = None,
    method: str = "average",
    max_k: int = 10,
) -> tuple[np.ndarray, np.ndarray, int, dict]:
    """
    层次聚类，支持自动选 k。

    Parameters
    ----------
    n_clusters : 指定 cluster 数。None 时自动选择。
    max_k : 自动选 k 时的搜索上限

    Returns
    -------
    labels : (N,) 聚类标签
    linkage_matrix : scipy linkage matrix
    actual_k : 实际使用的 cluster 数
    auto_info : 自动选 k 的诊断信息（n_clusters=None 时有意义）
    """
    Z = _build_linkage(dist_matrix, method)

    auto_info = {}
    if n_clusters is None:
        n_clusters, auto_info = auto_select_k(Z, max_k=max_k)
        confident = auto_info.get("confident", True)
        tag = "confident" if confident else "LOW-CONFIDENCE"
        print(f"  [auto-k] selected k={n_clusters} ({tag}, "
              f"rel_gap={auto_info.get('best_rel_gap', '?')}, "
              f"ratio={auto_info.get('confidence_ratio', '?')})")
        if "rel_gaps" in auto_info:
            print(f"    rel_gaps:  {auto_info['rel_gaps']}")
            print(f"    ranked_k:  {auto_info['ranked_k']}")
        if not confident:
            print(f"    [WARN] 最大 rel_gap 仅为第二大的 "
                  f"{auto_info.get('confidence_ratio', '?')}x，分界不明显")

    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    return labels, Z, n_clusters, auto_info


# ──────────────────────── 递归聚类 ────────────────────────

def recursive_clustering(
    dist_matrix: np.ndarray,
    episode_ids: list[int],
    method: str = "average",
    max_k: int = 10,
    min_cluster_size: int = 5,
    max_depth: int = 3,
    min_rel_gap: float = 0.3,
    max_leaf_clusters: int = 0,
    _depth: int = 0,
    _indent: str = "",
    _leaf_counter: list[int] | None = None,
) -> dict:
    """
    递归聚类：先 auto-k 分大簇，再对每个簇内部继续 auto-k。

    解决 "大尺度差异淹没小尺度差异" 的问题：
    第一层把左手/右手分开，第二层在同一只手的 episode 中继续
    按夹爪/运动路径等细粒度特征划分。

    停止条件（满足任一即停）:
    - 簇内样本数 < min_cluster_size
    - 递归深度达到 max_depth
    - auto-k 的 best_rel_gap < min_rel_gap（子层级分界不显著）
    - 全局叶子簇数达到 max_leaf_clusters（0=不限制）

    Returns
    -------
    tree : 嵌套 dict，包含 sub_clusters（非叶节点）或 leaf=True
    """
    if _leaf_counter is None:
        _leaf_counter = [0]

    N = len(episode_ids)
    node = {
        "count": N,
        "episodes": [int(e) for e in episode_ids],
    }

    if N < min_cluster_size:
        node["leaf"] = True
        node["stop_reason"] = f"too_few ({N} < {min_cluster_size})"
        _leaf_counter[0] += 1
        print(f"{_indent}  leaf: {N} episodes (too few)")
        return node

    if _depth >= max_depth:
        node["leaf"] = True
        node["stop_reason"] = f"max_depth ({max_depth})"
        _leaf_counter[0] += 1
        print(f"{_indent}  leaf: {N} episodes (max depth)")
        return node

    if max_leaf_clusters > 0 and _leaf_counter[0] >= max_leaf_clusters:
        node["leaf"] = True
        node["stop_reason"] = f"max_leaf_clusters ({max_leaf_clusters})"
        _leaf_counter[0] += 1
        print(f"{_indent}  leaf: {N} episodes (max leaf clusters reached)")
        return node

    Z = _build_linkage(dist_matrix, method)
    k, auto_info = auto_select_k(Z, max_k=min(max_k, N - 1))
    best_rel_gap = auto_info.get("best_rel_gap", 0)

    if _depth > 0 and best_rel_gap < min_rel_gap:
        node["leaf"] = True
        node["auto_k_info"] = auto_info
        node["stop_reason"] = (
            f"rel_gap too small ({best_rel_gap:.3f} < {min_rel_gap})"
        )
        _leaf_counter[0] += 1
        print(f"{_indent}  leaf: {N} episodes "
              f"(rel_gap={best_rel_gap:.3f} < {min_rel_gap})")
        return node

    if max_leaf_clusters > 0:
        budget = max_leaf_clusters - _leaf_counter[0]
        if budget < 2:
            node["leaf"] = True
            node["stop_reason"] = f"leaf_budget exhausted ({budget} < 2)"
            _leaf_counter[0] += 1
            print(f"{_indent}  leaf: {N} episodes (leaf budget exhausted)")
            return node
        k = min(k, budget)

    labels = fcluster(Z, t=k, criterion="maxclust")
    node["leaf"] = False
    node["k"] = k
    node["auto_k_info"] = auto_info
    node["sub_clusters"] = {}

    print(f"{_indent}  split → {k} clusters "
          f"(rel_gap={best_rel_gap:.3f}, "
          f"conf={auto_info.get('confidence_ratio', '?')})")

    for cid in sorted(set(int(x) for x in labels)):
        mask = labels == cid
        indices = np.where(mask)[0]
        sub_eps = [episode_ids[i] for i in indices]
        sub_dist = dist_matrix[np.ix_(indices, indices)]

        label = str(cid)
        print(f"{_indent}  cluster {label}: {len(sub_eps)} episodes")

        node["sub_clusters"][label] = recursive_clustering(
            sub_dist, sub_eps,
            method=method, max_k=max_k,
            min_cluster_size=min_cluster_size,
            max_depth=max_depth,
            min_rel_gap=min_rel_gap,
            max_leaf_clusters=max_leaf_clusters,
            _depth=_depth + 1,
            _indent=_indent + "    ",
            _leaf_counter=_leaf_counter,
        )

    return node


def flatten_tree(tree: dict, prefix: str = "") -> dict[str, list[int]]:
    """
    将递归聚类树展平为 {leaf_path: [episode_ids]}。

    例: {"1.1": [0,3,5,...], "1.2": [10,14,...], "2": [1,2,4,...]}
    """
    if tree.get("leaf", True):
        key = prefix.rstrip(".") if prefix else "0"
        return {key: tree["episodes"]}

    result = {}
    for cid, subtree in tree.get("sub_clusters", {}).items():
        new_prefix = f"{prefix}{cid}."
        result.update(flatten_tree(subtree, new_prefix))
    return result


def tree_to_labels(
    tree: dict,
    episode_ids: list[int],
) -> tuple[np.ndarray, dict[str, int]]:
    """
    将递归聚类结果转换为与 episode_ids 对齐的 numeric label 数组。

    Returns
    -------
    labels : (N,) 数值标签
    path_to_label : {leaf_path: numeric_label}
    """
    flat = flatten_tree(tree)
    sorted_paths = sorted(flat.keys())
    path_to_label = {p: i for i, p in enumerate(sorted_paths)}

    ep_to_path = {}
    for path, episodes in flat.items():
        for ep in episodes:
            ep_to_path[ep] = path

    labels = np.array([path_to_label[ep_to_path[ep]] for ep in episode_ids])
    return labels, path_to_label


def kmedoids_clustering(
    dist_matrix: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    K-Medoids 聚类（直接基于距离矩阵，不需要坐标）。

    Returns
    -------
    labels : (N,) 聚类标签
    medoid_indices : (n_clusters,) 每个簇的 medoid 在原数组中的下标
    """
    try:
        from sklearn_extra.cluster import KMedoids
        km = KMedoids(n_clusters=n_clusters, metric="precomputed",
                       random_state=random_state, method="pam")
        km.fit(dist_matrix)
        return km.labels_, km.medoid_indices_
    except Exception:
        print("[WARN] sklearn_extra unavailable, falling back to simple k-medoids")
        return _simple_kmedoids(dist_matrix, n_clusters, random_state)


def _simple_kmedoids(
    dist_matrix: np.ndarray,
    n_clusters: int,
    random_state: int,
    max_iter: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """当 sklearn_extra 不可用时的简易 K-Medoids 实现。"""
    rng = np.random.RandomState(random_state)
    N = dist_matrix.shape[0]
    medoids = rng.choice(N, size=n_clusters, replace=False)

    for _ in range(max_iter):
        # assign
        dists_to_medoids = dist_matrix[:, medoids]
        labels = np.argmin(dists_to_medoids, axis=1)

        # update medoids
        new_medoids = np.empty(n_clusters, dtype=int)
        for k in range(n_clusters):
            members = np.where(labels == k)[0]
            if len(members) == 0:
                new_medoids[k] = medoids[k]
                continue
            sub = dist_matrix[np.ix_(members, members)]
            best = members[np.argmin(sub.sum(axis=1))]
            new_medoids[k] = best

        if np.array_equal(new_medoids, medoids):
            break
        medoids = new_medoids

    dists_to_medoids = dist_matrix[:, medoids]
    labels = np.argmin(dists_to_medoids, axis=1)
    return labels, medoids


# ──────────────────────── 可视化 ────────────────────────

def plot_distance_heatmap(
    dist_matrix: np.ndarray,
    episode_ids: list[int],
    labels: np.ndarray | None = None,
    output_path: str = "distance_heatmap.png",
):
    """绘制距离矩阵热力图，可选按聚类标签排序。"""
    if labels is not None:
        order = np.argsort(labels)
        sorted_mat = dist_matrix[np.ix_(order, order)]
        sorted_ids = [episode_ids[i] for i in order]
    else:
        sorted_mat = dist_matrix
        sorted_ids = episode_ids

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(sorted_mat, ax=ax, cmap="YlOrRd",
                xticklabels=sorted_ids, yticklabels=sorted_ids)
    ax.set_title("Trajectory DTW Distance Matrix")
    ax.set_xlabel("Episode ID")
    ax.set_ylabel("Episode ID")

    N = len(sorted_ids)
    tick_step = max(1, N // 20)
    ax.set_xticks(range(0, N, tick_step))
    ax.set_xticklabels([sorted_ids[i] for i in range(0, N, tick_step)], rotation=45, fontsize=7)
    ax.set_yticks(range(0, N, tick_step))
    ax.set_yticklabels([sorted_ids[i] for i in range(0, N, tick_step)], fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved heatmap -> {output_path}")


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    episode_ids: list[int],
    n_clusters: int,
    output_path: str = "dendrogram.png",
):
    """绘制层次聚类树状图。"""
    fig, ax = plt.subplots(figsize=(max(14, len(episode_ids) * 0.2), 6))
    dendrogram(
        linkage_matrix,
        labels=[str(e) for e in episode_ids],
        color_threshold=linkage_matrix[-(n_clusters - 1), 2] if n_clusters > 1 else 0,
        leaf_rotation=90,
        leaf_font_size=7,
        ax=ax,
    )
    ax.set_title(f"Hierarchical Clustering Dendrogram (k={n_clusters})")
    ax.set_xlabel("Episode ID")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved dendrogram -> {output_path}")


def plot_mds_embedding(
    dist_matrix: np.ndarray,
    episode_ids: list[int],
    labels: np.ndarray,
    output_path: str = "mds_embedding.png",
):
    """用 MDS 将距离矩阵降维到 2D 并按聚类上色。"""
    mds = MDS(n_components=2, dissimilarity="precomputed",
              random_state=42, normalized_stress="auto")
    coords = mds.fit_transform(dist_matrix)

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap("tab10", len(unique_labels))

    for k in unique_labels:
        mask = labels == k
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[cmap(k - min(unique_labels))], label=f"Cluster {k}",
                   s=60, alpha=0.8, edgecolors="w", linewidth=0.5)
        for idx in np.where(mask)[0]:
            ax.annotate(str(episode_ids[idx]),
                        (coords[idx, 0], coords[idx, 1]),
                        fontsize=6, alpha=0.7)

    ax.set_title("Trajectory MDS Embedding (2D)")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("MDS-1")
    ax.set_ylabel("MDS-2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved MDS plot -> {output_path}")


def print_cluster_summary(
    labels: np.ndarray,
    episode_ids: list[int],
    medoid_indices: np.ndarray | None = None,
):
    """打印每个聚类的轨迹数量和成员 episode ID。"""
    unique_labels = np.unique(labels)
    print(f"\n{'='*60}")
    print(f"  Clustering Summary: {len(unique_labels)} clusters, {len(labels)} trajectories")
    print(f"{'='*60}")

    for k in unique_labels:
        members = np.where(labels == k)[0]
        member_ids = [episode_ids[i] for i in members]
        medoid_str = ""
        if medoid_indices is not None:
            medoid_ep = episode_ids[medoid_indices[k if k < len(medoid_indices) else 0]]
            medoid_str = f"  medoid=ep_{medoid_ep}"
        print(f"\n  Cluster {k}: {len(members)} trajectories{medoid_str}")
        print(f"    Episodes: {member_ids}")

    print(f"\n{'='*60}\n")
