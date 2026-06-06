"""
Trajectory clustering and visualization based on DTW distance matrices.
Supports hierarchical clustering, K-Medoids, and multiple visualization methods.
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


# ──────────────────────── Auto-select k ────────────────────────

def auto_select_k(
    linkage_matrix: np.ndarray,
    max_k: int = 10,
    min_k: int = 2,
) -> tuple[int, dict]:
    """
    Automatically select the best k based on merge-height jumps in the dendrogram.

    Definitions
    -----------
    merge_height(k) = Z[N-k, 2]
        The distance required to merge k clusters into k-1.

    abs_gap(k)  = merge_height(k) - merge_height(k+1)
        "How much more expensive merging from k to k-1 is than from k+1 to k".
        Larger values indicate k is a good boundary.

    rel_gap(k)  = abs_gap(k) / merge_height(k+1)     (> 0)
        Normalized to merge scale, making datasets with different units comparable.

    k Selection Strategy
    --------------------
    1. Compute abs_gap and rel_gap for each k
    2. Sort by rel_gap, select k with the largest rel_gap
    3. Confidence = best_rel_gap / second_rel_gap (>= 2 considered high confidence)

    Returns
    -------
    best_k : recommended number of clusters
    info : diagnostic information
    """
    N = linkage_matrix.shape[0] + 1
    max_k = min(max_k, N - 1)
    if max_k < min_k:
        return min_k, {"note": "too few samples for auto-k"}

    merge_dists = linkage_matrix[:, 2]

    # merge_height(k) = cost of merging k clusters into k-1
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

    # Sort by rel_gap to select best_k
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


# ──────────────────────── Clustering Methods ────────────────────────

def _build_linkage(dist_matrix: np.ndarray, method: str) -> np.ndarray:
    """Build linkage matrix."""
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
    Hierarchical clustering with auto-k support.

    Parameters
    ----------
    n_clusters : number of clusters. None for auto-selection.
    max_k : upper bound for auto-k search

    Returns
    -------
    labels : (N,) cluster labels
    linkage_matrix : scipy linkage matrix
    actual_k : actual number of clusters used
    auto_info : auto-k diagnostic information (meaningful when n_clusters=None)
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
            print(f"    [WARN] Largest rel_gap is only "
                  f"{auto_info.get('confidence_ratio', '?')}x of second largest, boundary is not clear")

    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    return labels, Z, n_clusters, auto_info


# ──────────────────────── Recursive Clustering ────────────────────────

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
    Recursive clustering: first auto-k to split into large clusters, then continue auto-k within each cluster.

    Addresses the "large-scale differences drowning small-scale differences" problem:
    The first level separates left/right arm, the second level further divides
    episodes of the same arm by finer features like gripper state or motion path.

    Stop conditions (any one triggers stop):
    - Cluster size < min_cluster_size
    - Recursion depth reaches max_depth
    - Auto-k's best_rel_gap < min_rel_gap (sub-level boundary is not significant)
    - Global leaf cluster count reaches max_leaf_clusters (0=unlimited)

    Returns
    -------
    tree : nested dict containing sub_clusters (non-leaf nodes) or leaf=True
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
    Flatten the recursive clustering tree into {leaf_path: [episode_ids]}.

    Example: {"1.1": [0,3,5,...], "1.2": [10,14,...], "2": [1,2,4,...]}
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
    Convert recursive clustering results into a numeric label array aligned with episode_ids.

    Returns
    -------
    labels : (N,) numeric labels
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
    K-Medoids clustering (directly based on distance matrix, no coordinates needed).

    Returns
    -------
    labels : (N,) cluster labels
    medoid_indices : (n_clusters,) index of each cluster's medoid in the original array
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
    """Simple K-Medoids implementation when sklearn_extra is unavailable."""
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


# ──────────────────────── Visualization ────────────────────────

def plot_distance_heatmap(
    dist_matrix: np.ndarray,
    episode_ids: list[int],
    labels: np.ndarray | None = None,
    output_path: str = "distance_heatmap.png",
):
    """Plot distance matrix heatmap, optionally sorted by cluster labels."""
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
    """Plot hierarchical clustering dendrogram."""
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
    """Use MDS to reduce distance matrix to 2D and color by cluster labels."""
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
    """Print the number of trajectories and member episode IDs for each cluster."""
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
