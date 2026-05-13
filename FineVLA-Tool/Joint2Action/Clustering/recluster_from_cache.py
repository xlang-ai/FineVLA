#!/usr/bin/env python3
"""
从已缓存的 distance_matrix.npz 重新聚类（不重算 DTW）。

用法:
    # 对 RDT 所有子数据集重新聚类，固定 4 簇
    python recluster_from_cache.py \
        --results_root ./results_two_stage \
        --dataset RDT \
        --n_clusters 4

    # 递归聚类模式
    python recluster_from_cache.py \
        --results_root ./results_two_stage \
        --dataset RDT \
        --recursive --max_depth 2 --min_cluster_size 3

    # 只处理指定子数据集
    python recluster_from_cache.py \
        --results_root ./results_two_stage \
        --dataset RDT \
        --sub_datasets airpods_on_second_layer_both arrange_word_2024_both \
        --n_clusters 4
"""

import argparse
import json
import os
import sys

import numpy as np

from utils.clustering_analysis import (
    hierarchical_clustering,
    kmedoids_clustering,
    recursive_clustering,
    flatten_tree,
    tree_to_labels,
    plot_distance_heatmap,
    plot_dendrogram,
    plot_mds_embedding,
    print_cluster_summary,
)


def recluster_one(
    output_dir: str,
    n_clusters: int,
    cluster_method: str,
    linkage_method: str,
    recursive: bool,
    max_depth: int,
    min_cluster_size: int,
    min_rel_gap: float,
):
    """从 output_dir 中的 distance_matrix.npz 重新聚类。"""
    npz_path = os.path.join(output_dir, "distance_matrix.npz")
    if not os.path.isfile(npz_path):
        print(f"  [SKIP] no distance_matrix.npz in {output_dir}")
        return

    # 读取已有的距离矩阵元信息
    dm_json_path = os.path.join(output_dir, "distance_matrix.json")
    dm_meta = {}
    if os.path.isfile(dm_json_path):
        with open(dm_json_path) as f:
            dm_meta = json.load(f)

    data = np.load(npz_path)
    dist_matrix = data["dist_matrix"]
    episode_ids = data["episode_ids"].tolist()
    N = len(episode_ids)

    print(f"  Loaded {N}x{N} distance matrix, episodes: {episode_ids[:5]}...")

    if N < 2:
        print(f"  [SKIP] only {N} trajectories")
        return

    upper = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    stats = {
        "min": float(upper.min()), "max": float(upper.max()),
        "mean": float(upper.mean()), "std": float(upper.std()),
    }

    dataset_root = dm_meta.get("dataset_root", "")
    dataset_name = dm_meta.get("dataset_name", "")
    side = dm_meta.get("side", "")
    sides_computed = dm_meta.get("sides_computed", [])

    result = {
        "dataset_name": dataset_name,
        "dataset_root": dataset_root,
        "side": side,
        "sides_computed": sides_computed,
        "rot_type": dm_meta.get("rot_type", ""),
        "num_trajectories": N,
        "distance_stats": stats,
    }

    def _build_detail(labels, ep_ids, medoids=None):
        clusters = {}
        for k in sorted(set(int(x) for x in labels)):
            members = [int(ep_ids[i]) for i, lb in enumerate(labels) if int(lb) == k]
            entry = {"count": len(members), "episodes": members}
            if medoids is not None and k < len(medoids):
                entry["medoid_episode"] = int(ep_ids[int(medoids[k])])
            clusters[str(k)] = entry
        return clusters

    if recursive:
        print(f"  Recursive clustering (max_depth={max_depth}, min_size={min_cluster_size})")
        tree = recursive_clustering(
            dist_matrix, episode_ids,
            method=linkage_method, max_k=10,
            min_cluster_size=min_cluster_size,
            max_depth=max_depth, min_rel_gap=min_rel_gap,
        )
        leaf_clusters = flatten_tree(tree)
        primary_labels, path_to_label = tree_to_labels(tree, episode_ids)
        n_leaf = len(leaf_clusters)

        result["mode"] = "recursive"
        result["n_leaf_clusters"] = n_leaf
        result["recursive_tree"] = tree
        result["leaf_clusters"] = {
            p: {"count": len(eps), "episodes": eps}
            for p, eps in sorted(leaf_clusters.items())
        }

        rng = np.random.RandomState(42)
        result["cluster_representation"] = {
            p: f"{dataset_root}/{int(rng.choice(eps))}"
            for p, eps in sorted(leaf_clusters.items())
        }

        ep_to_cluster = {}
        for p, eps in leaf_clusters.items():
            for ep in eps:
                ep_to_cluster[str(ep)] = p
        result["episode_to_cluster"] = ep_to_cluster

        print(f"  -> {n_leaf} leaf clusters")
        for p in sorted(leaf_clusters.keys()):
            print(f"    [{p}] {len(leaf_clusters[p])} episodes")

        linkage_mat = None
        actual_k = n_leaf
    else:
        use_k = min(n_clusters, N) if n_clusters > 0 else None
        print(f"  Flat clustering (k={'auto' if use_k is None else use_k})")

        hier_labels, linkage_mat = None, None
        kmed_labels, medoid_indices = None, None
        auto_info = {}

        if cluster_method in ("hierarchical", "both"):
            hier_labels, linkage_mat, actual_k, auto_info = hierarchical_clustering(
                dist_matrix, n_clusters=use_k, method=linkage_method,
            )
            print_cluster_summary(hier_labels, episode_ids)

        if cluster_method in ("kmedoids", "both"):
            actual_k_for_kmed = actual_k if hier_labels is not None else (use_k or 5)
            kmed_labels, medoid_indices = kmedoids_clustering(
                dist_matrix, n_clusters=actual_k_for_kmed,
            )
            print_cluster_summary(kmed_labels, episode_ids, medoid_indices)

        primary_labels = kmed_labels if kmed_labels is not None else hier_labels

        result["mode"] = "flat"
        result["n_clusters"] = actual_k

        if hier_labels is not None:
            result["hierarchical"] = {
                "method": linkage_method,
                "clusters": _build_detail(hier_labels, episode_ids),
            }
            rng = np.random.RandomState(42)
            representation = {}
            for k in sorted(set(int(x) for x in hier_labels)):
                members = [int(episode_ids[i]) for i, lb in enumerate(hier_labels) if int(lb) == k]
                chosen = int(rng.choice(members))
                representation[str(k)] = f"{dataset_root}/{chosen}"
            result["cluster_representation"] = representation

        if auto_info:
            result["auto_k_info"] = auto_info
        if kmed_labels is not None:
            result["kmedoids"] = {
                "clusters": _build_detail(kmed_labels, episode_ids, medoid_indices),
            }

    # 保存结果
    with open(os.path.join(output_dir, "cluster_results.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # 可视化
    try:
        plot_distance_heatmap(
            dist_matrix, episode_ids, labels=primary_labels,
            output_path=os.path.join(output_dir, "distance_heatmap.png"),
        )
        if not recursive and linkage_mat is not None:
            plot_dendrogram(
                linkage_mat, episode_ids, n_clusters=actual_k,
                output_path=os.path.join(output_dir, "dendrogram.png"),
            )
        if primary_labels is not None:
            plot_mds_embedding(
                dist_matrix, episode_ids, primary_labels,
                output_path=os.path.join(output_dir, "mds_embedding.png"),
            )
    except Exception as e:
        print(f"  [WARN] Visualization failed: {e}")

    print(f"  -> Saved to {output_dir}")


def main():
    p = argparse.ArgumentParser(description="Re-cluster from cached distance matrices")
    p.add_argument("--results_root", type=str, default="./results_two_stage")
    p.add_argument("--dataset", type=str, required=True, help="数据集名称，如 RDT")
    p.add_argument("--sub_datasets", nargs="+", default=None,
                   help="指定子数据集文件夹名（默认处理全部）")

    p.add_argument("--n_clusters", type=int, default=4, help="聚类数（0=自动）")
    p.add_argument("--cluster_method", type=str, default="both",
                   choices=["hierarchical", "kmedoids", "both"])
    p.add_argument("--linkage_method", type=str, default="average",
                   choices=["average", "complete", "single", "ward"])

    p.add_argument("--recursive", action="store_true", help="启用递归聚类")
    p.add_argument("--max_depth", type=int, default=2)
    p.add_argument("--min_cluster_size", type=int, default=3)
    p.add_argument("--min_rel_gap", type=float, default=0.3)

    args = p.parse_args()

    dataset_dir = os.path.join(args.results_root, args.dataset)
    if not os.path.isdir(dataset_dir):
        print(f"ERROR: {dataset_dir} not found")
        sys.exit(1)

    if args.sub_datasets:
        sub_dirs = args.sub_datasets
    else:
        sub_dirs = sorted(
            d for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d))
        )

    print(f"Re-clustering {args.dataset}: {len(sub_dirs)} sub-datasets")
    print(f"  mode={'recursive' if args.recursive else 'flat'}, "
          f"n_clusters={args.n_clusters}\n")

    for idx, sub_name in enumerate(sub_dirs):
        output_dir = os.path.join(dataset_dir, sub_name)
        print(f"[{idx+1}/{len(sub_dirs)}] {sub_name}")
        try:
            recluster_one(
                output_dir=output_dir,
                n_clusters=args.n_clusters,
                cluster_method=args.cluster_method,
                linkage_method=args.linkage_method,
                recursive=args.recursive,
                max_depth=args.max_depth,
                min_cluster_size=args.min_cluster_size,
                min_rel_gap=args.min_rel_gap,
            )
        except Exception as e:
            print(f"  [FAILED] {e}")

    print(f"\nDone! {len(sub_dirs)} sub-datasets processed.")


if __name__ == "__main__":
    main()
