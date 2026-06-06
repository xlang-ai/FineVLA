#!/usr/bin/env python3
"""
VLA Trajectory Action/State Similarity Analysis and Clustering -- Main Entry Point.

Supports two running modes:
  1. Single dataset mode: specify --dataset_root to analyze a single dataset
  2. Config mode: specify --dataset_name to load configuration from config.py

Usage examples:
    # Use config (recommended)
    python run_analysis.py --dataset_name Galaxea --side right

    # Single sub-dataset
    python run_analysis.py \
        --dataset_name RDT \
        --dataset_root $VLA_DATA_ROOT/RDT-yhq/airpods_on_second_layer \
        --side both --n_clusters 4 \
        --recursive --max_depth 2 --min_cluster_size 3 \
        --output_dir ./results_test

    # Load from cache
    python run_analysis.py \\
        --dataset_name Galaxea \\
        --dataset_root /path/to/dataset \\
        --load_cache ./results/distance_matrix.npz

    # Test run on multiple sub-datasets
    cd ClusteringAndSampling/



"""

import argparse
import json
import os
import time

import numpy as np

from config import DatasetConfig, get_config, list_configs
from utils.trajectory_loader import (
    load_trajectories, discover_sub_datasets, normalize_trajectories,
    load_filter_report, infer_robot_type_from_path,
)
from utils.dtw_distance import compute_distance_matrix
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


def parse_args():
    p = argparse.ArgumentParser(
        description="VLA trajectory similarity analysis via DTW + clustering",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # Data
    p.add_argument("--dataset_name", type=str, default=None,
                   help="Dataset name (loads config from config.py)")
    p.add_argument("--dataset_root", type=str, default=None,
                   help="Dataset root directory (overrides dataset_path in config)")
    p.add_argument("--side", type=str, default=None,
                   help="Which arm side to analyze (default: use the first one in config)")
    p.add_argument("--max_episodes", type=int, default=None,
                   help="Max number of trajectories to load (for quick debugging)")

    # DTW parameters (override config defaults)
    p.add_argument("--w_pos", type=float, default=None,
                   help="Position distance weight")
    p.add_argument("--w_rot", type=float, default=None,
                   help="Orientation distance weight")
    p.add_argument("--w_grip", type=float, default=None,
                   help="Gripper distance weight")
    p.add_argument("--normalize", action="store_true", default=None,
                   help="Normalize DTW distance by path length")
    p.add_argument("--no_normalize", action="store_true",
                   help="Disable DTW distance normalization")
    p.add_argument("--window", type=int, default=None,
                   help="Sakoe-Chiba window size (None=unconstrained)")
    p.add_argument("--n_jobs", type=int, default=None,
                   help="Number of parallel processes")

    # Clustering
    p.add_argument("--n_clusters", type=int, default=None,
                   help="Number of clusters (0=auto-select best k)")
    p.add_argument("--cluster_method", type=str, default="both",
                   choices=["hierarchical", "kmedoids", "both"],
                   help="Clustering method (default: both)")
    p.add_argument("--linkage_method", type=str, default="average",
                   choices=["average", "complete", "single", "ward"],
                   help="Hierarchical clustering linkage method (default: average)")

    # Recursive clustering
    p.add_argument("--recursive", action="store_true",
                   help="Enable recursive clustering (split into large clusters first, then subdivide)")
    p.add_argument("--max_depth", type=int, default=3,
                   help="Maximum recursion depth (default: 3)")
    p.add_argument("--min_cluster_size", type=int, default=5,
                   help="Minimum cluster size to stop recursion (default: 5)")
    p.add_argument("--min_rel_gap", type=float, default=None,
                   help="Minimum rel_gap threshold for sub-level auto-k (default: use config value)")

    # Output
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory for results (default: auto-generated)")
    p.add_argument("--load_cache", type=str, default=None,
                   help="Load pre-computed distance matrix from .npz cache file")

    # Data filtering
    p.add_argument("--filter_report_path", type=str, default=None,
                   help="Path to filter_report.json to exclude problematic episodes")
    p.add_argument("--sub_dataset", type=str, default=None,
                   help="Sub-dataset name (for matching keys in filter_report)")

    return p.parse_args()


def _resolve_config(args) -> DatasetConfig:
    """Build/merge configuration based on command-line arguments."""
    if args.dataset_name:
        cfg = get_config(args.dataset_name)
    else:
        cfg = DatasetConfig(
            dataset_name="custom",
            dataset_path=args.dataset_root or ".",
            has_sub_datasets=False,
            rot_type="quaternion",
            available_sides=["right", "left", "single"],
        )

    if args.w_pos is not None:
        cfg.w_pos = args.w_pos
    if args.w_rot is not None:
        cfg.w_rot = args.w_rot
    if args.w_grip is not None:
        cfg.w_grip = args.w_grip
    if args.no_normalize:
        cfg.normalize = False
    elif args.normalize is not None:
        cfg.normalize = args.normalize
    if args.n_jobs is not None:
        cfg.n_jobs = args.n_jobs
    if args.n_clusters is not None:
        cfg.n_clusters = args.n_clusters

    return cfg


def _compute_single_side(dataset_root, cfg, side, args, exclude_episodes=None, robot_type=None):
    """Compute single-arm DTW distance matrix. Returns (dist_matrix, episode_ids, elapsed) or (None, ids, 0)."""
    trajectories = load_trajectories(
        dataset_root, config=cfg, side=side, max_episodes=args.max_episodes,
        exclude_episodes=exclude_episodes, robot_type=robot_type,
    )
    if len(trajectories) < 2:
        return None, [t.episode_id for t in trajectories], 0.0

    if cfg.scale_features:
        trajectories, norm_stats = normalize_trajectories(trajectories, cfg.rot_type)
        print(f"    [{side}] feature normalization applied: {len(norm_stats)} dims scaled to [0,1]")
        for dim_key, s in norm_stats.items():
            print(f"      {dim_key}: range [{s['min']} → {s['max']}] (span={s['range']})")

    episode_ids = [t.episode_id for t in trajectories]
    lengths = [t.length for t in trajectories]
    N = len(trajectories)
    n_pairs = N * (N - 1) // 2
    print(f"    [{side}] {N} traj, dim={trajectories[0].dim}, "
          f"len: {min(lengths)}~{max(lengths)}, {n_pairs} pairs")

    t0 = time.time()
    two_stage = getattr(cfg, 'two_stage', False)
    dist_matrix = compute_distance_matrix(
        trajectories,
        w_pos=cfg.w_pos, w_rot=cfg.w_rot, w_grip=cfg.w_grip,
        normalize=cfg.normalize, window=args.window,
        n_jobs=cfg.n_jobs, rot_type=cfg.rot_type,
        two_stage=two_stage,
    )
    elapsed = time.time() - t0
    print(f"    [{side}] Done in {elapsed:.1f}s ({elapsed/max(n_pairs,1)*1000:.1f}ms/pair)")
    return dist_matrix, episode_ids, elapsed


def run_single(
    dataset_root: str,
    cfg: DatasetConfig,
    side: str,
    output_dir: str,
    args,
):
    """Run the full analysis pipeline on a single (sub-)dataset."""
    os.makedirs(output_dir, exist_ok=True)

    # ── Load filter_report (if available) ──
    exclude_episodes = None
    if args.filter_report_path or cfg.filter_report_path:
        filter_path = args.filter_report_path or cfg.filter_report_path
        if os.path.exists(filter_path):
            print(f"\n[0/4] Loading filter_report from: {filter_path}")
            problem_map = load_filter_report(filter_path)

            # Match sub-dataset name
            if args.sub_dataset:
                sub_name = args.sub_dataset
            else:
                sub_name = os.path.basename(dataset_root.rstrip("/"))

            if sub_name in problem_map:
                exclude_episodes = problem_map[sub_name]
                print(f"  -> Excluded {len(exclude_episodes)} problematic episodes from filter_report")
            else:
                print(f"  -> Sub-dataset '{sub_name}' not found in filter_report")

    # Infer robot_type (for RoboMindV2.0 dynamic config loading)
    robot_type = None
    if cfg.has_modality_json and cfg.sub_dataset_depth >= 2:
        robot_type = infer_robot_type_from_path(dataset_root, cfg.dataset_path)
        print(f"  -> Inferred robot_type: {robot_type}")

    # ── 1 & 2. Load trajectories + DTW distance matrix ──
    print(f"\n[1/4] Loading trajectories from: {dataset_root}")
    print(f"  config: {cfg.dataset_name}, rot_type={cfg.rot_type}, side={side}")

    if args.load_cache:
        print(f"\n[2/4] Loading cached distance matrix: {args.load_cache}")
        data = np.load(args.load_cache)
        dist_matrix = data["dist_matrix"]
        episode_ids = data["episode_ids"].tolist()
        sides_computed = [side]
    elif side == "both":
        both_sides = [s for s in cfg.arms.keys() if s != "single"]
        if len(both_sides) < 2:
            both_sides = list(cfg.arms.keys())[:1]
        sides_computed = both_sides

        two_stage = getattr(cfg, 'two_stage', False)
        print(f"\n[2/4] Computing DTW for both arms: {sides_computed}")
        print(f"  weights: pos={cfg.w_pos}, rot={cfg.w_rot}, grip={cfg.w_grip}")
        print(f"  two_stage={two_stage}, normalize={cfg.normalize}, window={args.window}, n_jobs={cfg.n_jobs}")

        dist_matrix = None
        episode_ids = None
        for s in sides_computed:
            dm, eids, _ = _compute_single_side(dataset_root, cfg, s, args, exclude_episodes, robot_type)
            if dm is None:
                print(f"    [{s}] too few trajectories, skipping side")
                continue
            if dist_matrix is None:
                dist_matrix, episode_ids = dm, eids
            else:
                if eids != episode_ids:
                    common = sorted(set(episode_ids) & set(eids))
                    idx_a = [episode_ids.index(e) for e in common]
                    idx_b = [eids.index(e) for e in common]
                    dist_matrix = dist_matrix[np.ix_(idx_a, idx_a)] + dm[np.ix_(idx_b, idx_b)]
                    episode_ids = common
                else:
                    dist_matrix = dist_matrix + dm

        if dist_matrix is None or len(episode_ids) < 2:
            print(f"  -> Too few trajectories after merging sides, skipping.")
            return
    else:
        sides_computed = [side]
        two_stage = getattr(cfg, 'two_stage', False)
        print(f"\n[2/4] Computing DTW distance matrix")
        print(f"  weights: pos={cfg.w_pos}, rot={cfg.w_rot}, grip={cfg.w_grip}")
        print(f"  two_stage={two_stage}, normalize={cfg.normalize}, window={args.window}, n_jobs={cfg.n_jobs}")

        dist_matrix, episode_ids, _ = _compute_single_side(dataset_root, cfg, side, args, exclude_episodes, robot_type)
        if dist_matrix is None:
            print(f"  -> Only {len(episode_ids)} trajectories, skipping.")
            return

    N = len(episode_ids)

    if not args.load_cache:
        cache_path = os.path.join(output_dir, "distance_matrix.npz")
        np.savez(cache_path, dist_matrix=dist_matrix, episode_ids=np.array(episode_ids))
        print(f"  -> Cached (npz) to {cache_path}")

    # Distance statistics
    upper = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    print(f"  Distance stats ({N} traj): min={upper.min():.4f}, max={upper.max():.4f}, "
          f"mean={upper.mean():.4f}, std={upper.std():.4f}")

    # Save distance matrix JSON
    if not args.load_cache:
        dm_json = {
            "dataset_name": cfg.dataset_name,
            "dataset_root": dataset_root,
            "side": side, "sides_computed": sides_computed,
            "rot_type": cfg.rot_type,
            "episode_ids": episode_ids,
            "stats": {
                "num_trajectories": N,
                "min": float(upper.min()),
                "max": float(upper.max()),
                "mean": float(upper.mean()),
                "std": float(upper.std()),
            },
            "params": {
                "w_pos": cfg.w_pos,
                "w_rot": cfg.w_rot,
                "w_grip": cfg.w_grip,
                "normalize": cfg.normalize,
                "window": args.window,
            },
        }
        dm_json_path = os.path.join(output_dir, "distance_matrix.json")
        with open(dm_json_path, "w") as f:
            json.dump(dm_json, f, indent=2)
        print(f"  -> Saved (json) to {dm_json_path}")

    # ── 3. Clustering ──
    n_clusters = cfg.n_clusters
    if n_clusters > 0:
        n_clusters = min(n_clusters, N)
    use_k = n_clusters if n_clusters > 0 else None

    result = {
        "dataset_name": cfg.dataset_name,
        "dataset_root": dataset_root,
        "side": side, "sides_computed": sides_computed,
        "rot_type": cfg.rot_type,
        "num_trajectories": N,
        "distance_stats": {
            "min": float(upper.min()),
            "max": float(upper.max()),
            "mean": float(upper.mean()),
            "std": float(upper.std()),
        },
    }

    def _build_cluster_detail(labels, ep_ids, medoids=None):
        clusters = {}
        for k in sorted(set(int(x) for x in labels)):
            members = [int(ep_ids[i]) for i in range(len(labels)) if int(labels[i]) == k]
            entry = {"count": len(members), "episodes": members}
            if medoids is not None:
                med_idx = int(medoids[k]) if k < len(medoids) else None
                if med_idx is not None:
                    entry["medoid_episode"] = int(ep_ids[med_idx])
            clusters[str(k)] = entry
        return clusters

    if getattr(args, "recursive", False):
        # ── Recursive clustering mode ──
        min_rel_gap = args.min_rel_gap if args.min_rel_gap is not None else cfg.min_rel_gap
        print(f"\n[3/4] Recursive clustering (linkage={args.linkage_method}, "
              f"max_depth={args.max_depth}, min_size={args.min_cluster_size}, "
              f"min_rel_gap={min_rel_gap})")
        tree = recursive_clustering(
            dist_matrix, episode_ids,
            method=args.linkage_method,
            max_k=10,
            min_cluster_size=args.min_cluster_size,
            max_depth=args.max_depth,
            min_rel_gap=min_rel_gap,
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

        print(f"\n  Recursive result: {n_leaf} leaf clusters")
        for p in sorted(leaf_clusters.keys()):
            print(f"    [{p}] {len(leaf_clusters[p])} episodes: "
                  f"{leaf_clusters[p][:10]}{'...' if len(leaf_clusters[p]) > 10 else ''}")

        # Leaf labels for visualization
        linkage_mat = None
        hier_labels = None
    else:
        # ── Original flat clustering mode ──
        print(f"\n[3/4] Clustering (k={'auto' if use_k is None else use_k})")

        hier_labels, linkage_mat = None, None
        kmed_labels, medoid_indices = None, None
        auto_info = {}

        if args.cluster_method in ("hierarchical", "both"):
            print(f"  Hierarchical clustering (linkage={args.linkage_method})...")
            hier_labels, linkage_mat, n_clusters, auto_info = hierarchical_clustering(
                dist_matrix, n_clusters=use_k, method=args.linkage_method,
            )
            print_cluster_summary(hier_labels, episode_ids)

        if args.cluster_method in ("kmedoids", "both"):
            print(f"  K-Medoids clustering (k={n_clusters})...")
            kmed_labels, medoid_indices = kmedoids_clustering(
                dist_matrix, n_clusters=n_clusters,
            )
            print_cluster_summary(kmed_labels, episode_ids, medoid_indices)

        primary_labels = kmed_labels if kmed_labels is not None else hier_labels

        result["mode"] = "flat"
        result["n_clusters"] = n_clusters

        if hier_labels is not None:
            result["hierarchical"] = {
                "method": args.linkage_method,
                "clusters": _build_cluster_detail(hier_labels, episode_ids),
            }
            rng = np.random.RandomState(42)
            representation = {}
            for k in sorted(set(int(x) for x in hier_labels)):
                members = [int(episode_ids[i]) for i in range(len(hier_labels))
                           if int(hier_labels[i]) == k]
                chosen = int(rng.choice(members))
                representation[str(k)] = f"{dataset_root}/{chosen}"
            result["cluster_representation"] = representation

        if auto_info:
            result["auto_k_info"] = auto_info

        if kmed_labels is not None:
            result["kmedoids"] = {
                "clusters": _build_cluster_detail(kmed_labels, episode_ids, medoid_indices),
            }

    # ── 4. Visualization ──
    print(f"\n[4/4] Generating visualizations -> {output_dir}")

    plot_distance_heatmap(
        dist_matrix, episode_ids, labels=primary_labels,
        output_path=os.path.join(output_dir, "distance_heatmap.png"),
    )

    if linkage_mat is not None:
        plot_dendrogram(
            linkage_mat, episode_ids, n_clusters=n_clusters,
            output_path=os.path.join(output_dir, "dendrogram.png"),
        )

    if primary_labels is not None:
        plot_mds_embedding(
            dist_matrix, episode_ids, primary_labels,
            output_path=os.path.join(output_dir, "mds_embedding.png"),
        )

    labels_path = os.path.join(output_dir, "cluster_results.json")
    with open(labels_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  -> Saved cluster results to {labels_path}")

    print("\nDone!")


def main():
    args = parse_args()
    cfg = _resolve_config(args)

    side = args.side or cfg.available_sides[0]

    dataset_root = args.dataset_root or cfg.dataset_path

    if args.output_dir:
        output_dir = args.output_dir
    else:
        sub_name = os.path.basename(dataset_root.rstrip("/"))
        output_dir = os.path.join(
            "results", cfg.dataset_name, sub_name,
        )

    run_single(dataset_root, cfg, side, output_dir, args)


if __name__ == "__main__":
    main()
