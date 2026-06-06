#!/usr/bin/env python3
"""
Task-level VLA trajectory clustering analysis -- for large-scale flat datasets like BC_Z / RT-1.

These datasets have no physical subdirectories; all episodes are mixed under a single data/ folder,
with task information recorded in meta/episodes.jsonl.
This script uses task as the clustering unit: parse episodes.jsonl -> load corresponding parquets by task -> DTW + clustering.

Usage:
    # Process BC_Z (default recursive, max_leaf_clusters=20)
    python batch_run_by_task.py --datasets BC_Z --output_root ./results_by_task

    # Process RT-1
    python batch_run_by_task.py --datasets RT-1 --output_root ./results_by_task

    # Debug: only run 3 tasks
    python batch_run_by_task.py --datasets BC_Z --output_root ./results_by_task --max_tasks 3

    # Resume from checkpoint
    python batch_run_by_task.py --datasets BC_Z --output_root ./results_by_task --resume

    # Use flat clustering
    python batch_run_by_task.py --datasets BC_Z --output_root ./results_by_task --no_recursive --n_clusters 5

    # Process two datasets at once
    python batch_run_by_task.py --datasets BC_Z RT-1 --output_root ./results_by_task --resume

  python batch_run_by_task.py --datasets BC_Z --output_root ./results_by_task --no_recursive --n_clusters 20 --cluster_method both --linkage_method average --filter_report auto --resume
  python batch_run_by_task.py --datasets RT-1 --output_root ./results_by_task --no_recursive --n_clusters 10 --cluster_method both --linkage_method average --filter_report auto --resume

python batch_run_by_task.py \
      --datasets RH20T-RoboInter \
      --output_root ./results_by_task \
      --no_recursive \
      --n_clusters 10 \
      --cluster_method both \
      --linkage_method average \
      --filter_report auto \
      --resume 
    

  python batch_run_by_task.py \
        --datasets droid_RoboInter \
        --output_root ./results_by_task \
        --no_recursive \
        --n_clusters 10 \
        --cluster_method both \
        --linkage_method average \
        --filter_report auto \
        --resume
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from config import get_config, DatasetConfig
from utils.trajectory_loader import (
    discover_tasks,
    discover_sub_datasets,
    load_trajectories_by_indices,
    normalize_trajectories,
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
)


def load_filter_report(report_path: str) -> set[int]:
    """Extract problematic episode index set from *filter_report.json."""
    with open(report_path, encoding="utf-8") as f:
        data = json.load(f)
    problem = set()
    for key, val in data.items():
        if key == "summary":
            continue
        if isinstance(val, dict):
            for ep_idx_str in val.keys():
                try:
                    problem.add(int(ep_idx_str))
                except ValueError:
                    pass
    return problem


def find_filter_reports(dataset_root: str, recursive: bool = False) -> list[str]:
    """Find all files ending with filter_report.json under dataset_root.

    recursive=False: only search in dataset_root (non-recursive)
    recursive=True:  search in dataset_root and its one level of subdirectories
    """
    results = []
    for fname in os.listdir(dataset_root):
        fpath = os.path.join(dataset_root, fname)
        if fname.endswith("filter_report.json") and os.path.isfile(fpath):
            results.append(fpath)
        elif recursive and os.path.isdir(fpath):
            for sub_fname in os.listdir(fpath):
                if sub_fname.endswith("filter_report.json") and os.path.isfile(os.path.join(fpath, sub_fname)):
                    results.append(os.path.join(fpath, sub_fname))
    return sorted(results)


def sanitize_task_name(name: str, max_len: int = 120) -> str:
    """Convert a task name into a filesystem-safe directory name."""
    s = re.sub(r'[^\w\s\-]', '', name)
    s = re.sub(r'\s+', '_', s.strip())
    return s[:max_len] if s else "unknown_task"


def setup_logger(output_root: str) -> logging.Logger:
    os.makedirs(output_root, exist_ok=True)
    logger = logging.getLogger("batch_run_by_task")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(os.path.join(output_root, "batch_run_by_task.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def generate_task_per_episode(
    dataset_root: str,
    dataset_name: str,
    task_map: dict[str, list[int]],
    output_path: str,
):
    """Generate task_per_episode.json."""
    total_eps = sum(len(v) for v in task_map.values())
    result = {
        "dataset_name": dataset_name,
        "total_tasks": len(task_map),
        "total_episodes": total_eps,
        "tasks": {
            name: {"count": len(eps), "episode_indices": eps}
            for name, eps in sorted(task_map.items(), key=lambda x: -len(x[1]))
        },
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def process_task(
    dataset_root: str,
    cfg: DatasetConfig,
    task_name: str,
    episode_indices: list[int],
    chunk_size: int,
    side: str,
    output_dir: str,
    window: int | None,
    n_clusters: int,
    cluster_method: str,
    linkage_method: str,
    logger: logging.Logger,
    recursive: bool = True,
    max_depth: int = 3,
    min_cluster_size: int = 4,
    min_rel_gap: float = 0.3,
    max_leaf_clusters: int = 20,
) -> dict:
    """Process a single task: load its episodes and run clustering."""

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load trajectories
    trajectories = load_trajectories_by_indices(
        dataset_root, cfg, side, episode_indices, chunk_size,
    )

    if len(trajectories) < 2:
        logger.warning(f"    Only {len(trajectories)} trajectories loaded, skipping.")
        return {"status": "skipped", "reason": "too_few_trajectories", "n_traj": len(trajectories)}

    if cfg.scale_features:
        trajectories, norm_stats = normalize_trajectories(trajectories, cfg.rot_type)
        logger.info(f"    Feature normalization: {len(norm_stats)} dims → [0,1]")

    episode_ids = [t.episode_id for t in trajectories]
    lengths = [t.length for t in trajectories]
    N = len(trajectories)
    n_pairs = N * (N - 1) // 2
    logger.info(f"    {N} traj, dim={trajectories[0].dim}, "
                f"len: {min(lengths)}~{max(lengths)}, {n_pairs} pairs")

    # 2. DTW distance matrix
    t0 = time.time()
    two_stage = getattr(cfg, 'two_stage', False)
    dist_matrix = compute_distance_matrix(
        trajectories,
        w_pos=cfg.w_pos, w_rot=cfg.w_rot, w_grip=cfg.w_grip,
        normalize=cfg.normalize, window=window,
        n_jobs=cfg.n_jobs, rot_type=cfg.rot_type,
        two_stage=two_stage,
    )
    dtw_elapsed = time.time() - t0
    logger.info(f"    DTW done in {dtw_elapsed:.1f}s ({dtw_elapsed / max(n_pairs, 1) * 1000:.1f}ms/pair)")

    np.savez(
        os.path.join(output_dir, "distance_matrix.npz"),
        dist_matrix=dist_matrix, episode_ids=np.array(episode_ids),
    )

    upper = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    stats = {
        "min": float(upper.min()), "max": float(upper.max()),
        "mean": float(upper.mean()), "std": float(upper.std()),
    }

    dm_json = {
        "dataset_name": cfg.dataset_name,
        "dataset_root": dataset_root,
        "task": task_name,
        "side": side,
        "rot_type": cfg.rot_type,
        "episode_ids": episode_ids,
        "stats": {**stats, "num_trajectories": N},
        "params": {
            "w_pos": cfg.w_pos, "w_rot": cfg.w_rot, "w_grip": cfg.w_grip,
            "normalize": cfg.normalize, "window": window,
        },
    }
    with open(os.path.join(output_dir, "distance_matrix.json"), "w") as f:
        json.dump(dm_json, f, indent=2, ensure_ascii=False)

    # 3. Clustering
    result = {
        "dataset_name": cfg.dataset_name,
        "dataset_root": dataset_root,
        "task": task_name,
        "side": side,
        "rot_type": cfg.rot_type,
        "num_trajectories": N,
        "dtw_seconds": round(dtw_elapsed, 2),
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

    t_cluster = time.time()

    if recursive:
        logger.info(f"    Recursive clustering (max_depth={max_depth}, "
                     f"min_size={min_cluster_size}, min_rel_gap={min_rel_gap}, "
                     f"max_leaf={max_leaf_clusters})")
        tree = recursive_clustering(
            dist_matrix, episode_ids,
            method=linkage_method, max_k=10,
            min_cluster_size=min_cluster_size,
            max_depth=max_depth, min_rel_gap=min_rel_gap,
            max_leaf_clusters=max_leaf_clusters,
        )
        leaf_clusters = flatten_tree(tree)
        primary_labels, path_to_label = tree_to_labels(tree, episode_ids)
        n_leaf = len(leaf_clusters)

        result["mode"] = "recursive"
        result["n_leaf_clusters"] = n_leaf
        result["max_leaf_clusters_setting"] = max_leaf_clusters
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

        logger.info(f"    Recursive result: {n_leaf} leaf clusters "
                     f"({time.time() - t_cluster:.1f}s)")
        for p in sorted(leaf_clusters.keys()):
            logger.info(f"      [{p}] {len(leaf_clusters[p])} episodes")

        linkage_mat = None
        actual_k = n_leaf
    else:
        actual_k = min(n_clusters, N) if n_clusters > 0 else n_clusters

        hier_labels, linkage_mat = None, None
        kmed_labels, medoid_indices = None, None
        auto_info = {}

        if cluster_method in ("hierarchical", "both"):
            use_k = actual_k if n_clusters > 0 else None
            logger.info(f"    Hierarchical clustering ...")
            hier_labels, linkage_mat, actual_k, auto_info = hierarchical_clustering(
                dist_matrix, n_clusters=use_k, method=linkage_method,
            )
            logger.info(f"    Hierarchical done ({time.time() - t_cluster:.1f}s)")

        if cluster_method in ("kmedoids", "both"):
            t_kmed = time.time()
            logger.info(f"    K-Medoids clustering ...")
            kmed_labels, medoid_indices = kmedoids_clustering(
                dist_matrix, n_clusters=actual_k,
            )
            logger.info(f"    K-Medoids done ({time.time() - t_kmed:.1f}s)")

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

    with open(os.path.join(output_dir, "cluster_results.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # 4. Visualization
    t_viz = time.time()
    try:
        plot_distance_heatmap(
            dist_matrix, episode_ids, labels=primary_labels,
            output_path=os.path.join(output_dir, "distance_heatmap.png"),
        )
        if linkage_mat is not None:
            plot_dendrogram(
                linkage_mat, episode_ids, n_clusters=actual_k,
                output_path=os.path.join(output_dir, "dendrogram.png"),
            )
        if primary_labels is not None:
            plot_mds_embedding(
                dist_matrix, episode_ids, primary_labels,
                output_path=os.path.join(output_dir, "mds_embedding.png"),
            )
        logger.info(f"    Visualizations done ({time.time() - t_viz:.1f}s)")
    except Exception as e:
        logger.warning(f"    Visualization failed: {e}")

    total_time = time.time() - t0
    logger.info(f"    Total: {total_time:.1f}s")
    return {"status": "success", "n_traj": N, "n_clusters": actual_k, "dtw_seconds": round(dtw_elapsed, 2), "stats": stats}


def main():
    p = argparse.ArgumentParser(
        description="Task-level VLA trajectory clustering for BC_Z / RT-1",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--datasets", nargs="+", required=True,
                   help="Dataset names, e.g. BC_Z RT-1")
    p.add_argument("--output_root", type=str, default="./results_by_task",
                   help="Output root directory (default: ./results_by_task)")
    p.add_argument("--side", type=str, default=None,
                   help="Which arm side to analyze (default: use config)")
    p.add_argument("--resume", action="store_true",
                   help="Skip tasks that already have cluster_results.json")
    p.add_argument("--window", type=int, default=None,
                   help="Sakoe-Chiba window")

    # Task filtering
    p.add_argument("--max_tasks", type=int, default=None,
                   help="Max tasks to process per dataset (for debugging)")
    p.add_argument("--min_episodes_per_task", type=int, default=3,
                   help="Skip tasks with fewer episodes than this value (default: 3)")
    p.add_argument("--task_filter", type=str, default=None,
                   help="Only process tasks whose name contains this substring (for debugging)")

    # Clustering parameters
    p.add_argument("--no_recursive", action="store_true",
                   help="Disable recursive clustering, use flat mode")
    p.add_argument("--n_clusters", type=int, default=None,
                   help="Number of clusters in flat mode (0=auto, default: use config)")
    p.add_argument("--cluster_method", type=str, default="both",
                   choices=["hierarchical", "kmedoids", "both"])
    p.add_argument("--linkage_method", type=str, default="average",
                   choices=["average", "complete", "single", "ward"])

    # Recursive clustering parameters
    p.add_argument("--max_depth", type=int, default=3,
                   help="Maximum recursion depth (default: 3)")
    p.add_argument("--min_cluster_size", type=int, default=4,
                   help="Minimum cluster size to stop recursion (default: 4)")
    p.add_argument("--min_rel_gap", type=float, default=None,
                   help="Minimum rel_gap threshold for sub-level auto-k (default: use config)")
    p.add_argument("--max_leaf_clusters", type=int, default=20,
                   help="Max leaf clusters per task (default: 20)")

    # Episode filtering
    p.add_argument("--filter_report", type=str, default=None,
                   help="Path to filter_report.json to exclude problematic episodes. "
                        "Pass 'auto' to automatically search for *filter_report.json in dataset directory")

    args = p.parse_args()

    logger = setup_logger(args.output_root)
    logger.info("=" * 70)
    logger.info("Task-Level VLA Trajectory Clustering")
    logger.info("=" * 70)

    recursive = not args.no_recursive

    summary = {}
    total_t0 = time.time()

    def _process_one_root(
        dataset_root: str,
        cfg: DatasetConfig,
        side: str,
        output_prefix: str,
        sub_label: str = "",
    ) -> list[dict]:
        """Process all tasks under a single dataset root directory, returns ds_results list."""
        label = f"{cfg.dataset_name}/{sub_label}" if sub_label else cfg.dataset_name

        # Discover tasks
        logger.info(f"  [{label}] Discovering tasks from meta/episodes.jsonl ...")
        task_map, chunk_size = discover_tasks(dataset_root)
        logger.info(f"  [{label}] Found {len(task_map)} tasks, chunk_size={chunk_size}")

        # Load filter_report to exclude problematic episodes
        problem_episodes: set[int] = set()
        if args.filter_report:
            if args.filter_report == "auto":
                report_paths = find_filter_reports(dataset_root)
            else:
                report_paths = [args.filter_report]

            for rp in report_paths:
                probs = load_filter_report(rp)
                logger.info(f"  [{label}] Filter report {os.path.basename(rp)}: {len(probs)} problem episodes")
                problem_episodes |= probs

            if problem_episodes:
                total_before = sum(len(v) for v in task_map.values())
                for task_name in list(task_map.keys()):
                    task_map[task_name] = [ei for ei in task_map[task_name] if ei not in problem_episodes]
                    if not task_map[task_name]:
                        del task_map[task_name]
                total_after = sum(len(v) for v in task_map.values())
                logger.info(f"  [{label}] Filtered: {total_before} -> {total_after} episodes "
                             f"({total_before - total_after} removed), "
                             f"{len(task_map)} tasks remaining")

        # Generate task_per_episode.json
        tpe_path = os.path.join(output_prefix, "task_per_episode.json")
        os.makedirs(os.path.dirname(tpe_path), exist_ok=True)
        generate_task_per_episode(dataset_root, label, task_map, tpe_path)
        logger.info(f"  [{label}] Saved task_per_episode.json -> {tpe_path}")

        # Filter tasks
        tasks_to_process = []
        for task_name, ep_indices in sorted(task_map.items(), key=lambda x: -len(x[1])):
            if len(ep_indices) < args.min_episodes_per_task:
                continue
            if args.task_filter and args.task_filter.lower() not in task_name.lower():
                continue
            tasks_to_process.append((task_name, ep_indices))

        logger.info(f"  [{label}] Tasks to process: {len(tasks_to_process)} "
                     f"(filtered by min_episodes={args.min_episodes_per_task})")

        if args.max_tasks:
            tasks_to_process = tasks_to_process[:args.max_tasks]
            logger.info(f"  [{label}] Limiting to first {args.max_tasks} tasks")

        n_clusters = args.n_clusters if args.n_clusters is not None else cfg.n_clusters
        min_rel_gap = args.min_rel_gap if args.min_rel_gap is not None else cfg.min_rel_gap

        results = []
        for idx, (task_name, ep_indices) in enumerate(tasks_to_process):
            safe_name = sanitize_task_name(task_name)
            output_dir = os.path.join(output_prefix, safe_name)

            logger.info(f"\n  [{label}] [{idx + 1}/{len(tasks_to_process)}] \"{task_name}\" "
                         f"({len(ep_indices)} episodes)")

            if args.resume and os.path.exists(os.path.join(output_dir, "cluster_results.json")):
                logger.info(f"    SKIPPED (resume: results exist)")
                results.append({"task": task_name, "status": "skipped_resume"})
                continue

            try:
                res = process_task(
                    dataset_root=dataset_root,
                    cfg=cfg,
                    task_name=task_name,
                    episode_indices=ep_indices,
                    chunk_size=chunk_size,
                    side=side,
                    output_dir=output_dir,
                    window=args.window,
                    n_clusters=n_clusters,
                    cluster_method=args.cluster_method,
                    linkage_method=args.linkage_method,
                    logger=logger,
                    recursive=recursive,
                    max_depth=args.max_depth,
                    min_cluster_size=args.min_cluster_size,
                    min_rel_gap=min_rel_gap,
                    max_leaf_clusters=args.max_leaf_clusters,
                )
                res["task"] = task_name
                results.append(res)
                logger.info(f"    -> {res['status']}")
            except Exception as e:
                logger.error(f"    FAILED: {e}")
                logger.error(traceback.format_exc())
                results.append({"task": task_name, "status": "failed", "error": str(e)})

        return results

    for ds_name in args.datasets:
        cfg = get_config(ds_name)
        dataset_root = cfg.dataset_path

        logger.info(f"\n{'━' * 60}")
        logger.info(f"Dataset: {cfg.dataset_name} ({dataset_root})")
        logger.info(f"  rot_type={cfg.rot_type}, has_sub_datasets={cfg.has_sub_datasets}")

        side = args.side or cfg.side or cfg.available_sides[0]

        ds_results = []

        if cfg.has_sub_datasets:
            # Iterate sub-datasets, each one clusters by task independently
            sub_datasets = discover_sub_datasets(cfg)
            logger.info(f"  Found {len(sub_datasets)} sub-datasets")
            for sub_idx, sub_path in enumerate(sub_datasets):
                sub_name = os.path.relpath(sub_path, cfg.dataset_path)
                logger.info(f"\n  ── Sub-dataset [{sub_idx + 1}/{len(sub_datasets)}]: {sub_name}")
                output_prefix = os.path.join(args.output_root, cfg.dataset_name, sub_name)
                sub_results = _process_one_root(
                    dataset_root=sub_path,
                    cfg=cfg,
                    side=side,
                    output_prefix=output_prefix,
                    sub_label=sub_name,
                )
                for r in sub_results:
                    r["sub_dataset"] = sub_name
                ds_results.extend(sub_results)
        else:
            # Flat dataset (BC_Z, RT-1, etc.)
            output_prefix = os.path.join(args.output_root, cfg.dataset_name)
            ds_results = _process_one_root(
                dataset_root=dataset_root,
                cfg=cfg,
                side=side,
                output_prefix=output_prefix,
            )

        summary[cfg.dataset_name] = {
            "total_tasks_processed": len([r for r in ds_results if r.get("status") == "success"]),
            "tasks_skipped": len([r for r in ds_results if "skipped" in r.get("status", "")]),
            "tasks_failed": len([r for r in ds_results if r.get("status") == "failed"]),
            "details": ds_results,
        }

    total_elapsed = time.time() - total_t0
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Batch complete in {total_elapsed:.1f}s")
    logger.info(f"{'=' * 70}")

    summary_path = os.path.join(args.output_root, "batch_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary saved to {summary_path}")

    for ds_name, info in summary.items():
        logger.info(
            f"  {ds_name}: {info['total_tasks_processed']} success, "
            f"{info['tasks_skipped']} skipped, {info['tasks_failed']} failed"
        )


if __name__ == "__main__":
    main()
