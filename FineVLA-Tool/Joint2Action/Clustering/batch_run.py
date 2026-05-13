#!/usr/bin/env python3
"""
批量运行 VLA 轨迹聚类分析。

遍历 config.py 中注册的所有数据集（或指定子集），
自动发现子数据集，逐个执行 DTW + 聚类流程。

用法:
    # 运行所有数据集
    python batch_run.py --output_root ./results

    # 只运行指定数据集
    python batch_run.py --datasets Galaxea RDT --output_root ./results

    # 断点续跑（跳过已有结果的子数据集）
    python batch_run.py --resume --output_root ./results

    # 指定手臂侧
    python batch_run.py --datasets Galaxea --side right --output_root ./results

    # 只处理每个数据集的前 N 个子数据集（调试用）
    python batch_run.py --max_sub_datasets 3 --output_root ./results


    # 测试跑4个datasets
    python batch_run.py \
    --datasets Galaxea \
    --side both \
    --output_root ./resultsGalaxea-10-both-normalize \
    --max_sub_datasets 2

    #测试整个数据集
    python batch_run.py \
    --datasets Galaxea \
    --side both \
    --output_root ./resultsRDT-10-both-normalize \
    --max_sub_datasets 2


    # 测试RDT数据集
    python batch_run.py \
        --datasets RDT \
        --side both \
        --resume \
        --recursive \
        --max_depth 2 \
        --min_cluster_size 3 \
        --output_root ./results_two_stage
    
    #测试
    python batch_run.py \
        --datasets RDT \
        --side both \
        --resume \
        --recursive \
        --max_depth 2 \
        --min_cluster_size 3 \
        --output_root ./results_two_stage
    
    python batch_run.py \
        --datasets RoboMindV1 \
        --side both \
        --resume \
        --recursive \
        --max_depth 2 \
        --min_cluster_size 3 \
        --output_root ./results_two_stage


    python batch_run.py \
          --datasets RoboCOIN \
            --side both \
            --output_root ./results_two_stage \
            --n_clusters 15 \
            --cluster_method both \
            --linkage_method average \
            --filter_report auto \
            --resume

    python batch_run.py \
        --datasets RoboCOIN_add1201 \
        --side both \
        --output_root ./results_two_stage \
        --auto_n_clusters_divisor 10 \
        --linkage_method average \
        --filter_report auto \
        --resume

    # RoboMindV1, 每一个task 重复度非常高。每个task 分成10个cluster
    python batch_run.py \
      --datasets RoboMindV1.0 \
      --side both \
      --output_root ./results_two_stage \
      --n_clusters 10 \
      --cluster_method both \
      --linkage_method average \
      --filter_report auto \
      --resume
      
    # RoboMindV2, 每一个task 重复度非常高。每个task 分成10个cluster
    python batch_run.py \
      --datasets RoboMindV2.0 \
      --side both \
      --output_root ./results_two_stage \
      --n_clusters 10 \
      --cluster_method both \
      --linkage_method average \
      --filter_report auto \
      --resume

    # RH20T ，重复性依旧很高，所以每个cluster 10个
    python batch_run.py \
        --datasets RH20T \
        --side both \
        --output_root ./results_two_stage \
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
import sys
import time
import traceback

from config import get_config, list_configs, DatasetConfig
from utils.trajectory_loader import (
    load_trajectories, discover_sub_datasets, normalize_trajectories,
    load_filter_report as load_filter_report_v2,
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


import numpy as np


def load_filter_report(report_path: str) -> set[int]:
    """从 *filter_report.json 中提取有问题的 episode index 集合（平面格式）。"""
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
    """在 dataset_root 下查找所有以 filter_report.json 结尾的文件。

    recursive=False: 只在 dataset_root 下查找（非递归）
    recursive=True:  在 dataset_root 及其一层子目录下查找
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


def setup_logger(output_root: str) -> logging.Logger:
    os.makedirs(output_root, exist_ok=True)
    logger = logging.getLogger("batch_run")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(os.path.join(output_root, "batch_run.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def _compute_single_side_distance(
    dataset_root: str,
    cfg: DatasetConfig,
    side: str,
    window: int | None,
    max_episodes: int | None,
    logger: logging.Logger,
    exclude_episodes: set[int] | None = None,
    robot_type: str | None = None,
) -> tuple[np.ndarray | None, list[int], float]:
    """计算单侧手臂的 DTW 距离矩阵，返回 (dist_matrix, episode_ids, elapsed)。"""
    trajectories = load_trajectories(
        dataset_root, config=cfg, side=side, max_episodes=max_episodes,
        exclude_episodes=exclude_episodes, robot_type=robot_type,
    )
    if len(trajectories) < 2:
        return None, [t.episode_id for t in trajectories], 0.0

    if cfg.scale_features:
        trajectories, norm_stats = normalize_trajectories(trajectories, cfg.rot_type)
        logger.info(f"    [{side}] feature normalization: {len(norm_stats)} dims → [0,1]")
        for dim_key, s in norm_stats.items():
            logger.info(f"      {dim_key}: [{s['min']} → {s['max']}] span={s['range']}")

    episode_ids = [t.episode_id for t in trajectories]
    lengths = [t.length for t in trajectories]
    N = len(trajectories)
    n_pairs = N * (N - 1) // 2
    logger.info(f"    [{side}] {N} traj, dim={trajectories[0].dim}, "
                f"len: {min(lengths)}~{max(lengths)}, {n_pairs} pairs")

    two_stage = getattr(cfg, 'two_stage', False)
    t0 = time.time()
    dist_matrix = compute_distance_matrix(
        trajectories,
        w_pos=cfg.w_pos, w_rot=cfg.w_rot, w_grip=cfg.w_grip,
        normalize=cfg.normalize, window=window,
        n_jobs=cfg.n_jobs, rot_type=cfg.rot_type,
        two_stage=two_stage,
    )
    elapsed = time.time() - t0
    logger.info(f"    [{side}] DTW done in {elapsed:.1f}s ({elapsed/max(n_pairs,1)*1000:.1f}ms/pair)")
    return dist_matrix, episode_ids, elapsed


def process_sub_dataset(
    dataset_root: str,
    cfg: DatasetConfig,
    side: str,
    output_dir: str,
    window: int | None,
    n_clusters: int,
    cluster_method: str,
    linkage_method: str,
    max_episodes: int | None,
    logger: logging.Logger,
    recursive: bool = False,
    max_depth: int = 3,
    min_cluster_size: int = 4,
    min_rel_gap: float = 0.3,
    exclude_episodes: set[int] | None = None,
    robot_type: str | None = None,
) -> dict:
    """处理单个子数据集，返回结果摘要。"""

    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载 & 计算距离矩阵
    sides_to_run = ["left", "right"] if side == "both" else [side]
    side_label = side

    if side == "both":
        both_sides = list(cfg.arms.keys())
        both_sides = [s for s in both_sides if s != "single"]
        if len(both_sides) < 2:
            logger.warning(f"  Config only has arms: {list(cfg.arms.keys())}, falling back to first")
            sides_to_run = [list(cfg.arms.keys())[0]]
            side_label = sides_to_run[0]
        else:
            sides_to_run = both_sides[:2]

    logger.info(f"  Sides: {sides_to_run}")

    dist_matrix = None
    episode_ids = None
    total_elapsed = 0.0

    for s in sides_to_run:
        dm, eids, elapsed = _compute_single_side_distance(
            dataset_root, cfg, s, window, max_episodes, logger,
            exclude_episodes=exclude_episodes, robot_type=robot_type,
        )
        total_elapsed += elapsed
        if dm is None:
            logger.warning(f"  [{s}] too few trajectories, skipping this side.")
            continue

        if dist_matrix is None:
            dist_matrix = dm
            episode_ids = eids
        else:
            if eids != episode_ids:
                logger.warning(f"  Episode IDs mismatch between sides, using intersection")
                common = sorted(set(episode_ids) & set(eids))
                idx_a = [episode_ids.index(e) for e in common]
                idx_b = [eids.index(e) for e in common]
                dist_matrix = dist_matrix[np.ix_(idx_a, idx_a)] + dm[np.ix_(idx_b, idx_b)]
                episode_ids = common
            else:
                dist_matrix = dist_matrix + dm

    if dist_matrix is None or (episode_ids is not None and len(episode_ids) < 2):
        # 尝试 fallback 到其他 side（处理异构数据集如 RoboMindV1）
        fallback_sides = [s for s in cfg.available_sides if s not in sides_to_run and s in cfg.arms]
        for fb_side in fallback_sides:
            fb_sides = ["left", "right"] if fb_side == "both" else [fb_side]
            if fb_side == "both":
                fb_arms = [s for s in cfg.arms.keys() if s != "single"]
                fb_sides = fb_arms[:2] if len(fb_arms) >= 2 else fb_arms
            logger.info(f"  Primary side(s) {sides_to_run} yielded no data, trying fallback: {fb_sides}")
            for s in fb_sides:
                if s not in cfg.arms:
                    continue
                dm, eids, elapsed = _compute_single_side_distance(
                    dataset_root, cfg, s, window, max_episodes, logger,
                    exclude_episodes=exclude_episodes, robot_type=robot_type,
                )
                total_elapsed += elapsed
                if dm is None:
                    continue
                if dist_matrix is None:
                    dist_matrix = dm
                    episode_ids = eids
                    sides_to_run = fb_sides
                    side_label = fb_side
                else:
                    if eids == episode_ids:
                        dist_matrix = dist_matrix + dm
                    else:
                        common = sorted(set(episode_ids) & set(eids))
                        idx_a = [episode_ids.index(e) for e in common]
                        idx_b = [eids.index(e) for e in common]
                        dist_matrix = dist_matrix[np.ix_(idx_a, idx_a)] + dm[np.ix_(idx_b, idx_b)]
                        episode_ids = common
            if dist_matrix is not None:
                break

    if dist_matrix is None or (episode_ids is not None and len(episode_ids) < 2):
        n_traj = len(episode_ids) if episode_ids else 0
        logger.warning(f"  Only {n_traj} trajectories after merging sides, skipping.")
        return {"status": "skipped", "reason": "too_few_trajectories", "n_traj": n_traj}

    N = len(episode_ids)
    n_pairs = N * (N - 1) // 2
    logger.info(f"  Combined distance matrix: {N}x{N}, elapsed={total_elapsed:.1f}s")

    # 保存距离矩阵
    logger.info(f"  Saving distance matrix ...")
    np.savez(
        os.path.join(output_dir, "distance_matrix.npz"),
        dist_matrix=dist_matrix, episode_ids=np.array(episode_ids),
    )

    upper = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    stats = {
        "min": float(upper.min()), "max": float(upper.max()),
        "mean": float(upper.mean()), "std": float(upper.std()),
    }
    logger.info(f"  Distance stats: {stats}")

    dm_json = {
        "dataset_name": cfg.dataset_name,
        "dataset_root": dataset_root,
        "side": side_label, "sides_computed": sides_to_run,
        "rot_type": cfg.rot_type,
        "episode_ids": episode_ids,
        "stats": {**stats, "num_trajectories": N},
        "params": {
            "w_pos": cfg.w_pos, "w_rot": cfg.w_rot, "w_grip": cfg.w_grip,
            "normalize": cfg.normalize, "window": window,
        },
    }
    with open(os.path.join(output_dir, "distance_matrix.json"), "w") as f:
        json.dump(dm_json, f, indent=2)
    logger.info(f"  Distance matrix saved.")

    # 3. 聚类
    elapsed = total_elapsed
    result = {
        "dataset_name": cfg.dataset_name,
        "dataset_root": dataset_root,
        "side": side_label, "sides_computed": sides_to_run,
        "rot_type": cfg.rot_type,
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

    t_cluster = time.time()
    if recursive:
        # ── 递归聚类 ──
        logger.info(f"  Recursive clustering (max_depth={max_depth}, "
                     f"min_size={min_cluster_size}, min_rel_gap={min_rel_gap})")
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

        logger.info(f"  Recursive result: {n_leaf} leaf clusters "
                     f"({time.time()-t_cluster:.1f}s)")
        for p in sorted(leaf_clusters.keys()):
            logger.info(f"    [{p}] {len(leaf_clusters[p])} episodes")

        linkage_mat = None
        actual_k = n_leaf
    else:
        # ── 原始单层聚类 ──
        actual_k = min(n_clusters, N) if n_clusters > 0 else n_clusters

        hier_labels, linkage_mat = None, None
        kmed_labels, medoid_indices = None, None
        auto_info = {}

        if cluster_method in ("hierarchical", "both"):
            use_k = actual_k if n_clusters > 0 else None
            logger.info(f"  Hierarchical clustering ...")
            hier_labels, linkage_mat, actual_k, auto_info = hierarchical_clustering(
                dist_matrix, n_clusters=use_k, method=linkage_method,
            )
            logger.info(f"  Hierarchical done ({time.time()-t_cluster:.1f}s)")
        if cluster_method in ("kmedoids", "both"):
            t_kmed = time.time()
            logger.info(f"  K-Medoids clustering ...")
            kmed_labels, medoid_indices = kmedoids_clustering(
                dist_matrix, n_clusters=actual_k,
            )
            logger.info(f"  K-Medoids done ({time.time()-t_kmed:.1f}s)")

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

    # 4. 可视化
    t_viz = time.time()
    logger.info(f"  Generating visualizations ...")
    try:
        plot_distance_heatmap(
            dist_matrix, episode_ids, labels=primary_labels,
            output_path=os.path.join(output_dir, "distance_heatmap.png"),
        )
        logger.info(f"  Heatmap done ({time.time()-t_viz:.1f}s)")
        if linkage_mat is not None:
            t_dend = time.time()
            plot_dendrogram(
                linkage_mat, episode_ids, n_clusters=actual_k,
                output_path=os.path.join(output_dir, "dendrogram.png"),
            )
            logger.info(f"  Dendrogram done ({time.time()-t_dend:.1f}s)")
        if primary_labels is not None:
            t_mds = time.time()
            plot_mds_embedding(
                dist_matrix, episode_ids, primary_labels,
                output_path=os.path.join(output_dir, "mds_embedding.png"),
            )
            logger.info(f"  MDS plot done ({time.time()-t_mds:.1f}s)")
    except Exception as e:
        logger.warning(f"  Visualization failed: {e}")

    logger.info(f"  Total post-DTW: {time.time()-t_cluster:.1f}s")
    return {"status": "success", "n_traj": N, "dtw_seconds": elapsed, "stats": stats}


def main():
    p = argparse.ArgumentParser(description="Batch VLA trajectory clustering")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="要处理的数据集名称（默认全部）") 
    p.add_argument("--output_root", type=str, default="./results",
                   help="输出根目录 (default: ./results)")
    p.add_argument("--side", type=str, default=None,
                   help="手臂侧: right/left/both（默认使用 config 中的第一个，both=左右臂距离加和）")
    p.add_argument("--resume", action="store_true",
                   help="跳过已有 distance_matrix.npz 的子数据集")
    p.add_argument("--window", type=int, default=None,
                   help="Sakoe-Chiba 窗口")
    p.add_argument("--n_clusters", type=int, default=None,
                   help="聚类数（0=自动选择最佳 k）")
    p.add_argument("--auto_n_clusters_divisor", type=int, default=None,
                   help="按子数据集 episode 数自动计算 n_clusters = total_episodes // divisor"
                        "（优先于 --n_clusters）")
    p.add_argument("--cluster_method", type=str, default="both",
                   choices=["hierarchical", "kmedoids", "both"])
    p.add_argument("--linkage_method", type=str, default="average",
                   choices=["average", "complete", "single", "ward"])
    p.add_argument("--max_sub_datasets", type=int, default=None,
                   help="每个数据集最多处理几个子数据集（调试用）")
    p.add_argument("--max_episodes", type=int, default=None,
                   help="每个子数据集最多加载几条轨迹")

    # 递归聚类
    p.add_argument("--recursive", action="store_true",
                   help="启用递归聚类（先分大簇，再在簇内继续划分）")
    p.add_argument("--max_depth", type=int, default=2,
                   help="递归最大深度 (default: 3)")
    p.add_argument("--min_cluster_size", type=int, default=5,
                   help="递归终止的最小簇大小 (default: 5)")
    p.add_argument("--min_rel_gap", type=float, default=None,
                   help="子层级 auto-k 的最小 rel_gap 阈值（默认使用 config 中的值）")

    # episode 过滤
    p.add_argument("--filter_report", type=str, default="auto",
                   help="filter_report.json 路径，排除问题 episode。"
                        "传 'auto' 则自动在数据集目录下查找 *filter_report.json")
    args = p.parse_args()

    logger = setup_logger(args.output_root)
    logger.info("=" * 70)
    logger.info("Batch VLA Trajectory Clustering")
    logger.info("=" * 70)

    if args.datasets:
        configs = [get_config(name) for name in args.datasets]
    else:
        configs = list_configs(include_skip=False)

    summary = {}
    total_t0 = time.time()

    for cfg in configs:
        logger.info(f"\n{'━' * 60}")
        logger.info(f"Dataset: {cfg.dataset_name} ({cfg.dataset_path})")
        logger.info(f"  rot_type={cfg.rot_type}, skip={cfg.skip}")

        if cfg.skip:
            logger.info("  SKIPPED (marked skip in config)")
            summary[cfg.dataset_name] = {"status": "skipped_config"}
            continue

        if args.side:
            side = args.side
        elif cfg.side:
            side = cfg.side
        elif len(cfg.available_sides) >= 2 and "single" not in cfg.available_sides:
            side = "both"
        else:
            side = cfg.available_sides[0]
        n_clusters = args.n_clusters or cfg.n_clusters
        max_depth = args.max_depth if args.max_depth != 2 else cfg.max_depth

        # 加载 filter_report，排除问题 episode
        # per_sub_filter=True 时每个子数据集各自查找 filter_report（适用于 RH20T 等）
        # RoboMindV2.0 使用结构化的 filter_report，需要按 subdataset 名称匹配
        exclude_episodes: set[int] | None = None
        per_sub_filter = False
        filter_report_map: dict[str, set[int]] = {}  # 用于 RoboMindV2.0 结构化 filter_report

        if args.filter_report:
            if args.filter_report == "auto":
                report_paths = find_filter_reports(cfg.dataset_path)
                if not report_paths:
                    # 根目录没找到，标记为按子数据集各自查找
                    per_sub_filter = True
                    logger.info(f"  No filter_report in dataset root, will search per sub-dataset")
            else:
                report_paths = [args.filter_report]

            if not per_sub_filter and report_paths:
                # 检查是否是 RoboMindV2.0 结构化格式（有 subdatasets 字段）
                for rp in report_paths:
                    with open(rp, encoding="utf-8") as f:
                        report_data = json.load(f)

                    if "subdatasets" in report_data:
                        # RoboMindV2.0 结构化格式
                        logger.info(f"  Filter report (structured): {os.path.basename(rp)}")
                        filter_report_map = load_filter_report_v2(rp)
                        total_problems = sum(len(eps) for eps in filter_report_map.values())
                        logger.info(f"    {len(filter_report_map)} subdatasets, {total_problems} total problem episodes")
                    else:
                        # 平面格式
                        if exclude_episodes is None:
                            exclude_episodes = set()
                        probs = load_filter_report(rp)
                        logger.info(f"  Filter report (flat): {os.path.basename(rp)}: {len(probs)} problem episodes")
                        exclude_episodes |= probs

                if exclude_episodes:
                    logger.info(f"  Total excluded episodes (flat): {len(exclude_episodes)}")

        sub_datasets = discover_sub_datasets(cfg)
        logger.info(f"  Found {len(sub_datasets)} sub-datasets, side={side}")

        if args.max_sub_datasets:
            sub_datasets = sub_datasets[:args.max_sub_datasets]
            logger.info(f"  Limiting to first {args.max_sub_datasets}")

        ds_results = []
        for idx, sub_info in enumerate(sub_datasets):
            # 适配新的返回格式：dict with keys 'path', 'robot_type', 'name'
            sub_path = sub_info["path"]
            robot_type = sub_info.get("robot_type", "")
            sub_name = sub_info.get("name", os.path.relpath(sub_path, cfg.dataset_path))

            side_suffix = f"_{side}" if side == "both" else f"_{side}"
            output_dir = os.path.join(args.output_root, cfg.dataset_name, sub_name + side_suffix)

            logger.info(f"\n  [{idx+1}/{len(sub_datasets)}] {sub_name}")
            if robot_type:
                logger.info(f"    robot_type: {robot_type}")

            if args.resume and os.path.exists(os.path.join(output_dir, "distance_matrix.npz")):
                logger.info(f"    SKIPPED (resume: cache exists)")
                ds_results.append({"sub": sub_name, "status": "skipped_resume"})
                continue

            try:
                min_rel_gap = args.min_rel_gap if args.min_rel_gap is not None else cfg.min_rel_gap

                # 按子数据集查找 filter_report（当根目录没有时）
                sub_exclude = exclude_episodes

                # 如果有结构化 filter_report，匹配 subdataset 名称
                if filter_report_map:
                    # 尝试多种匹配方式
                    task_name = os.path.basename(sub_path.rstrip("/"))
                    matched_key = None

                    # 1. 精确匹配完整路径（去掉开头的 robot_type）
                    for key in filter_report_map.keys():
                        if sub_name.endswith(key) or key in sub_name:
                            matched_key = key
                            break

                    # 2. 尝试只匹配 task 名称
                    if not matched_key and task_name in filter_report_map:
                        matched_key = task_name

                    if matched_key:
                        sub_exclude = filter_report_map[matched_key]
                        logger.info(f"    Filter: matched '{matched_key}', excluding {len(sub_exclude)} episodes")
                    else:
                        logger.info(f"    Filter: no match for '{sub_name}' in filter_report")

                elif per_sub_filter:
                    sub_reports = find_filter_reports(sub_path)
                    if sub_reports:
                        sub_exclude = set()
                        for rp in sub_reports:
                            probs = load_filter_report(rp)
                            logger.info(f"    Filter report {os.path.basename(rp)}: {len(probs)} problem episodes")
                            sub_exclude |= probs

                # 按子数据集 episode 数动态计算 n_clusters
                sub_n_clusters = n_clusters
                if args.auto_n_clusters_divisor:
                    sub_info_path = os.path.join(sub_path, "meta", "info.json")
                    if os.path.isfile(sub_info_path):
                        with open(sub_info_path) as f:
                            sub_info = json.load(f)
                        total_ep = sub_info.get("total_episodes", 0)
                        sub_n_clusters = max(2, total_ep // args.auto_n_clusters_divisor)
                        logger.info(f"    auto n_clusters: {total_ep} episodes // {args.auto_n_clusters_divisor} = {sub_n_clusters}")

                res = process_sub_dataset(
                    dataset_root=sub_path,
                    cfg=cfg,
                    side=side,
                    output_dir=output_dir,
                    window=args.window,
                    n_clusters=sub_n_clusters,
                    cluster_method=args.cluster_method,
                    linkage_method=args.linkage_method,
                    max_episodes=args.max_episodes,
                    logger=logger,
                    recursive=args.recursive,
                    max_depth=max_depth,
                    min_cluster_size=args.min_cluster_size,
                    min_rel_gap=min_rel_gap,
                    exclude_episodes=sub_exclude,
                    robot_type=robot_type,
                )
                res["sub"] = sub_name
                ds_results.append(res)
                logger.info(f"    -> {res['status']}")
            except Exception as e:
                logger.error(f"    FAILED: {e}")
                logger.error(traceback.format_exc())
                ds_results.append({"sub": sub_name, "status": "failed", "error": str(e)})

        summary[cfg.dataset_name] = {
            "total": len(sub_datasets),
            "processed": len([r for r in ds_results if r.get("status") == "success"]),
            "skipped": len([r for r in ds_results if "skipped" in r.get("status", "")]),
            "failed": len([r for r in ds_results if r.get("status") == "failed"]),
            "details": ds_results,
        }

    total_elapsed = time.time() - total_t0
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Batch complete in {total_elapsed:.1f}s")
    logger.info(f"{'=' * 70}")

    # 保存总结
    summary_path = os.path.join(args.output_root, "batch_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary saved to {summary_path}")

    # 打印概要
    for ds_name, info in summary.items():
        if isinstance(info, dict) and "total" in info:
            logger.info(
                f"  {ds_name}: {info['processed']}/{info['total']} success, "
                f"{info['skipped']} skipped, {info['failed']} failed"
            )
        else:
            logger.info(f"  {ds_name}: {info.get('status', '?')}")


if __name__ == "__main__":
    main()
