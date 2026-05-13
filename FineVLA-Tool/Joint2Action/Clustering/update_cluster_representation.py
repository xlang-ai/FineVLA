"""
更新 cluster_results.json 中的 cluster_representation，
对每个簇选择最多3个代表性 episode（基于距离矩阵选 medoid + 最远的2个）。
如果簇内 episode 数 <= 3，则全选。

用法:
    python update_cluster_representation.py \
        --results_root ./results_two_stage \
        --dataset Galaxea \
        --num_repr 3
"""

import argparse
import json
import os
import numpy as np


def select_representatives(episode_ids: list[int], dist_matrix: np.ndarray,
                           ep_id_to_idx: dict[int, int], num_repr: int = 3) -> list[int]:
    """从一个簇的 episode 列表中选择 num_repr 个代表性 episode。

    策略：
    1. 选 medoid（簇内到其他所有 episode 距离之和最小的）
    2. 贪心选剩余：每次选与已选集合距离最大的
    """
    if len(episode_ids) <= num_repr:
        return episode_ids

    # 提取簇内子距离矩阵
    indices = [ep_id_to_idx[eid] for eid in episode_ids if eid in ep_id_to_idx]
    if len(indices) <= num_repr:
        return episode_ids[:num_repr]

    sub_dist = dist_matrix[np.ix_(indices, indices)]

    # Step 1: medoid
    sum_dists = sub_dist.sum(axis=1)
    medoid_local = int(np.argmin(sum_dists))
    selected_local = [medoid_local]

    # Step 2: 贪心选最远的
    for _ in range(num_repr - 1):
        max_min_dist = -1
        best_candidate = -1
        for i in range(len(indices)):
            if i in selected_local:
                continue
            min_dist_to_selected = min(sub_dist[i, s] for s in selected_local)
            if min_dist_to_selected > max_min_dist:
                max_min_dist = min_dist_to_selected
                best_candidate = i
        if best_candidate >= 0:
            selected_local.append(best_candidate)

    return [episode_ids[i] for i in selected_local]


def process_subdataset(subdir_path: str, num_repr: int) -> bool:
    """处理单个子数据集目录，更新 cluster_results.json。返回是否成功。"""
    json_path = os.path.join(subdir_path, "cluster_results.json")
    npz_path = os.path.join(subdir_path, "distance_matrix.npz")

    if not os.path.isfile(json_path):
        return False

    with open(json_path) as f:
        data = json.load(f)

    leaf_clusters = data.get("leaf_clusters")
    dataset_root = data.get("dataset_root", "")
    if not leaf_clusters:
        return False

    # 加载距离矩阵
    if os.path.isfile(npz_path):
        npz = np.load(npz_path)
        dist_matrix = npz["dist_matrix"]
        episode_ids_arr = npz["episode_ids"]
        ep_id_to_idx = {int(eid): i for i, eid in enumerate(episode_ids_arr)}
    else:
        dist_matrix = None
        ep_id_to_idx = {}

    new_repr = {}
    for cluster_id, cluster_info in leaf_clusters.items():
        episodes = cluster_info.get("episodes", [])
        if not episodes:
            continue

        if dist_matrix is not None and len(episodes) > num_repr:
            selected = select_representatives(episodes, dist_matrix, ep_id_to_idx, num_repr)
        else:
            selected = episodes[:num_repr]

        new_repr[cluster_id] = [f"{dataset_root}/{ep}" for ep in selected]

    data["cluster_representation"] = new_repr

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return True


def main():
    p = argparse.ArgumentParser(description="Update cluster_representation to N episodes per cluster")
    p.add_argument("--results_root", type=str, default="./results_two_stage")
    p.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. Galaxea)")
    p.add_argument("--num_repr", type=int, default=3, help="Number of representative episodes per cluster")
    args = p.parse_args()

    dataset_dir = os.path.join(args.results_root, args.dataset)
    if not os.path.isdir(dataset_dir):
        print(f"ERROR: {dataset_dir} not found")
        return

    total = 0
    updated = 0
    for sub_name in sorted(os.listdir(dataset_dir)):
        sub_path = os.path.join(dataset_dir, sub_name)
        if not os.path.isdir(sub_path):
            continue
        total += 1
        if process_subdataset(sub_path, args.num_repr):
            updated += 1

    print(f"Done. Updated {updated}/{total} subdatasets in {args.dataset}")


if __name__ == "__main__":
    main()
