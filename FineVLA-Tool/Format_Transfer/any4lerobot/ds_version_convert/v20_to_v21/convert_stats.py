import json
import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats, sample_indices
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import (
    EPISODES_STATS_PATH,
    load_episodes_stats,
    write_episode_stats,
)
from lerobot.datasets.v21.convert_dataset_v20_to_v21 import V20
from tqdm import tqdm


def _load_completed_episodes(root: Path) -> set[int]:
    """Load episode indices that already have stats on disk (for resume)."""
    stats_path = root / EPISODES_STATS_PATH
    if not stats_path.is_file():
        return set()
    try:
        existing = load_episodes_stats(root)
        return set(existing.keys())
    except Exception:
        return set()


def sample_episode_video_frames(dataset: LeRobotDataset, episode_index: int, ft_key: str) -> np.ndarray:
    ep_len = dataset.meta.episodes[episode_index]["length"]
    sampled_indices = sample_indices(ep_len)
    query_timestamps = dataset._get_query_timestamps(0.0, {ft_key: sampled_indices})
    video_frames = dataset._query_videos(query_timestamps, episode_index)
    return video_frames[ft_key].numpy()


def convert_episode_stats(dataset: LeRobotDataset, ep_idx: int):
    ep_start_idx = dataset.episode_data_index["from"][ep_idx]
    ep_end_idx = dataset.episode_data_index["to"][ep_idx]
    ep_data = dataset.hf_dataset.select(range(ep_start_idx, ep_end_idx))

    ep_stats = {}
    anomalies = []
    for key, ft in dataset.features.items():
        if ft["dtype"] == "video":
            ep_ft_data = sample_episode_video_frames(dataset, ep_idx, key)
        else:
            ep_ft_data = np.array(ep_data[key])

        if ft["dtype"] in ["image", "video"] and ep_ft_data.ndim == 3:
            anomalies.append({
                "feature": key,
                "expected_ndim": 4,
                "actual_ndim": 3,
                "actual_shape": list(ep_ft_data.shape),
            })
            ep_ft_data = np.expand_dims(ep_ft_data, axis=0)
        axes_to_reduce = (0, 2, 3) if ft["dtype"] in ["image", "video"] else 0
        keepdims = True if ft["dtype"] in ["image", "video"] else ep_ft_data.ndim == 1
        ep_stats[key] = get_feature_stats(ep_ft_data, axis=axes_to_reduce, keepdims=keepdims)

        if ft["dtype"] in ["image", "video"]:  # remove batch dim
            ep_stats[key] = {k: v if k == "count" else np.squeeze(v, axis=0) for k, v in ep_stats[key].items()}

    return ep_stats, ep_idx, anomalies


# ---------------------------------------------------------------------------
# Backend: ProcessPoolExecutor with initializer
#   Each worker loads dataset once; subsequent tasks only pass ep_idx (int).
# ---------------------------------------------------------------------------
_worker_dataset = None


def _init_worker(repo_id, root):
    global _worker_dataset
    os.environ["HF_HUB_OFFLINE"] = "1"
    logging.disable(logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _worker_dataset = LeRobotDataset(repo_id, root, revision=V20)
    logging.disable(logging.NOTSET)


def _worker_compute(ep_idx):
    return convert_episode_stats(_worker_dataset, ep_idx)


def _convert_stats_process(dataset, num_workers, todo_episodes, all_anomalies):
    actual_workers = min(num_workers, len(todo_episodes))
    print(f"  ProcessPoolExecutor: {actual_workers} workers, {len(todo_episodes)} episodes")
    with ProcessPoolExecutor(
        max_workers=actual_workers,
        initializer=_init_worker,
        initargs=(dataset.repo_id, str(dataset.root)),
    ) as executor:
        futures = {
            executor.submit(_worker_compute, ep_idx): ep_idx
            for ep_idx in todo_episodes
        }
        for future in tqdm(as_completed(futures), total=len(todo_episodes)):
            ep_stats, ep_idx, anomalies = future.result()
            dataset.meta.episodes_stats[ep_idx] = ep_stats
            write_episode_stats(ep_idx, ep_stats, dataset.root)
            if anomalies:
                all_anomalies[ep_idx] = anomalies


# ---------------------------------------------------------------------------
# Backend: Ray Actors
#   Each Actor holds its own dataset; dynamic work-stealing for load balance.
# ---------------------------------------------------------------------------
def _convert_stats_ray(dataset, num_workers, todo_episodes, all_anomalies):
    import ray

    ray.init(num_cpus=num_workers, ignore_reinit_error=True)

    @ray.remote
    class StatsWorker:
        def __init__(self, repo_id, root):
            import resource as _resource
            try:
                _resource.setrlimit(_resource.RLIMIT_NOFILE, (1048576, 1048576))
            except (ValueError, OSError):
                pass
            os.environ["HF_HUB_OFFLINE"] = "1"
            logging.disable(logging.WARNING)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.dataset = LeRobotDataset(repo_id, root, revision=V20)
            logging.disable(logging.NOTSET)

        def compute(self, ep_idx):
            return convert_episode_stats(self.dataset, ep_idx)

        def ping(self):
            return True

    actual_workers = min(num_workers, len(todo_episodes))
    print(f"  Ray: spawning {actual_workers} actors (sequential init to avoid CPFS FD exhaustion)")
    workers = []
    for i in range(actual_workers):
        w = StatsWorker.remote(dataset.repo_id, str(dataset.root))
        ray.get(w.ping.remote())
        workers.append(w)
        print(f"    Actor {i+1}/{actual_workers} ready")

    ep_iter = iter(todo_episodes)
    pending = {}  # ray ObjectRef -> (ep_idx, worker)

    for worker in workers:
        ep_idx = next(ep_iter, None)
        if ep_idx is None:
            break
        ref = worker.compute.remote(ep_idx)
        pending[ref] = (ep_idx, worker)

    with tqdm(total=len(todo_episodes)) as pbar:
        while pending:
            done, _ = ray.wait(list(pending.keys()), num_returns=1)
            for ref in done:
                ep_stats, ep_idx, anomalies = ray.get(ref)
                dataset.meta.episodes_stats[ep_idx] = ep_stats
                write_episode_stats(ep_idx, ep_stats, dataset.root)
                if anomalies:
                    all_anomalies[ep_idx] = anomalies
                _, worker = pending.pop(ref)
                pbar.update(1)

                next_ep = next(ep_iter, None)
                if next_ep is not None:
                    new_ref = worker.compute.remote(next_ep)
                    pending[new_ref] = (next_ep, worker)

    ray.shutdown()


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def _save_anomalies(all_anomalies: dict, root: Path):
    """Write anomalous episodes report to meta/anomalous_episodes.json."""
    if not all_anomalies:
        return
    report_path = root / "meta" / "anomalous_episodes.json"
    sorted_report = {k: all_anomalies[k] for k in sorted(all_anomalies)}
    report_path.write_text(json.dumps(sorted_report, indent=2))
    print(f"\n{'='*60}")
    print(f"WARNING: {len(sorted_report)} episodes had anomalous video dimensions (3D instead of 4D).")
    print(f"  These episodes were still processed (with expand_dims fix),")
    print(f"  but may indicate corrupted or single-frame videos.")
    print(f"  Anomalous episode indices: {list(sorted_report.keys())}")
    print(f"  Full report saved to: {report_path}")
    print(f"{'='*60}\n")


def convert_stats(dataset: LeRobotDataset, num_workers: int = 0, backend: str = "process"):
    assert dataset.episodes is None
    total_episodes = dataset.meta.total_episodes

    done_eps = _load_completed_episodes(dataset.root)
    todo_episodes = sorted(set(range(total_episodes)) - done_eps)

    if done_eps:
        print(f"Resuming: {len(done_eps)}/{total_episodes} episodes already done, "
              f"{len(todo_episodes)} remaining")
        existing_stats = load_episodes_stats(dataset.root)
        for ep_idx in done_eps:
            dataset.meta.episodes_stats[ep_idx] = existing_stats[ep_idx]
    else:
        print(f"Computing episodes stats (backend={backend}, workers={num_workers})")

    if not todo_episodes:
        print("All episodes already computed, skipping.")
        return

    print(f"Computing {len(todo_episodes)} episodes (backend={backend}, workers={num_workers})")

    all_anomalies = {}

    if backend == "ray" and num_workers > 0:
        _convert_stats_ray(dataset, num_workers, todo_episodes, all_anomalies)
    elif num_workers > 0:
        _convert_stats_process(dataset, num_workers, todo_episodes, all_anomalies)
    else:
        for ep_idx in tqdm(todo_episodes):
            ep_stats, _, anomalies = convert_episode_stats(dataset, ep_idx)
            dataset.meta.episodes_stats[ep_idx] = ep_stats
            write_episode_stats(ep_idx, ep_stats, dataset.root)
            if anomalies:
                all_anomalies[ep_idx] = anomalies

    _save_anomalies(all_anomalies, dataset.root)
    print(f"All {total_episodes} episodes stats computed and written to disk.")


def check_aggregate_stats(
    dataset: LeRobotDataset,
    reference_stats: dict[str, dict[str, np.ndarray]],
    video_rtol_atol: tuple[float] = (1e-2, 1e-2),
    default_rtol_atol: tuple[float] = (5e-6, 6e-5),
):
    """Verifies that the aggregated stats from episodes_stats are close to reference stats."""
    agg_stats = aggregate_stats(list(dataset.meta.episodes_stats.values()))
    for key, ft in dataset.features.items():
        if ft["dtype"] == "video":
            rtol, atol = video_rtol_atol
        else:
            rtol, atol = default_rtol_atol

        for stat, val in agg_stats[key].items():
            if key in reference_stats and stat in reference_stats[key]:
                err_msg = f"feature='{key}' stats='{stat}'"
                np.testing.assert_allclose(val, reference_stats[key][stat], rtol=rtol, atol=atol, err_msg=err_msg)
