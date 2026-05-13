"""
Port Mobile ALOHA-style episode_*.hdf5 to LeRobot dataset (new lerobot package layout).

Your raw HDF5 format:
  root keys: action (T,14), base_action (T,2), observations/
  observations keys: qpos (T,14), qvel (T,14), effort (T,14), images/<cam> (T,480,640,3) uint8
  cameras: cam_high, cam_low, cam_left_wrist, cam_right_wrist

This script:
  - ignores base_action (only arms)
  - writes observation.state = qpos
  - optionally writes observation.velocity (qvel) and observation.effort
  - writes images as "video" or "image" dtype in features (but always feeds HWC uint8 frames)
  - writes into --out_dir (no LEROBOT_HOME)
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal, Optional
import inspect

import h5py
import numpy as np
import tqdm
import tyro
import datasets

from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    # Some lerobot versions accept these in create(); others ignore them.
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 16
    image_writer_threads: int = 8
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def _call_create_dataset(
    *,
    repo_id: str,
    root: Path,
    fps: int,
    robot_type: str,
    features: dict,
    dataset_config: DatasetConfig,
) -> LeRobotDataset:
    """Call LeRobotDataset.create with best-effort compatibility across versions."""
    sig = inspect.signature(LeRobotDataset.create)
    kwargs = {
        "repo_id": repo_id,
        "fps": fps,
        "robot_type": robot_type,
        "features": features,
    }

    if "root" in sig.parameters:
        kwargs["root"] = root

    # optional args
    for k in ["use_videos", "tolerance_s", "image_writer_processes", "image_writer_threads", "video_backend"]:
        if k in sig.parameters:
            kwargs[k] = getattr(dataset_config, k)

    return LeRobotDataset.create(**kwargs)


def _add_frame_compat(dataset: LeRobotDataset, frame: dict, task: str):
    """
    Compatible with:
      - dataset.add_frame(frame, task)
      - dataset.add_frame(frame) where frame contains "task"
    """
    try:
        sig = inspect.signature(dataset.add_frame)
        if len(sig.parameters) >= 2:
            dataset.add_frame(frame, task)
            return
    except Exception:
        pass

    # fallback
    frame2 = dict(frame)
    frame2["task"] = task
    dataset.add_frame(frame2)


def _save_episode_compat(dataset: LeRobotDataset, task: str):
    """
    Compatible with:
      - dataset.save_episode()
      - dataset.save_episode(task=task)
    """
    if not hasattr(dataset, "save_episode"):
        return
    try:
        sig = inspect.signature(dataset.save_episode)
        if "task" in sig.parameters:
            dataset.save_episode(task=task)
        else:
            dataset.save_episode()
    except TypeError:
        # last fallback
        dataset.save_episode()


def _finalize_compat(dataset: LeRobotDataset):
    # Some versions provide finalize(), others consolidate(), others nothing.
    if hasattr(dataset, "finalize"):
        try:
            dataset.finalize()
            return
        except Exception:
            pass
    if hasattr(dataset, "consolidate"):
        try:
            dataset.consolidate()
            return
        except Exception:
            pass


def _ensure_image_features(dataset: LeRobotDataset, cameras: list[str]) -> None:
    """Force HF features to use Image(decode=True) for image columns."""
    if not hasattr(dataset, "hf_dataset") or dataset.hf_dataset is None:
        return
    image_keys = [f"observation.images.{cam}" for cam in cameras]
    features = dataset.hf_dataset.features.copy()
    updated = False
    for key in image_keys:
        if key not in features:
            continue
        ft = features[key]
        if not isinstance(ft, datasets.Image) or getattr(ft, "decode", None) is not True:
            features[key] = datasets.Image(decode=True)
            updated = True
    if updated:
        dataset.hf_dataset = dataset.hf_dataset.cast(features)


def get_cameras_from_file(h5_path: Path) -> list[str]:
    with h5py.File(h5_path, "r") as ep:
        return list(ep["/observations/images"].keys())


def has_velocity(h5_path: Path) -> bool:
    with h5py.File(h5_path, "r") as ep:
        return "/observations/qvel" in ep


def has_effort(h5_path: Path) -> bool:
    with h5py.File(h5_path, "r") as ep:
        return "/observations/effort" in ep


def build_features(
    cameras: list[str],
    mode: Literal["video", "image"] = "image",
    has_velocity_: bool = True,
    has_effort_: bool = True,
) -> dict:
    motors = [
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
    }

    if has_velocity_:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        }

    if has_effort_:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        }

    # Store images as CHW to match common LeRobot metadata.
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,                # "image" or "video"
            "shape": (3, 480, 640),       # CHW
            "names": ["channels", "height", "width"],
        }

    return features


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    """
    Your data is uncompressed: (T,480,640,3) uint8.
    Keep a compressed branch in case you later meet such data.
    """
    imgs_per_cam: dict[str, np.ndarray] = {}
    for camera in cameras:
        ds = ep[f"/observations/images/{camera}"]
        if ds.ndim == 4:
            imgs_per_cam[camera] = ds[:]  # (T,H,W,3)
        else:
            import cv2
            imgs = []
            for data in ds:
                imgs.append(cv2.cvtColor(cv2.imdecode(data, 1), cv2.COLOR_BGR2RGB))
            imgs_per_cam[camera] = np.array(imgs)
    return imgs_per_cam


def load_raw_episode_data(ep_path: Path, cameras: list[str]):
    with h5py.File(ep_path, "r") as ep:
        state = ep["/observations/qpos"][:].astype(np.float32)   # (T,14)
        action = ep["/action"][:].astype(np.float32)             # (T,14)

        velocity = ep["/observations/qvel"][:].astype(np.float32) if "/observations/qvel" in ep else None
        effort = ep["/observations/effort"][:].astype(np.float32) if "/observations/effort" in ep else None

        imgs_per_cam = load_raw_images_per_camera(ep, cameras)

    return imgs_per_cam, state, action, velocity, effort


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    cameras: list[str],
    episodes: Optional[list[int]] = None,
):
    if episodes is None:
        episodes = list(range(len(hdf5_files)))

    for ep_idx in tqdm.tqdm(episodes, desc="Converting episodes"):
        ep_path = hdf5_files[ep_idx]

        imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path, cameras)
        T = state.shape[0]

        for t in range(T):
            frame = {
                "observation.state": state[t],
                "action": action[t],
            }

            for cam, arr in imgs_per_cam.items():
                # Convert HWC -> CHW to match feature metadata.
                frame[f"observation.images.{cam}"] = arr[t].transpose(2, 0, 1)

            if velocity is not None:
                frame["observation.velocity"] = velocity[t]
            if effort is not None:
                frame["observation.effort"] = effort[t]

            _add_frame_compat(dataset, frame, task)

        _save_episode_compat(dataset, task)

    return dataset


def port_mobile_aloha(
    raw_dir: Path,
    out_dir: Path,
    repo_id: str,
    task: str = "towel fold",
    *,
    episodes: Optional[list[int]] = None,
    mode: Literal["video", "image"] = "image",
    fps: int = 20,
    robot_type: str = "mobile_aloha",
    overwrite: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    raw_dir = raw_dir.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()

    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir does not exist: {raw_dir}")

    hdf5_files = sorted(raw_dir.glob("episode_*.hdf5"))
    if not hdf5_files:
        raise RuntimeError(f"No episode_*.hdf5 found in {raw_dir}")

    # enforce repo_id format "org/name" to satisfy some lerobot versions
    if "/" not in repo_id:
        repo_id = f"local/{repo_id}"

    # dataset root path
    root = out_dir / repo_id
    if root.exists():
        if overwrite:
            shutil.rmtree(root)
        else:
            raise FileExistsError(f"Output already exists: {root}. Use --overwrite true")

    cameras = get_cameras_from_file(hdf5_files[0])

    features = build_features(
        cameras=cameras,
        mode=mode,
        has_velocity_=has_velocity(hdf5_files[0]),
        has_effort_=has_effort(hdf5_files[0]),
    )

    dataset = _call_create_dataset(
        repo_id=repo_id,
        root=root,
        fps=fps,
        robot_type=robot_type,
        features=features,
        dataset_config=dataset_config,
    )
    _ensure_image_features(dataset, cameras)

    populate_dataset(dataset, hdf5_files, task=task, cameras=cameras, episodes=episodes)

    _finalize_compat(dataset)

    print(f"\n✅ Done. Dataset saved to: {root}")


if __name__ == "__main__":
    tyro.cli(port_mobile_aloha)
