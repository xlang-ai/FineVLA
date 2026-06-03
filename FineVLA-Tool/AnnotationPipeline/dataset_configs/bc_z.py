"""
Dataset configuration for BC-Z (bc_z).

Views available:
  - image  (single primary exterior view)

Strategy (2 stages):
  Both analysis and refinement use the single `image` view.
  dataset_dir is derived internally as "BC_Z" (not present in JSONL samples).
"""

import os
from typing import Any, Dict

from config import VIDEO_BASE_DIR as _VIDEO_BASE_DIR
from .base import BaseDatasetConfig, DEFAULT_STAGES, stable_hash


DS_DIR = "BC_Z"
VIEW = "image"


class BcZConfig(BaseDatasetConfig):
    dataset_name = "bc_z"
    stages = DEFAULT_STAGES

    def resolve_stage_views(
        self,
        sample: Dict[str, Any],
        video_base_dir: str = "",
    ) -> Dict[str, Dict[str, str]]:
        videos = sample.get("videos", {})
        if not videos:
            return {sd.name: {"view": "", "video_path": ""} for sd in self.stages}

        base = video_base_dir or _VIDEO_BASE_DIR
        dataset_dir = sample.get("dataset_dir", "") or DS_DIR
        view = VIEW if VIEW in videos else list(videos.keys())[0]
        video_path = os.path.join(base, dataset_dir, videos[view])
        return {sd.name: {"view": view, "video_path": video_path} for sd in self.stages}
