"""
Dataset configuration for Bridge (bridge).

Views available:
  - image_0  (primary) — only usable view; image_1/2/3 are mostly black
  - image_1  (84% black)
  - image_2  (84% black)
  - image_3  (99.8% black)

Strategy (2 stages):
  Both analysis and refinement use image_0.
"""

import os
from typing import Any, Dict

from config import VIDEO_BASE_DIR as _VIDEO_BASE_DIR
from .base import BaseDatasetConfig, DEFAULT_STAGES


DS_DIR = "Bridge"
VIEW = "image_0"


class BridgeConfig(BaseDatasetConfig):
    dataset_name = "bridge"
    stages = DEFAULT_STAGES

    def resolve_stage_views(
        self,
        sample: Dict[str, Any],
        video_base_dir: str = "",
    ) -> Dict[str, Dict[str, str]]:
        videos = sample.get("videos", {})
        if not videos or VIEW not in videos:
            return {sd.name: {"view": "", "video_path": ""} for sd in self.stages}

        base = video_base_dir or _VIDEO_BASE_DIR
        dataset_dir = sample.get("dataset_dir", "") or DS_DIR

        video_path = os.path.join(base, dataset_dir, videos[VIEW])
        return {
            sd.name: {"view": VIEW, "video_path": video_path}
            for sd in self.stages
        }
