"""
Dataset configuration for RT-1 (rt_1).

Views available:
  - image  (single primary exterior view)

Strategy (2 stages):
  Both analysis and refinement use the single `image` view.
  Analysis FPS is set to 3.0 to match the dataset's native 3 FPS capture rate,
  avoiding semantic ambiguity from over-sampling.
  dataset_dir is derived internally as "RT-1" (not present in JSONL samples).
"""

import os
from typing import Any, Dict

from config import VIDEO_BASE_DIR as _VIDEO_BASE_DIR, DEFAULT_REFINEMENT_FPS
from .base import BaseDatasetConfig, StageDefinition


DS_DIR = "RT-1"
VIEW = "image"


class Rt1Config(BaseDatasetConfig):
    dataset_name = "rt_1"
    stages = [
        StageDefinition(name="analysis",   fn_name="analysis",   default_fps=3.0),
        StageDefinition(name="refinement", fn_name="refinement", depends_on="analysis",
                        default_fps=DEFAULT_REFINEMENT_FPS),
    ]

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
