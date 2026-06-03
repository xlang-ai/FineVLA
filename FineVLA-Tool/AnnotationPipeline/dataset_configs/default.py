"""
Default / fallback dataset configuration.

Used for any dataset that does not have a dedicated DatasetConfig subclass.
Picks the best available view using keyword-based heuristics.
"""

import os
from typing import Any, Dict, List

from config import VIDEO_BASE_DIR as _VIDEO_BASE_DIR
from .base import BaseDatasetConfig, ViewType


class DefaultConfig(BaseDatasetConfig):
    """
    Fallback config for datasets without a dedicated subclass.

    Runs the standard 2-stage pipeline (analysis -> refinement) and
    picks the best view using keyword heuristics.
    """

    dataset_name = "_default"

    def resolve_stage_views(
        self,
        sample: Dict[str, Any],
        video_base_dir: str = "",
    ) -> Dict[str, Dict[str, str]]:
        views: List[str] = sample.get("views", [])
        videos: Dict[str, str] = sample.get("videos", {})
        stage_names = [sd.name for sd in self.stages]

        if not views or not videos:
            return {name: {"view": "", "video_path": ""} for name in stage_names}

        base = video_base_dir or _VIDEO_BASE_DIR
        dataset_dir = sample.get("dataset_dir", "")

        fallback = self._keyword_fallback(views)
        chosen = fallback[0] if fallback else views[0]

        video_path = os.path.join(base, dataset_dir, videos[chosen])
        return {name: {"view": chosen, "video_path": video_path} for name in stage_names}

    def classify_view(self, sample: Dict[str, Any]) -> str:
        return ViewType.PRIMARY.value
