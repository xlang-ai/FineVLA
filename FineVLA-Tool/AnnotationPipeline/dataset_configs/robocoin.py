"""
Dataset configuration for RoboCOIN (robocoin).

Views: varies by robot platform, but always has a main view + optional wrist views.
Main view priority: cam_high_rgb > camera_head_rgb > cam_front_rgb > cam_head_rgb
                    > cam_left_high > ego_view > first available

Strategy:
  - Samples WITH steps_raw (99.9%): per-step refinement using main view,
    bimanual prompt.
  - Samples WITHOUT steps_raw (0.1%): 2-stage pipeline (analysis → refinement),
    main view, bimanual prompt.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

from config import VIDEO_BASE_DIR as _VIDEO_BASE_DIR
from prompts import (
    GALAXEA_STEP_SYSTEM_PROMPT_BIMANUAL_ROBOCOIN,
    GALAXEA_STEP_PROMPT_TEMPLATE_ROBOCOIN,
    ANALYSIS_SYSTEM_PROMPT_BIMANUAL,
    REFINEMENT_SYSTEM_PROMPT_BIMANUAL,
)
from .base import BaseDatasetConfig, StageDefinition


_MAIN_VIEW_PRIORITY = [
    "cam_high_rgb",
    "camera_head_rgb",
    "cam_front_rgb",
    "cam_head_rgb",
    "cam_left_high",
    "ego_view",
]

_STEP_REFINEMENT_STAGES = [
    StageDefinition(
        name="refinement",
        fn_name="galaxea_step_refinement",
        default_fps=5.0,
    ),
]

_STANDARD_STAGES = [
    StageDefinition(
        name="analysis",
        fn_name="analysis",
        default_fps=3.0,
    ),
    StageDefinition(
        name="refinement",
        fn_name="refinement",
        depends_on="analysis",
        default_fps=5.0,
    ),
]


class RobocoinConfig(BaseDatasetConfig):
    dataset_name = "robocoin"

    stages = _STANDARD_STAGES

    def _pick_main_view(self, videos: Dict[str, str], views: List[str]) -> str:
        for v in _MAIN_VIEW_PRIORITY:
            if v in videos:
                return v
        # Fallback: first non-wrist view
        for v in views:
            if "wrist" not in v.lower():
                return v
        return views[0] if views else ""

    def get_active_stages(self, sample: Dict[str, Any]) -> List[StageDefinition]:
        if sample.get("steps_raw"):
            return _STEP_REFINEMENT_STAGES
        return _STANDARD_STAGES

    def resolve_stage_views(
        self,
        sample: Dict[str, Any],
        video_base_dir: str = "",
    ) -> Dict[str, Dict[str, str]]:
        views = sample.get("views", [])
        videos = sample.get("videos", {})

        active_stages = self.get_active_stages(sample)
        stage_names = [sd.name for sd in active_stages]

        if not views or not videos:
            return {name: {"view": "", "video_path": ""} for name in stage_names}

        base = video_base_dir or _VIDEO_BASE_DIR
        dataset_dir = sample.get("dataset_dir", "")

        chosen = self._pick_main_view(videos, views)
        video_path = os.path.join(base, dataset_dir, videos[chosen])

        return {name: {"view": chosen, "video_path": video_path} for name in stage_names}

    def get_prompt_override(
        self,
        stage: str,
        sample: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[str, str]]:
        if stage == "refinement" and sample and sample.get("steps_raw"):
            return (GALAXEA_STEP_SYSTEM_PROMPT_BIMANUAL_ROBOCOIN, GALAXEA_STEP_PROMPT_TEMPLATE_ROBOCOIN)
        if stage == "analysis":
            return (ANALYSIS_SYSTEM_PROMPT_BIMANUAL, "")
        if stage == "refinement":
            return (REFINEMENT_SYSTEM_PROMPT_BIMANUAL, "")
        return None
