"""
Dataset configuration for AgiBotWorld (agibotworld).

Views available (8 cameras):
  - head                (primary overview)
  - head_center_fisheye, head_left_fisheye, head_right_fisheye
  - hand_left, hand_right
  - back_left_fisheye, back_right_fisheye

Processing model:
  Same as Galaxea — samples have ``steps_raw`` with per-step frame ranges.
  Single stage iterates over each step, clips the head video, and calls
  the VLM to produce a refined description.

  All samples are dual-arm (a2d), so bimanual prompts are used.

  Stage:
    refinement → fn=galaxea_step_refinement
      For each step in steps_raw, clip head video to [start, end],
      call VLM to refine the step description.
      Output: fine_grained_steps + refined_instruction
"""

import os
from typing import Any, Dict, List, Optional, Tuple

from config import (
    VIDEO_BASE_DIR as _VIDEO_BASE_DIR,
    DEFAULT_REFINEMENT_FPS,
)
from prompts import GALAXEA_STEP_SYSTEM_PROMPT_BIMANUAL
from .base import BaseDatasetConfig, StageDefinition, ViewType


class AgiBotWorldConfig(BaseDatasetConfig):
    dataset_name = "agibotworld"

    stages = [
        StageDefinition(
            name="refinement",
            fn_name="galaxea_step_refinement",
            default_fps=DEFAULT_REFINEMENT_FPS,
        ),
    ]

    def resolve_stage_views(
        self,
        sample: Dict[str, Any],
        video_base_dir: str = "",
    ) -> Dict[str, Dict[str, str]]:
        views: List[str] = sample.get("views", [])
        videos: Dict[str, str] = sample.get("videos", {})

        if not views or not videos:
            return {"refinement": {"view": "", "video_path": ""}}

        base = video_base_dir or _VIDEO_BASE_DIR
        dataset_dir = sample.get("dataset_dir", "")

        chosen = "head" if "head" in videos else views[0]
        video_path = os.path.join(base, dataset_dir, videos[chosen])

        return {
            "refinement": {"view": chosen, "video_path": video_path},
        }

    def classify_view(self, sample: Dict[str, Any]) -> str:
        return ViewType.HEAD.value

    def get_prompt_override(
        self,
        stage: str,
        sample: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[str, str]]:
        if stage == "refinement":
            return (GALAXEA_STEP_SYSTEM_PROMPT_BIMANUAL, "")
        return None
