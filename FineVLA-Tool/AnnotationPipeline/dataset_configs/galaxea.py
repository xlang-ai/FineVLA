"""
Dataset configuration for Galaxea Open World (galaxea_open_world).

Views available:
  - head_rgb        (head-mounted, primary overview)
  - head_right_rgb  (head right)
  - left_wrist_rgb  (left wrist)
  - right_wrist_rgb (right wrist)

Processing model:
  Unlike the standard multi-stage pipeline (analysis → refinement), Galaxea
  samples already have ``steps_raw`` with per-step frame ranges (start/end).
  The pipeline has a single stage that iterates over each step, clips the
  video to the corresponding frame range, samples frames, and calls the VLM
  to produce a refined description for that step.

  Stage:
    refinement → fn=galaxea_step_refinement
      For each step in steps_raw, clip head_rgb video to [start, end],
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


class GalaxeaConfig(BaseDatasetConfig):
    dataset_name = "galaxea_open_world"

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

        chosen = "head_rgb" if "head_rgb" in videos else views[0]
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
