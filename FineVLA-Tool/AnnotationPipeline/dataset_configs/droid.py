"""
Dataset configuration for DROID (droid_robointer).

Views available:
  - primary  (main exterior view)
  - wrist    (wrist-mounted)

Strategy:
  - Samples WITH steps_raw (99.4%): Galaxea-style per-step refinement using
    primary view. Single stage: galaxea_step_refinement.
  - Samples WITHOUT steps_raw (0.6%): Standard 2-stage pipeline
    (analysis → refinement), both stages use primary view.
"""

import os
from typing import Any, Dict, List, Optional, Tuple


from config import (
    VIDEO_BASE_DIR as _VIDEO_BASE_DIR,
    DEFAULT_ANALYSIS_FPS,
    DEFAULT_REFINEMENT_FPS,
)
from prompts import GALAXEA_STEP_SYSTEM_PROMPT
from .base import BaseDatasetConfig, StageDefinition


_STEP_REFINEMENT_STAGES = [
    StageDefinition(
        name="refinement",
        fn_name="galaxea_step_refinement",
        default_fps=3.0,
    ),
]


class DroidConfig(BaseDatasetConfig):
    dataset_name = "droid_robointer"

    # Stages for samples without steps_raw: analysis=3fps, refinement=5fps
    stages = [
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

    def get_active_stages(self, sample: Dict[str, Any]) -> List[StageDefinition]:
        """Per-step refinement if steps_raw exists, otherwise standard 2-stage."""
        steps_raw = sample.get("steps_raw", [])
        if steps_raw:
            return _STEP_REFINEMENT_STAGES
        return self.stages

    def resolve_stage_views(
        self,
        sample: Dict[str, Any],
        video_base_dir: str = "",
    ) -> Dict[str, Dict[str, str]]:
        views = sample.get("views", [])
        videos = sample.get("videos", {})

        # Determine which stages this sample uses
        active_stages = self.get_active_stages(sample)
        stage_names = [sd.name for sd in active_stages]

        if not views or not videos:
            return {name: {"view": "", "video_path": ""} for name in stage_names}

        base = video_base_dir or _VIDEO_BASE_DIR
        dataset_dir = sample.get("dataset_dir", "")

        # Always use primary view
        chosen = "primary" if "primary" in videos else views[0]
        video_path = os.path.join(base, dataset_dir, videos[chosen])

        return {name: {"view": chosen, "video_path": video_path} for name in stage_names}

    def get_prompt_override(
        self,
        stage: str,
        sample: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[str, str]]:
        if stage == "refinement" and sample and sample.get("steps_raw"):
            return (GALAXEA_STEP_SYSTEM_PROMPT, "")
        return None
