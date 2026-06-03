"""
Dataset configuration for RDT (rdt).

Views available:
  - main        (third-person overview)
  - left_wrist  (first-person, left arm)
  - right_wrist (first-person, right arm)

Strategy (3 stages):
  1. analysis         — main view, extract action_sequence + main_object
  2. refinement       — main view, produce fine_grained_steps + refined_instruction
  3. detail_refinement — wrist view (randomly left or right, seeded by sample_id),
                         polish fine-grained steps using first-person close-up
"""

import os
import random
from typing import Any, Dict, Optional, Tuple

from config import (
    VIDEO_BASE_DIR as _VIDEO_BASE_DIR,
    DEFAULT_ANALYSIS_FPS,
    DEFAULT_REFINEMENT_FPS,
)
from prompts import (
    ANALYSIS_SYSTEM_PROMPT_BIMANUAL,
    REFINEMENT_SYSTEM_PROMPT_BIMANUAL,
    DETAIL_REFINEMENT_SYSTEM_PROMPT_BIMANUAL,
)
from .base import BaseDatasetConfig, StageDefinition, stable_hash


class RdtConfig(BaseDatasetConfig):
    dataset_name = "rdt"

    stages = [
        StageDefinition(
            name="analysis",
            fn_name="analysis",
            default_fps=DEFAULT_ANALYSIS_FPS,
        ),
        StageDefinition(
            name="refinement",
            fn_name="refinement",
            depends_on="analysis",
            default_fps=DEFAULT_REFINEMENT_FPS,
        ),
        StageDefinition(
            name="detail_refinement",
            fn_name="detail_refinement",
            depends_on="refinement",
            default_fps=DEFAULT_REFINEMENT_FPS,
        ),
    ]

    wrist_views = ["left_wrist", "right_wrist"]

    def resolve_stage_views(
        self,
        sample: Dict[str, Any],
        video_base_dir: str = "",
    ) -> Dict[str, Dict[str, str]]:
        views = sample.get("views", [])
        videos = sample.get("videos", {})
        stage_names = [sd.name for sd in self.stages]

        if not views or not videos:
            return {name: {"view": "", "video_path": ""} for name in stage_names}

        base = video_base_dir or _VIDEO_BASE_DIR
        dataset_dir = sample.get("dataset_dir", "")

        def _resolve(view_name: str) -> Dict[str, str]:
            return {
                "view": view_name,
                "video_path": os.path.join(base, dataset_dir, videos[view_name]),
            }

        # Stages 1 & 2: main view
        main_view = "main" if "main" in videos else views[0]
        result = {
            "analysis": _resolve(main_view),
            "refinement": _resolve(main_view),
        }

        # Stage 3: randomly pick a wrist view (seeded for reproducibility)
        available_wrist = [v for v in self.wrist_views if v in videos]
        if available_wrist:
            rng = random.Random(stable_hash(sample.get("sample_id", "")))
            chosen_wrist = rng.choice(available_wrist)
        else:
            chosen_wrist = main_view
        result["detail_refinement"] = _resolve(chosen_wrist)

        return result

    _WRIST_LABEL = {"left_wrist": "LEFT", "right_wrist": "RIGHT"}

    def get_prompt_override(
        self,
        stage: str,
        sample: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[str, str]]:
        if stage in ("analysis", "refinement"):
            _map = {
                "analysis": (ANALYSIS_SYSTEM_PROMPT_BIMANUAL, ""),
                "refinement": (REFINEMENT_SYSTEM_PROMPT_BIMANUAL, ""),
            }
            return _map[stage]

        if stage == "detail_refinement":
            wrist_view = (sample or {}).get("stage_views", {}).get(
                "detail_refinement", {}
            ).get("view", "")
            arm_label = self._WRIST_LABEL.get(wrist_view, "")
            wrist_note = (
                f"\nNOTE — This video is captured from the **{arm_label} wrist camera** "
                f"(mounted on the {arm_label.lower()} arm's end-effector).\n"
                if arm_label else ""
            )
            sys_prompt = DETAIL_REFINEMENT_SYSTEM_PROMPT_BIMANUAL + wrist_note
            return (sys_prompt, "")

        return None
