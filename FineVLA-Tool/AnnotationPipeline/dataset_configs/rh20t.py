"""
Dataset configuration for RH20T (rh20t_robointer).

Views: primary + wrist

Strategy:
  - Samples WITH steps_raw (94%): galaxea_step_refinement，按 step 逐段精化，使用 primary view。
  - Samples WITHOUT steps_raw (6%): 标准两阶段 analysis → refinement，随机选不同视角。
"""

import os
import random
from typing import Any, Dict, List, Optional, Tuple

from config import (
    VIDEO_BASE_DIR as _VIDEO_BASE_DIR,
    DEFAULT_ANALYSIS_FPS,
    DEFAULT_REFINEMENT_FPS,
)
from prompts import GALAXEA_STEP_SYSTEM_PROMPT
from .base import BaseDatasetConfig, StageDefinition, ViewType, stable_hash, DEFAULT_STAGES

_STEP_REFINEMENT_STAGES = [
    StageDefinition(
        name="refinement",
        fn_name="galaxea_step_refinement",
        default_fps=3.0,
    ),
]


class Rh20tConfig(BaseDatasetConfig):
    dataset_name = "rh20t_robointer"

    # 无 steps_raw 时走标准两阶段
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
    ]

    def get_active_stages(self, sample: Dict[str, Any]) -> List[StageDefinition]:
        if sample.get("steps_raw"):
            return _STEP_REFINEMENT_STAGES
        return DEFAULT_STAGES

    def resolve_stage_views(
        self,
        sample: Dict[str, Any],
        video_base_dir: str = "",
    ) -> Dict[str, Dict[str, str]]:
        views: List[str] = sample.get("views", [])
        videos: Dict[str, str] = sample.get("videos", {})

        active_stages = self.get_active_stages(sample)
        stage_names = [sd.name for sd in active_stages]

        if not views or not videos:
            return {name: {"view": "", "video_path": ""} for name in stage_names}

        base = video_base_dir or _VIDEO_BASE_DIR
        dataset_dir = sample.get("dataset_dir", "")

        def _resolve(view_name: str) -> Dict[str, str]:
            return {
                "view": view_name,
                "video_path": os.path.join(base, dataset_dir, videos[view_name]),
            }

        # 有 steps_raw：统一用 primary view
        if sample.get("steps_raw"):
            chosen = "primary" if "primary" in videos else views[0]
            return {"refinement": _resolve(chosen)}

        # 无 steps_raw：两阶段随机选不同视角
        available = [v for v in views if v in videos]
        if not available:
            available = views

        rng = random.Random(stable_hash(sample.get("sample_id", "")))
        rng.shuffle(available)

        view_analysis = available[0]
        view_refinement = available[1] if len(available) > 1 else available[0]

        return {
            "analysis": _resolve(view_analysis),
            "refinement": _resolve(view_refinement),
        }

    def get_prompt_override(
        self,
        stage: str,
        sample: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[str, str]]:
        if stage == "refinement" and sample and sample.get("steps_raw"):
            return (GALAXEA_STEP_SYSTEM_PROMPT, "")
        return None

    def classify_view(self, sample: Dict[str, Any]) -> str:
        return ViewType.PRIMARY.value
