"""
Dataset configuration for RoboMIND V1.0 (robomind_v1.0).

Pipeline:
  - Samples WITH steps_raw  → galaxea_step_refinement (per-step clip)
  - Samples WITHOUT steps_raw → analysis → refinement (2-stage)

View strategy:
  When both camera_top and camera_front are available, randomly pick one
  (seeded by sample_id for reproducibility).
  - galaxea_step_refinement : random(camera_top, camera_front) > first
  - analysis / refinement   : random swap between camera_top and camera_front
"""

import os
import random
from typing import Any, Dict, List, Optional, Tuple

from config import VIDEO_BASE_DIR as _VIDEO_BASE_DIR
from prompts import GALAXEA_STEP_SYSTEM_PROMPT
from .base import BaseDatasetConfig, StageDefinition, ViewType, stable_hash, DEFAULT_STAGES

_ANALYSIS_FPS    = 3.0
_REFINEMENT_FPS  = 5.0
_STEP_REFINE_FPS = 5.0

_STAGE_ANALYSIS = StageDefinition(
    name="analysis",
    fn_name="analysis",
    default_fps=_ANALYSIS_FPS,
)
_STAGE_REFINEMENT = StageDefinition(
    name="refinement",
    fn_name="refinement",
    depends_on="analysis",
    default_fps=_REFINEMENT_FPS,
)
_STAGE_STEP_REFINEMENT = StageDefinition(
    name="refinement",
    fn_name="galaxea_step_refinement",
    default_fps=_STEP_REFINE_FPS,
)

_TWO_STAGES  = [_STAGE_ANALYSIS, _STAGE_REFINEMENT]
_STEP_STAGES = [_STAGE_STEP_REFINEMENT]

_VIEW_TYPE = {
    "agilex_3rgb":                ViewType.PRIMARY,
    "franka_3rgb":                ViewType.OVERHEAD,
    "franka_1rgb":                ViewType.OVERHEAD,
    "franka_fr3_dual":            ViewType.PRIMARY,
    "tienkung_gello_1rgb":        ViewType.OVERHEAD,
    "tienkung_prod1_gello_1rgb":  ViewType.OVERHEAD,
    "tienkung_xsens_1rgb":        ViewType.OVERHEAD,
    "ur_1rgb":                    ViewType.OVERHEAD,
}

# Preferred third-person views (no wrist)
_THIRD_PERSON_VIEWS = ["camera_top", "camera_front"]


class RobomindV1Config(BaseDatasetConfig):
    dataset_name = "robomind_v1.0"
    stages = _TWO_STAGES

    def get_active_stages(self, sample: Dict[str, Any]) -> List[StageDefinition]:
        if sample.get("steps_raw"):
            return _STEP_STAGES
        return _TWO_STAGES

    def resolve_stage_views(
        self,
        sample: Dict[str, Any],
        video_base_dir: str = "",
    ) -> Dict[str, Dict[str, str]]:
        views: List[str] = sample.get("views", [])
        videos: Dict[str, str] = sample.get("videos", {})

        active = self.get_active_stages(sample)
        stage_names = [sd.name for sd in active]

        if not views or not videos:
            return {n: {"view": "", "video_path": ""} for n in stage_names}

        base = video_base_dir or _VIDEO_BASE_DIR
        dataset_dir = sample.get("dataset_dir", "")

        def _resolve(view_name: str) -> Dict[str, str]:
            return {
                "view": view_name,
                "video_path": os.path.join(base, dataset_dir, videos[view_name]),
            }

        # Find available third-person views
        available_tp = [v for v in _THIRD_PERSON_VIEWS if v in videos]
        first = views[0]
        rng = random.Random(stable_hash(sample.get("sample_id", "")))

        if sample.get("steps_raw"):
            # Single stage: randomly pick one third-person view
            if available_tp:
                v = rng.choice(available_tp)
            else:
                v = first
            return {"refinement": _resolve(v)}

        # 2-stage: if both top and front available, randomly assign order
        if len(available_tp) >= 2:
            rng.shuffle(available_tp)
            v_analysis = available_tp[0]
            v_refinement = available_tp[1]
        elif available_tp:
            v_analysis = v_refinement = available_tp[0]
        else:
            v_analysis = v_refinement = first

        return {
            "analysis":   _resolve(v_analysis),
            "refinement": _resolve(v_refinement),
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
        rt = sample.get("robot_type", "")
        return _VIEW_TYPE.get(rt, ViewType.OVERHEAD).value
