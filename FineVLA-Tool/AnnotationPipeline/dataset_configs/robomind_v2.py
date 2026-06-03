"""
Dataset configuration for RoboMIND V2.0 (robomind_v2.0).

All samples: analysis → refinement (2-stage, no steps_raw).

View strategy:
  When both camera_top and camera_front are available, randomly assign
  one to analysis and the other to refinement (seeded by sample_id).
  Falls back to first available view if neither exists.
"""

import os
import random
from typing import Any, Dict, List, Optional, Tuple

from config import VIDEO_BASE_DIR as _VIDEO_BASE_DIR
from .base import BaseDatasetConfig, StageDefinition, ViewType, stable_hash

_ANALYSIS_FPS   = 3.0
_REFINEMENT_FPS = 5.0

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

_TWO_STAGES = [_STAGE_ANALYSIS, _STAGE_REFINEMENT]

_VIEW_TYPE = {
    "agilex":        ViewType.PRIMARY,
    "agilex_mobile": ViewType.PRIMARY,
    "ark":           ViewType.OVERHEAD,
    "ark_mobile":    ViewType.OVERHEAD,
    "franka":        ViewType.PRIMARY,
    "franka_sim":    ViewType.PRIMARY,
    "tienkung":      ViewType.OVERHEAD,
    "tienkung_sim":  ViewType.HEAD,
    "tienyi":        ViewType.OVERHEAD,
    "tienyi_mobile": ViewType.OVERHEAD,
    "ur":            ViewType.PRIMARY,
    "ur_dex":        ViewType.PRIMARY,
}

# Preferred third-person views (no wrist)
_THIRD_PERSON_VIEWS = ["camera_top", "camera_front"]


class RobomindV2Config(BaseDatasetConfig):
    dataset_name = "robomind_v2.0"
    stages = _TWO_STAGES

    def get_active_stages(self, sample: Dict[str, Any]) -> List[StageDefinition]:
        return _TWO_STAGES

    def resolve_stage_views(
        self,
        sample: Dict[str, Any],
        video_base_dir: str = "",
    ) -> Dict[str, Dict[str, str]]:
        views: List[str] = sample.get("views", [])
        videos: Dict[str, str] = sample.get("videos", {})

        if not views or not videos:
            return {"analysis": {"view": "", "video_path": ""},
                    "refinement": {"view": "", "video_path": ""}}

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

        # If both top and front available, randomly assign order
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
        return None

    def classify_view(self, sample: Dict[str, Any]) -> str:
        rt = sample.get("robot_type", "")
        return _VIEW_TYPE.get(rt, ViewType.PRIMARY).value
