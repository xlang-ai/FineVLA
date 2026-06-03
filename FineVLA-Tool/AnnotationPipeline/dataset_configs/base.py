"""
Base class for per-dataset annotation pipeline configuration.

Each dataset can subclass BaseDatasetConfig to define:
  - How many stages the pipeline has and what they're called
  - Which stage function to run for each stage
  - Which camera views to use at each stage
  - View selection strategy (fixed, random, paired-shuffle, etc.)
  - Optional prompt overrides per stage

The runner will iterate over `config.stages` (a list of StageDefinition) and
call the corresponding stage function for each one.
"""

import hashlib
import os
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from config import (
    VIDEO_BASE_DIR as _VIDEO_BASE_DIR,
    DEFAULT_ANALYSIS_FPS,
    DEFAULT_REFINEMENT_FPS,
)


def stable_hash(s: str) -> int:
    """Deterministic hash that is consistent across Python processes and runs.

    Unlike built-in ``hash()``, this is not affected by ``PYTHONHASHSEED``.
    """
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)


class ViewType(Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    OVERHEAD = "overhead"
    WRIST = "wrist"
    FISHEYE = "fisheye"
    HEAD = "head"


@dataclass
class StageViewConfig:
    """Configuration for view selection at a single pipeline stage."""
    candidates: List[str]
    strategy: str = "fixed"  # "fixed" = first available, "random" = random pick


@dataclass
class StageDefinition:
    """
    Describes one stage of the annotation pipeline.

    Attributes:
        name:      Stage name (used as key in stage_views, stage_results, etc.)
        fn_name:   Key in stages.STAGE_REGISTRY that maps to the callable.
        depends_on: Name of a preceding stage whose result is required.
                    If the dependency failed, this stage is automatically skipped.
        default_fps:         Default frames-per-second for video sampling.
        default_sample_ratio: If set, use ratio-based sampling instead of FPS.
    """
    name: str
    fn_name: str
    depends_on: Optional[str] = None
    default_fps: float = 4.0
    default_sample_ratio: Optional[float] = None


# Default pipeline: two stages (analysis → refinement).
DEFAULT_STAGES = [
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


class BaseDatasetConfig:
    """
    Base configuration for a dataset's annotation pipeline.

    Subclass this and override:
      - `stages`              : ordered list of StageDefinition
      - `resolve_stage_views` : per-stage view/video resolution
      - `get_prompt_override` : per-stage prompt overrides
    """

    dataset_name: str = ""

    stages: List[StageDefinition] = DEFAULT_STAGES

    # Fallback view candidates when no stage-specific config is set.
    default_view_candidates: List[str] = []

    # Per-stage view config. Keys must match stage names in `self.stages`.
    stage_configs: Dict[str, StageViewConfig] = {}

    def resolve_stage_views(
        self,
        sample: Dict[str, Any],
        video_base_dir: str = "",
    ) -> Dict[str, Dict[str, str]]:
        """
        Determine which view and video path to use for each pipeline stage.

        Returns a dict keyed by stage name:
            {
                "<stage_name>": {"view": "cam_name", "video_path": "/abs/path.mp4"},
                ...
            }

        Default: pick a single best view and reuse it for all stages.
        """
        views: List[str] = sample.get("views", [])
        videos: Dict[str, str] = sample.get("videos", {})

        stage_names = [sd.name for sd in self.stages]

        if not views or not videos:
            return {name: {"view": "", "video_path": ""} for name in stage_names}

        base = video_base_dir or _VIDEO_BASE_DIR
        dataset_dir = sample.get("dataset_dir", "")

        result = {}
        for sd in self.stages:
            cfg = self.stage_configs.get(sd.name)
            candidates = cfg.candidates if cfg else self.default_view_candidates
            strategy = cfg.strategy if cfg else "fixed"

            available = [c for c in candidates if c in videos]

            if not available:
                available = self._keyword_fallback(views)

            if not available:
                available = [views[0]]

            if strategy == "random":
                seed = stable_hash(sample.get("sample_id", "")) + stable_hash(sd.name)
                chosen = random.Random(seed).choice(available)
            else:
                chosen = available[0]

            result[sd.name] = {
                "view": chosen,
                "video_path": os.path.join(base, dataset_dir, videos[chosen]),
            }

        return result

    def get_active_stages(
        self, sample: Dict[str, Any],
    ) -> List[StageDefinition]:
        """
        Return the stages that should actually run for this sample.

        Default: all stages defined in `self.stages`.
        Override when different samples within the same dataset need different
        stage counts (e.g. 1-view robots skip the 3rd stage).
        """
        return self.stages

    def classify_view(self, sample: Dict[str, Any]) -> str:
        """
        Return the ViewType string for this sample's primary view.

        Default: "primary". Subclasses (or DefaultConfig) may override using
        a (dataset, robot_type) lookup table.
        """
        return ViewType.PRIMARY.value

    def get_prompt_override(
        self,
        stage: str,
        sample: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[str, str]]:
        """
        Return (system_prompt, user_prompt_template) override for the given stage,
        or None to use the stage function's built-in defaults.

        ``sample`` is provided so configs that vary prompts by robot_type or
        other per-sample attributes can make that decision here.
        """
        return None

    @staticmethod
    def _keyword_fallback(views: List[str]) -> List[str]:
        """Pick views by keyword priority when explicit candidates miss."""
        priority = ["front", "high", "head", "main", "exterior", "top"]
        for kw in priority:
            matches = [v for v in views if kw in v.lower()]
            if matches:
                return matches
        return []
