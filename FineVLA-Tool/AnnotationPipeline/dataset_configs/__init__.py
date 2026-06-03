"""
Per-dataset configuration registry.

Usage:
    from dataset_configs import get_config, BaseDatasetConfig, StageDefinition

    cfg = get_config("droid_1.0.1")   # → DroidConfig
    cfg = get_config("robomind_v1.0") # → DefaultConfig (fallback)

    # get_config() always returns a BaseDatasetConfig — never None.

To add a new dataset:
    1. Create dataset_configs/<name>.py, subclass BaseDatasetConfig
    2. Override `stages` with your own List[StageDefinition]
    3. Override `resolve_stage_views` for custom view assignment
    4. Register it in DATASET_CONFIGS below
"""

from .base import (
    BaseDatasetConfig,
    StageDefinition,
    StageViewConfig,
    ViewType,
    DEFAULT_STAGES,
)
from .default import DefaultConfig
from .droid import DroidConfig
from .rdt import RdtConfig
from .robomind_v2 import RobomindV2Config
from .galaxea import GalaxeaConfig
from .agibotworld import AgiBotWorldConfig
from .robomind_v1 import RobomindV1Config
from .rh20t import Rh20tConfig
from .robocoin import RobocoinConfig
from .bc_z import BcZConfig
from .bridge import BridgeConfig
from .rt_1 import Rt1Config


_DEFAULT_CONFIG = DefaultConfig()

DATASET_CONFIGS = {
    "droid_1.0.1": DroidConfig(),
    "droid_robointer": DroidConfig(),
    "rh20t_robointer": Rh20tConfig(),
    "robocoin": RobocoinConfig(),
    "rdt": RdtConfig(),
    "rh20t": Rh20tConfig(),
    "robomind_v1.0": RobomindV1Config(),
    "robomind_v2.0": RobomindV2Config(),
    "galaxea_open_world": GalaxeaConfig(),
    "agibotworld": AgiBotWorldConfig(),
    "bc_z": BcZConfig(),
    "bridge": BridgeConfig(),
    "rt_1": Rt1Config(),
    "rt1": Rt1Config(),
}


def get_config(dataset_name: str) -> BaseDatasetConfig:
    """Return the DatasetConfig for the given dataset (never None)."""
    return DATASET_CONFIGS.get(dataset_name, _DEFAULT_CONFIG)


def has_dedicated_config(dataset_name: str) -> bool:
    """True if this dataset has a dedicated config (not DefaultConfig)."""
    return dataset_name in DATASET_CONFIGS
