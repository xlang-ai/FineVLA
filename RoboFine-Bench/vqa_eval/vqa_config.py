"""
VQA dataset configuration: per-dataset view selection, FPS, and video directory.

Each dataset specifies:
  - views:  list of camera views to use for VQA (None = dynamic from sample meta)
  - fps:    frame sampling rate (unified to 2.0)

For datasets with variable views (RoboCoin, RoboMINDV1, RoboMINDV2),
views are read from EvalSets.json meta.view_names at runtime.
"""

DATASET_CONFIGS = {
    "BridgeDataV2": {
        "views": ["image_0"],
        "fps": 2.0,
    },
    "BC-Z": {
        "views": ["image"],
        "fps": 2.0,
    },
    "RT-1": {
        "views": ["image"],
        "fps": 2.0,
    },
    "DROID-Robointer": {
        "views": ["primary", "wrist"],
        "fps": 2.0,
    },
    "RH20T-RoboInter": {
        "views": ["primary", "wrist"],
        "fps": 2.0,
    },
    "RDT": {
        "views": ["main", "left_wrist", "right_wrist"],
        "fps": 2.0,
    },
    # vqa_frame_index only record limited 500 frames for one sample 
    "Galaxea": {
        "views": ["head_rgb", "left_wrist_rgb", "right_wrist_rgb"],
        "fps": 2.0,
    },
    "RoboCoin": {
        "views": None,  # dynamic from meta.view_names
        "fps": 2.0,
    },
    "RoboMINDV1": {
        "views": None,  # dynamic from meta.view_names
        "fps": 2.0,
    },
    "RoboMINDV2": {
        "views": None,  # dynamic from meta.view_names
        "fps": 2.0,
    },
}

# Legacy dataset name mapping (old name -> new name)
_LEGACY_MAP = {
    "droid_1.0.1": "DROID-Robointer",
    "bridge": "BridgeDataV2",
    "bc_z": "BC-Z",
    "rt_1": "RT-1",
    "galaxea_open_world": "Galaxea",
}

DEFAULT_FPS = 2.0


def _resolve_dataset(dataset: str) -> str:
    """Resolve legacy dataset names to current names."""
    return _LEGACY_MAP.get(dataset, dataset)


def get_views(dataset: str, sample_meta: dict = None) -> list:
    """Return the view name list for a dataset.

    For datasets with None views (variable views per sample),
    falls back to sample_meta['view_names'].
    """
    ds = _resolve_dataset(dataset)
    cfg = DATASET_CONFIGS.get(ds)
    if cfg and cfg.get("views"):
        return cfg["views"]
    # Dynamic: read from sample meta
    if sample_meta and sample_meta.get("view_names"):
        return sample_meta["view_names"]
    return []


def get_view(dataset: str) -> str:
    """Legacy: return the first view name for a dataset."""
    views = get_views(dataset)
    return views[0] if views else None


def get_fps(dataset: str) -> float:
    """Return the FPS for a dataset."""
    ds = _resolve_dataset(dataset)
    cfg = DATASET_CONFIGS.get(ds)
    return cfg["fps"] if cfg else DEFAULT_FPS


def get_dataset_dir(dataset: str) -> str:
    """Return the fallback dataset_dir for a dataset (legacy compat)."""
    return ""
