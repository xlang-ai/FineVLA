"""
VQA dataset configuration: per-dataset view selection.

Each dataset specifies:
  - views:  list of camera views to use for VQA (None = dynamic from sample meta)

For datasets with variable views (RoboCoin, RoboMINDV1, RoboMINDV2),
views are read from EvalSets.json meta.view_names at runtime.

FPS is controlled by the --fps CLI argument (default 2.0).
"""

DATASET_CONFIGS = {
    "BridgeDataV2": {
        "views": ["image_0"],
    },
    "BC-Z": {
        "views": ["image"],
    },
    "RT-1": {
        "views": ["image"],
    },
    "DROID-Robointer": {
        "views": ["primary", "wrist"],
    },
    "RH20T-RoboInter": {
        "views": ["primary", "wrist"],
    },
    "RDT": {
        "views": ["main", "left_wrist", "right_wrist"],
    },
    "Galaxea": {
        "views": ["head_rgb", "left_wrist_rgb", "right_wrist_rgb"],
    },
    "RoboCoin": {
        "views": None,  # dynamic from meta.view_names
    },
    "RoboMINDV1": {
        "views": None,  # dynamic from meta.view_names
    },
    "RoboMINDV2": {
        "views": None,  # dynamic from meta.view_names
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


def get_dataset_dir(dataset: str) -> str:
    """Return the fallback dataset_dir for a dataset (legacy compat)."""
    return ""
