"""
Dataset field configuration for different robot dataset families.

Only datasets that need explicit field selection are listed here.
All other datasets use auto-discovery from info.json features.
"""

DATASET_CONFIGS = {
    "galaxea": {
        "match_keyword": "Galaxea-Open-World-Dataset",
        "state_fields": [
            "observation.state.left_arm",
            "observation.state.left_arm.velocities",
            "observation.state.right_arm",
            "observation.state.right_arm.velocities",
            "observation.state.chassis.imu",
            "observation.state.chassis",
            "observation.state.torso",
            "observation.state.torso.velocities",
            "observation.state.left_gripper",
            "observation.state.right_gripper",
            "observation.state.left_ee_pose",
            "observation.state.right_ee_pose",
        ],
        "action_fields": [
            "action.left_gripper",
            "action.right_gripper",
            "action.chassis.velocities",
            "action.torso.velocities",
            "action.left_arm",
            "action.right_arm",
        ],
    },
}

# Known dataset family keywords for display name detection
FAMILY_KEYWORDS = [
    ("Galaxea-Open-World-Dataset", "galaxea"),
    ("RoboCOIN_add0130", "robocoin_add0130"),
    ("RoboCOIN_add1201", "robocoin_add1201"),
    ("RoboCOIN", "robocoin"),
    ("RoboMindV1", "robomind_v1"),
    ("RoboMindV2", "robomind_v2"),
    ("RDT-yhq", "rdt"),
    ("RH20T-fjy", "rh20t"),
    ("agibotworld", "agibotworld"),
    ("Bridge", "bridge"),
    ("RT-1", "rt1"),
    ("BC_Z", "bc_z"),
    ("droid_1.0.1", "droid"),
    ("egodex_train_robot", "egodex"),
    ("xvla-soft-fold", "xvla"),
]


def match_dataset_config(parquet_path: str) -> tuple[str, dict | None]:
    """Match a parquet path to its dataset config.

    Returns (dataset_family_name, config_dict_or_None).
    """
    for family, cfg in DATASET_CONFIGS.items():
        if cfg["match_keyword"] in parquet_path:
            return family, cfg

    for keyword, family_name in FAMILY_KEYWORDS:
        if keyword in parquet_path:
            return family_name, None

    return "unknown", None
