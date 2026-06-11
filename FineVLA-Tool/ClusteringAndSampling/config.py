"""
VLA Trajectory Clustering System -- Dataset Configuration.

Each dataset family defines a DatasetConfig describing:
  - Directory structure (whether it has sub-datasets, hierarchy depth)
  - Trajectory field mapping (parquet column names, concatenation method)
  - Pose representation type (quaternion / euler / none)
  - DTW weights and default clustering parameters
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ArmConfig:
    """Single arm field mapping configuration."""
    eef_columns: list[str] = field(default_factory=list)
    gripper_columns: list[str] = field(default_factory=list)
    joint_columns: list[str] = field(default_factory=list)

    pos_indices: list[int] = field(default_factory=list)
    rot_indices: list[int] = field(default_factory=list)
    grip_indices: list[int] = field(default_factory=list)
    joint_indices: list[int] = field(default_factory=list)


@dataclass
class DatasetConfig:
    """Complete configuration for a dataset family."""
    dataset_name: str
    dataset_path: str

    # ── Directory Structure ──
    has_sub_datasets: bool = True
    sub_dataset_depth: int = 1

    # ── Pose Representation ──
    rot_type: str = "quaternion"  # "quaternion" | "euler" | "rotation_vector" | "none" means pure joint space, no Cartesian EEF

    # ── Single-arm / Dual-arm ──
    available_sides: list[str] = field(default_factory=lambda: ["right"])
    arms: dict[str, ArmConfig] = field(default_factory=dict)

    # ── DTW Weights ──
    w_pos: float = 1.0
    w_rot: float = 1.0
    w_grip: float = 100.0

    # ── Feature Normalization & Two-stage DTW ──
    scale_features: bool = True   # min-max normalize position/gripper to [0,1]
    two_stage: bool = True        # two-stage DTW: EEF alignment path + gripper statistics along the path

    # ── Clustering Default Parameters ──
    n_clusters: int = 0  # 0 = automatically select best k
    normalize: bool = True  # normalize DTW distance by path length
    n_jobs: int = 8
    min_rel_gap: float = 0.3  # minimum rel_gap threshold for recursive clustering sub-level auto-k

    # ── Runtime Default Parameters ──
    side: str = ""  # default side to analyze: "right"/"left"/"both"/""(empty=use available_sides[0])
    max_depth: int = 2  # maximum depth for recursive clustering
    min_cluster_size: int = 5  # minimum cluster size to stop recursive clustering

    # ── Other ──
    has_modality_json: bool = False
    filter_report_path: str = ""  # data quality filter report path
    skip: bool = False
    notes: str = ""


# ═══════════════════════════════════════════════════════════
#  Dataset Root Directory
# ═══════════════════════════════════════════════════════════
DATA_ROOT = os.environ.get("VLA_DATA_ROOT", "/path/to/your/Lerobot_v21")

# ═══════════════════════════════════════════════════════════
#  Dataset Configurations
# ═══════════════════════════════════════════════════════════

GALAXEA = DatasetConfig(
    dataset_name="Galaxea",
    dataset_path=f"{DATA_ROOT}/Galaxea-Open-World-Dataset",
    has_sub_datasets=True,
    sub_dataset_depth=1,
    rot_type="quaternion",
    has_modality_json=True,
    available_sides=["right", "left"],
    arms={
        "right": ArmConfig(
            eef_columns=["observation.state.right_ee_pose"],
            gripper_columns=["action.right_gripper"],
            joint_columns=["observation.state.right_arm"],
            pos_indices=[0, 1, 2],
            rot_indices=[3, 4, 5, 6],
            grip_indices=[7],
        ),
        "left": ArmConfig(
            eef_columns=["observation.state.left_ee_pose"],
            gripper_columns=["action.left_gripper"],
            joint_columns=["observation.state.left_arm"],
            pos_indices=[0, 1, 2],
            rot_indices=[3, 4, 5, 6],
            grip_indices=[7],
        ),
    },
    w_pos=1.0, w_rot=1.0, w_grip=100.0,
    n_clusters=0, n_jobs=16,
    min_rel_gap=0.03,
)

RDT = DatasetConfig(
    dataset_name="RDT",
    dataset_path=f"{DATA_ROOT}/RDT-yhq",
    has_sub_datasets=True,
    sub_dataset_depth=1,
    rot_type="none",
    has_modality_json=False,
    available_sides=["right", "left"],
    arms={
        "right": ArmConfig(
            eef_columns=[],
            gripper_columns=[],
            joint_columns=["action"],
            joint_indices=[7, 8, 9, 10, 11, 12],
            grip_indices=[13],
        ),
        "left": ArmConfig(
            eef_columns=[],
            gripper_columns=[],
            joint_columns=["action"],
            joint_indices=[0, 1, 2, 3, 4, 5],
            grip_indices=[6],
        ),
    },
    w_pos=1.0, w_rot=1.0, w_grip=100.0,
    n_clusters=4, n_jobs=16,
    side="both",
    max_depth=2,
    min_cluster_size=3,
    notes="ALOHA dual-arm, pure joint space, no separate EEF columns",
)

ROBOCOIN = DatasetConfig(
    dataset_name="RoboCOIN",
    dataset_path=f"{DATA_ROOT}/RoboCOIN",
    has_sub_datasets=True,
    sub_dataset_depth=1,
    rot_type="euler",
    has_modality_json=False,
    available_sides=["right", "left"],
    arms={
        "right": ArmConfig(
            eef_columns=["eef_sim_pose_state"],
            gripper_columns=["eef_direction_state"],
            pos_indices=[6, 7, 8],
            rot_indices=[9, 10, 11],
            grip_indices=[12],
        ),
        "left": ArmConfig(
            eef_columns=["eef_sim_pose_state"],
            gripper_columns=["eef_direction_state"],
            pos_indices=[0, 1, 2],
            rot_indices=[3, 4, 5],
            grip_indices=[6],
        ),
    },
    w_pos=1.0, w_rot=1.0, w_grip=100.0,
    n_clusters=0, n_jobs=96,
    notes="eef orientation is 3D, assumed to be Euler angles; gripper uses eef_direction",
)

ROBOCOIN_ADD = DatasetConfig(
    dataset_name="RoboCOIN_add0130",
    dataset_path=f"{DATA_ROOT}/RoboCOIN_add0130",
    has_sub_datasets=True,
    sub_dataset_depth=1,
    rot_type="euler",
    has_modality_json=False,
    available_sides=["right", "left"],
    arms={
        "right": ArmConfig(
            eef_columns=["eef_sim_pose_state"],
            gripper_columns=["eef_direction_state"],
            pos_indices=[6, 7, 8],
            rot_indices=[9, 10, 11],
            grip_indices=[12],
        ),
        "left": ArmConfig(
            eef_columns=["eef_sim_pose_state"],
            gripper_columns=["eef_direction_state"],
            pos_indices=[0, 1, 2],
            rot_indices=[3, 4, 5],
            grip_indices=[6],
        ),
    },
    w_pos=1.0, w_rot=1.0, w_grip=100.0,
    n_clusters=0, n_jobs=8,
)

ROBOCOIN_ADD1201 = DatasetConfig(
    dataset_name="RoboCOIN_add1201",
    dataset_path=f"{DATA_ROOT}/RoboCOIN_add1201",
    has_sub_datasets=True,
    sub_dataset_depth=1,
    rot_type="euler",
    has_modality_json=True,
    available_sides=["right", "left"],
    arms={
        "right": ArmConfig(
            eef_columns=["eef_sim_pose_state"],
            gripper_columns=["eef_direction_state"],
            pos_indices=[6, 7, 8],
            rot_indices=[9, 10, 11],
            grip_indices=[12],
        ),
        "left": ArmConfig(
            eef_columns=["eef_sim_pose_state"],
            gripper_columns=["eef_direction_state"],
            pos_indices=[0, 1, 2],
            rot_indices=[3, 4, 5],
            grip_indices=[6],
        ),
    },
    w_pos=1.0, w_rot=1.0, w_grip=100.0,
    n_clusters=0, n_jobs=32,
)

BRIDGE = DatasetConfig(
    dataset_name="Bridge",
    dataset_path=f"{DATA_ROOT}/Bridge",
    has_sub_datasets=False,
    rot_type="euler",
    has_modality_json=False,
    available_sides=["single"],
    arms={
        "single": ArmConfig(
            eef_columns=["action"],
            gripper_columns=[],
            pos_indices=[0, 1, 2],
            rot_indices=[3, 4, 5],
            grip_indices=[6],
        ),
    },
    w_pos=1.0, w_rot=1.0, w_grip=100.0,
    n_clusters=0, n_jobs=8,
    notes="action shape=[7] [x,y,z,roll,pitch,yaw,gripper]; single-arm",
)

RT1 = DatasetConfig(
    dataset_name="RT-1",
    dataset_path=f"{DATA_ROOT}/RT-1",
    has_sub_datasets=False,
    rot_type="quaternion",
    has_modality_json=False,
    available_sides=["single"],
    arms={
        "single": ArmConfig(
            eef_columns=["observation.state"],
            gripper_columns=[],
            pos_indices=[0, 1, 2],
            rot_indices=[3, 4, 5, 6],
            grip_indices=[7],
        ),
    },
    w_pos=1.0, w_rot=1.0, w_grip=1.0,
    n_clusters=20, n_jobs=64,
    min_rel_gap=0.0,
    max_depth=5,
    min_cluster_size=3,
    notes="state=[x,y,z,rx,ry,rz,rw,gripper] quaternion; flat ward k=20 recommended",
)

BC_Z = DatasetConfig(
    dataset_name="BC_Z",
    dataset_path=f"{DATA_ROOT}/BC_Z",
    has_sub_datasets=False,
    rot_type="euler",
    has_modality_json=False,
    available_sides=["single"],
    arms={
        "single": ArmConfig(
            eef_columns=["observation.state"],
            gripper_columns=[],
            pos_indices=[0, 1, 2],
            rot_indices=[3, 4, 5],
            grip_indices=[7],
        ),
    },
    w_pos=1.0, w_rot=1.0, w_grip=1.0,
    n_clusters=20, n_jobs=8,
    min_rel_gap=0.0,
    max_depth=5,
    min_cluster_size=3,
    notes="state=[8] [x,y,z,roll,pitch,yaw,pad,gripper]; flat ward k=20 recommended",
)

DROID = DatasetConfig(
    dataset_name="droid",
    dataset_path=f"{DATA_ROOT}/droid_1.0.1",
    has_sub_datasets=False,
    rot_type="euler",
    has_modality_json=False,
    available_sides=["single"],
    arms={
        "single": ArmConfig(
            eef_columns=["observation.state.cartesian_position"],
            gripper_columns=["observation.state.gripper_position"],
            pos_indices=[0, 1, 2],
            rot_indices=[3, 4, 5],
            grip_indices=[6],
        ),
    },
    w_pos=1.0, w_rot=1.0, w_grip=100.0,
    n_clusters=0, n_jobs=8,
    notes="EEF stored in two columns: cartesian_position(6) + gripper_position(1)",
)

DROID_ROBOINTER = DatasetConfig(
    dataset_name="droid_RoboInter",
    dataset_path=f"{DATA_ROOT}/droid_RoboInter",
    has_sub_datasets=False,
    rot_type="euler",
    has_modality_json=False,
    filter_report_path=f"{DATA_ROOT}/droid_RoboInter/droid_RoboInter_filter_report.json",
    available_sides=["single"],
    arms={
        "single": ArmConfig(
            eef_columns=["state"],
            gripper_columns=[],
            pos_indices=[0, 1, 2],
            rot_indices=[3, 4, 5],
            grip_indices=[6],
        ),
    },
    w_pos=1.0, w_rot=1.0, w_grip=100.0,
    n_clusters=0, n_jobs=96,
    min_rel_gap=0.0,
    max_depth=5,
    min_cluster_size=3,
    notes="state=[x,y,z,rx,ry,rz,gripper]; 152986 episodes, 43026 tasks; Franka + Robotiq",
)

EGODEX = DatasetConfig(
    dataset_name="egodex",
    dataset_path=f"{DATA_ROOT}/egodex_train_robot_yhq",
    has_sub_datasets=False,
    rot_type="euler",
    has_modality_json=False,
    available_sides=["right", "left"],
    arms={
        "right": ArmConfig(
            eef_columns=["action"],
            gripper_columns=[],
            pos_indices=[6, 7, 8],
            rot_indices=[9, 10, 11],
            grip_indices=[13],
        ),
        "left": ArmConfig(
            eef_columns=["action"],
            gripper_columns=[],
            pos_indices=[0, 1, 2],
            rot_indices=[3, 4, 5],
            grip_indices=[12],
        ),
    },
    w_pos=1.0, w_rot=1.0, w_grip=100.0,
    n_clusters=0, n_jobs=8,
    notes="action=[left_xyz+rpy(6), right_xyz+rpy(6), grip_l, grip_r] world frame",
)

RH20T = DatasetConfig(
    dataset_name="RH20T",
    dataset_path=f"{DATA_ROOT}/RH20T-fjy",
    has_sub_datasets=True,
    sub_dataset_depth=1,
    rot_type="none",
    has_modality_json=False,
    available_sides=["single"],
    arms={
        "single": ArmConfig(
            eef_columns=[],
            gripper_columns=[],
            joint_columns=["action"],
            joint_indices=[0, 1, 2, 3, 4, 5, 6],
            grip_indices=[7],
        ),
    },
    w_pos=1.0, w_rot=1.0, w_grip=100.0,
    n_clusters=0, n_jobs=8,
    notes="action=[joint_0..6, gripper], pure joint space",
)

RH20T_ROBOINTER = DatasetConfig(
    dataset_name="RH20T-RoboInter",
    dataset_path=f"{DATA_ROOT}/RH20T-RoboInter",
    has_sub_datasets=False,
    rot_type="euler",
    has_modality_json=False,
    filter_report_path=f"{DATA_ROOT}/RH20T-RoboInter/RH20T-RoboInter_filter_report.json",
    available_sides=["single"],
    arms={
        "single": ArmConfig(
            eef_columns=["state"],
            gripper_columns=[],
            pos_indices=[0, 1, 2],
            rot_indices=[3, 4, 5],
            grip_indices=[6],
        ),
    },
    w_pos=1.0, w_rot=1.0, w_grip=100.0,
    n_clusters=0, n_jobs=32,
    min_rel_gap=0.0,
    max_depth=5,
    min_cluster_size=3,
    notes="state=[x,y,z,rx,ry,rz,gripper]; 82894 episodes, 146 tasks",
)

AGIBOTWORLD = DatasetConfig(
    dataset_name="agibotworld",
    dataset_path=f"{DATA_ROOT}/agibotworld_hyy",
    has_sub_datasets=True,
    sub_dataset_depth=1,
    rot_type="quaternion",
    has_modality_json=False,
    available_sides=["right", "left"],
    arms={
        "right": ArmConfig(
            eef_columns=[
                "observation.states.end.position",
                "observation.states.end.orientation",
            ],
            gripper_columns=["actions.effector.position"],
            pos_indices=[3, 4, 5],       # right_x/y/z in end.position(6)
            rot_indices=[6 + 4, 6 + 5, 6 + 6, 6 + 7],  # right quat in end.orientation(8)
            grip_indices=[14 + 1],        # right gripper in effector.position(2)
        ),
        "left": ArmConfig(
            eef_columns=[
                "observation.states.end.position",
                "observation.states.end.orientation",
            ],
            gripper_columns=["actions.effector.position"],
            pos_indices=[0, 1, 2],
            rot_indices=[6, 7, 8, 9],
            grip_indices=[14],
        ),
    },
    w_pos=1.0, w_rot=1.0, w_grip=100.0,
    n_clusters=0, n_jobs=8,
    notes="end.position=[l_xyz(3)+r_xyz(3)], end.orientation=[l_xyzw(4)+r_xyzw(4)], effector=[l_grip,r_grip]",
)

ROBOMIND_V2 = DatasetConfig(
    dataset_name="RoboMindV2.0",
    dataset_path=f"{DATA_ROOT}/RoboMindV2.0",
    has_sub_datasets=True,
    sub_dataset_depth=2,
    rot_type="none",
    has_modality_json=True,  # each robot_type has its own modality.json
    filter_report_path=f"{DATA_ROOT}/RoboMindV2.0/RoboMindV2.0_filter_report.json",
    available_sides=["right", "left"],
    arms={
        # Note: actual configuration is dynamically loaded from modality.json
        # The defaults here serve as a template, only used when dynamic loading is not supported
        "right": ArmConfig(
            eef_columns=[],
            gripper_columns=[],  # dynamically loaded
            joint_columns=[],     # dynamically loaded
            joint_indices=[],     # dynamically loaded
            grip_indices=[],      # dynamically loaded
        ),
        "left": ArmConfig(
            eef_columns=[],
            gripper_columns=[],
            joint_columns=[],
            joint_indices=[],
            grip_indices=[],
        ),
    },
    w_pos=1.0, w_rot=1.0, w_grip=100.0,
    n_clusters=0, n_jobs=8,
    notes="Two-level sub-directory structure robot_type/task/; config dynamically loaded from modality.json, supports heterogeneous robots (6/7-dim joints, gripper/hand)",
)
ROBOMIND_V1 = DatasetConfig(
    dataset_name="RoboMindV1.0",
    dataset_path=f"{DATA_ROOT}/RoboMindV1.0",
    has_sub_datasets=True,
    sub_dataset_depth=3,
    rot_type="none",
    has_modality_json=False,
    available_sides=["single", "right", "left"],
    arms={
        "single": ArmConfig(
            eef_columns=[],
            gripper_columns=[],
            joint_columns=["actions.joint_position"],
            # joint_indices / grip_indices left empty -> auto-inferred: last column is gripper, rest are joints
        ),
        "right": ArmConfig(
            eef_columns=[],
            gripper_columns=[],
            joint_columns=["actions.joint_position_right"],
            # auto-infer indices
        ),
        "left": ArmConfig(
            eef_columns=[],
            gripper_columns=[],
            joint_columns=["actions.joint_position_left"],
            # auto-infer indices
        ),
    },
    w_pos=1.0, w_rot=1.0, w_grip=100.0,
    n_clusters=0, n_jobs=8,
    side="single",
    notes="Three-level sub-directory benchmark/robot_type/task; heterogeneous robots: single-arm uses single, dual-arm agilex uses right+left",
)

XVLA = DatasetConfig(
    dataset_name="xvla",
    dataset_path=f"{DATA_ROOT}/xvla-soft-fold_franka_v3_franka",
    has_sub_datasets=False,
    rot_type="none",
    has_modality_json=False,
    available_sides=["single"],
    arms={
        "single": ArmConfig(
            eef_columns=[],
            gripper_columns=[],
            joint_columns=["action"],
            joint_indices=[0, 1, 2, 3, 4, 5, 6],
            grip_indices=[7],
        ),
    },
    skip=True,
    notes="state shape=[96] complex structure, action=[14] joints, skipped for now",
)


# ═══════════════════════════════════════════════════════════
#  Registry: name -> config
# ═══════════════════════════════════════════════════════════

DATASET_REGISTRY: dict[str, DatasetConfig] = {
    "Galaxea": GALAXEA,
    "RDT": RDT,
    "RoboCOIN": ROBOCOIN,
    "RoboCOIN_add0130": ROBOCOIN_ADD,
    "RoboCOIN_add1201": ROBOCOIN_ADD1201,
    "Bridge": BRIDGE,
    "RT-1": RT1,
    "BC_Z": BC_Z,
    "droid": DROID,
    "droid_RoboInter": DROID_ROBOINTER,
    "egodex": EGODEX,
    "RH20T": RH20T,
    "RH20T-RoboInter": RH20T_ROBOINTER,
    "agibotworld": AGIBOTWORLD,
    "RoboMindV1.0": ROBOMIND_V1,
    "RoboMindV2.0": ROBOMIND_V2,
    "xvla": XVLA,
}


def get_config(name: str) -> DatasetConfig:
    if name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    return DATASET_REGISTRY[name]


def list_configs(include_skip: bool = False) -> list[DatasetConfig]:
    return [c for c in DATASET_REGISTRY.values() if include_skip or not c.skip]
