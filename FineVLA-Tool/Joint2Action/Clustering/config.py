"""
VLA 轨迹聚类系统 —— 数据集配置。

每个数据集族定义一个 DatasetConfig，描述：
  - 目录结构（是否有子数据集、层级深度）
  - 轨迹字段映射（parquet 列名、拼接方式）
  - 姿态表示类型（quaternion / euler / none）
  - DTW 权重与聚类默认参数
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ArmConfig:
    """单臂的字段映射配置。"""
    eef_columns: list[str] = field(default_factory=list)
    gripper_columns: list[str] = field(default_factory=list)
    joint_columns: list[str] = field(default_factory=list)

    pos_indices: list[int] = field(default_factory=list)
    rot_indices: list[int] = field(default_factory=list)
    grip_indices: list[int] = field(default_factory=list)
    joint_indices: list[int] = field(default_factory=list)


@dataclass
class DatasetConfig:
    """一个数据集族的完整配置。"""
    dataset_name: str
    dataset_path: str

    # ── 目录结构 ──
    has_sub_datasets: bool = True
    sub_dataset_depth: int = 1

    # ── 姿态表示 ──
    rot_type: str = "quaternion"  # "quaternion" | "euler" | "rotation_vector" | “none” 表示纯 joint 空间，无笛卡尔空间 EEF

    # ── 单臂 / 双臂 ──
    available_sides: list[str] = field(default_factory=lambda: ["right"])
    arms: dict[str, ArmConfig] = field(default_factory=dict)

    # ── DTW 权重 ──
    w_pos: float = 1.0
    w_rot: float = 1.0
    w_grip: float = 100.0

    # ── 特征归一化 & 两阶段 DTW ──
    scale_features: bool = True   # 对 position/gripper 做 min-max 归一化到 [0,1]
    two_stage: bool = True        # 两阶段 DTW: EEF 对齐路径 + gripper 沿路径统计

    # ── 聚类默认参数 ──
    n_clusters: int = 0  # 0 = 自动选择最佳 k
    normalize: bool = True  # DTW 距离按路径长度归一化
    n_jobs: int = 8
    min_rel_gap: float = 0.3  # 递归聚类子层级 auto-k 的最小 rel_gap 阈值

    # ── 运行默认参数 ──
    side: str = ""  # 默认分析哪侧: "right"/"left"/"both"/""(空=用 available_sides[0])
    max_depth: int = 2  # 递归聚类最大深度
    min_cluster_size: int = 5  # 递归聚类停止的最小簇大小

    # ── 其他 ──
    has_modality_json: bool = False
    filter_report_path: str = ""  # 数据质量过滤报告路径
    skip: bool = False
    notes: str = ""


# ═══════════════════════════════════════════════════════════
#  数据集根目录
# ═══════════════════════════════════════════════════════════
DATA_ROOT = "/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21"

# ═══════════════════════════════════════════════════════════
#  各数据集配置
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
    notes="ALOHA 双臂，纯关节空间，无独立 EEF 列",
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
    notes="eef orientation 为 3D，假设是欧拉角；gripper 使用 eef_direction",
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
    notes="action shape=[7] [x,y,z,roll,pitch,yaw,gripper]；单臂",
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
    notes="state=[x,y,z,rx,ry,rz,rw,gripper] 四元数; flat ward k=20 推荐",
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
    notes="state=[8] [x,y,z,roll,pitch,yaw,pad,gripper]; flat ward k=20 推荐",
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
    notes="EEF 分两列存储: cartesian_position(6) + gripper_position(1)",
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
    notes="action=[joint_0..6, gripper]，纯关节空间",
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
    has_modality_json=True,  # 每个 robot_type 有独立的 modality.json
    filter_report_path=f"{DATA_ROOT}/RoboMindV2.0/RoboMindV2.0_filter_report.json",
    available_sides=["right", "left"],
    arms={
        # 注意：实际配置从 modality.json 动态加载
        # 这里提供默认模板，仅用于不支持动态加载的场景
        "right": ArmConfig(
            eef_columns=[],
            gripper_columns=[],  # 动态读取
            joint_columns=[],     # 动态读取
            joint_indices=[],     # 动态读取
            grip_indices=[],      # 动态读取
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
    notes="二级子目录结构 robot_type/task/；配置从 modality.json 动态加载，支持异构机器人（6/7维关节，gripper/hand）",
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
            # joint_indices / grip_indices 留空 → 自动推断：最后一列为 gripper，其余为 joints
        ),
        "right": ArmConfig(
            eef_columns=[],
            gripper_columns=[],
            joint_columns=["actions.joint_position_right"],
            # 自动推断索引
        ),
        "left": ArmConfig(
            eef_columns=[],
            gripper_columns=[],
            joint_columns=["actions.joint_position_left"],
            # 自动推断索引
        ),
    },
    w_pos=1.0, w_rot=1.0, w_grip=100.0,
    n_clusters=0, n_jobs=8,
    side="single",
    notes="三级子目录 benchmark/robot_type/task；异构机器人：单臂用 single，双臂 agilex 用 right+left",
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
    notes="state shape=[96] 复杂结构，action=[14] joints，暂时跳过",
)


# ═══════════════════════════════════════════════════════════
#  注册表：name → config
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
