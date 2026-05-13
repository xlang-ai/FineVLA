# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#


from abc import ABC, abstractmethod

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform, ModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.concat import ConcatTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
)
from starVLA.dataloader.gr00t_lerobot.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
# from gr00t.model.transforms import GR00TTransform


class BaseDataConfig(ABC):
    @abstractmethod
    def modality_config(self) -> dict[str, ModalityConfig]:
        pass

    @abstractmethod
    def transform(self) -> ModalityTransform:
        pass


###########################################################################################

class OxeDroidDataConfig:
    video_keys = [
        "video.exterior_image_1",
        "video.exterior_image_2",
        "video.wrist_image",
    ]
    state_keys = [
        "state.eef_position",
        "state.eef_rotation",
        "state.gripper_position",
    ]
    action_keys = [
        "action.eef_position_delta",
        "action.eef_rotation_delta",
        "action.gripper_position",
    ]
    language_keys = ["annotation.language.language_instruction"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.eef_position": "min_max",
                    "state.gripper_position": "min_max",
                },
                target_rotations={
                    "state.eef_rotation": "rotation_6d",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.gripper_position": "binary",
                },
                target_rotations={"action.eef_rotation_delta": "axis_angle"},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class OxeBridgeDataConfig:
    video_keys = [
        "video.image_0",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.pad",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.x": "q99",
                    "state.y": "q99",
                    "state.z": "q99",
                    "state.roll": "q99",
                    "state.pitch": "q99",
                    "state.yaw": "q99",
                    "state.pad": "q99",
                    "state.gripper": "binary",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "q99",
                    "action.y": "q99",
                    "action.z": "q99",
                    "action.roll": "q99",
                    "action.pitch": "q99",
                    "action.yaw": "q99",
                    "action.gripper": "binary",
                },
            ),
            # concat transforms
            # ConcatTransform(
            #     # video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
            # GR00TTransform(
            #     state_horizon=len(self.observation_indices),
            #     action_horizon=len(self.action_indices),
            #     max_state_dim=64,
            #     max_action_dim=32,
            # ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################

class OxeRT1DataConfig:
    video_keys = [
        "video.image",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.rx",
        "state.ry",
        "state.rz",
        "state.rw",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.x": "q99",
                    "state.y": "q99",
                    "state.z": "q99",
                    "state.rx": "q99",
                    "state.ry": "q99",
                    "state.rz": "q99",
                    "state.rw": "q99",
                    "state.gripper": "binary",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "q99",
                    "action.y": "q99",
                    "action.z": "q99",
                    "action.roll": "q99",
                    "action.pitch": "q99",
                    "action.yaw": "q99",
                    "action.gripper": "binary",
                },
            ),
            # concat transforms
            # ConcatTransform(
            #     # video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
            # GR00TTransform(
            #     state_horizon=len(self.observation_indices),
            #     action_horizon=len(self.action_indices),
            #     max_state_dim=64,
            #     max_action_dim=32,
            # ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class SingleFrankaRobotiqDeltaEefDataConfig:
    video_keys = [
        "video.base_view",
        "video.ego_view",
    ]
    state_keys = [
        "state.eef_position",
        "state.eef_rotation",
    ]
    action_keys = [
        "action.delta_eef_position",
        "action.delta_eef_rotation",
        "action.gripper_close",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.eef_position": "min_max",
                    "state.eef_rotation": "min_max",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.delta_eef_position": "min_max",
                    "action.delta_eef_rotation": "min_max",
                    "action.gripper_close": "binary",
                },
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

###########################################################################################

class Libero4in1DataConfig:
    video_keys = [
        "video.image",
        "video.wrist_image",
    ]
    
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.pad",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    
    language_keys = ["annotation.human.action.task_description"]

    observation_indices = [0]
    action_indices = list(range(16))


    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
            apply_to=self.action_keys,
            normalization_modes={
                "action.x": "min_max",
                "action.y": "min_max",
                "action.z": "min_max",
                "action.roll": "min_max",
                "action.pitch": "min_max",
                "action.yaw": "min_max",
            },
        ),
        ]

        return ComposedModalityTransform(transforms=transforms)

###########################################################################################


class SingleFrankaRobotiqDeltaJointsDataConfig:
    video_keys = [
        "video.base_view",
        "video.ego_view",
    ]
    state_keys = [
        "state.joints",
    ]
    action_keys = [
        "action.delta_joints",
        "action.gripper_close",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.joints": "min_max",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.delta_joints": "min_max",
                    "action.gripper_close": "binary",
                },
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################

class FourierGr1ArmsWaistDataConfig:
    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.waist",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.waist",
    ]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [0]
    action_indices = list(range(16))


    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            # ConcatTransform(
            #     video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
        ]
        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class Robocasa365DataConfig:
    """RoboCasa365 Panda-Omron (LeRobot v3) data schema."""

    video_keys = [
        "video.robot0_eye_in_hand",
        "video.robot0_agentview_left",
        "video.robot0_agentview_right",
    ]

    state_keys = [
        "state.base_position",
        "state.base_rotation",
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.gripper_qpos",
    ]

    action_keys = [
        "action.base_motion",
        "action.control_mode",
        "action.end_effector_position",
        "action.end_effector_rotation",
        "action.gripper_close",
    ]

    language_keys = ["annotation.human.task_description"]

    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


###########################################################################################

class SO101Config:
    #input
    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    
    state_keys = [
        "state.shoulder_pan.pos",
        "state.shoulder_lift.pos",
        "state.elbow_flex.pos",
        "state.wrist_flex.pos",
        "state.wrist_roll.pos",
        "state.gripper.pos",
    ]
    language_keys = ["annotation.human.action.task_description"]

    # output
    action_keys = [
        "action.shoulder_pan.pos",
        "action.shoulder_lift.pos",
        "action.elbow_flex.pos",
        "action.wrist_flex.pos",
        "action.wrist_roll.pos",
        "action.gripper.pos",
    ]
    

    observation_indices = [0]
    action_indices = list(range(16))


    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    key: "min_max" for key in self.state_keys
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    key: "min_max" for key in self.action_keys
                },
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)



class ArxX5DataConfig:
    video_keys = [
        "video.cam_high",
        "video.cam_left_wrist",
        "video.cam_right_wrist",
    ]
    state_keys = [
        "state.left_joints",
        "state.right_joints",
        "state.left_gripper",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_joints",
        "action.right_joints",
        "action.left_gripper",
        "action.right_gripper",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_joints": "min_max",
                    "state.right_joints": "min_max",
                    "state.left_gripper": "binary",
                    "state.right_gripper": "binary",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_joints": "min_max",
                    "action.right_joints": "min_max",
                    "action.left_gripper": "binary",
                    "action.right_gripper": "binary",
                },
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

###########################################################################################


class AgilexDataConfig:
    video_keys = [
        "video.cam_high",
        "video.cam_left_wrist",
        "video.cam_right_wrist",
    ]
    state_keys = [
        "state.left_joints",
        "state.right_joints",
        "state.left_gripper",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_joints",
        "action.right_joints",#@JinhuiYE this order is different from Dataset
        "action.left_gripper",
        "action.right_gripper",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_joints": "min_max",
                    "state.right_joints": "min_max",
                    "state.left_gripper": "binary",
                    "state.right_gripper": "binary",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_joints": "min_max",
                    "action.right_joints": "min_max",
                    "action.left_gripper": "binary",
                    "action.right_gripper": "binary",
                },
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

###########################################################################################

class BEHAVIORDataConfig: # TBD 能否直接覆盖掉 starVLA/dataloader/gr00t_lerobot/embodiment_tags.py
    video_keys = [
        "video.head_rgb",
        # for dino
        "video.left_wrist_rgb",
        "video.right_wrist_rgb",

    ]
    
    # state_keys = [
    #     "state.base_qvel",
    #     "state.trunk_qpos",
    #     "state.arm_left_qpos",
    #     "state.arm_right_qpos",
    # ]
    state_keys = [
        "state.joint_qpos_sin",
        "state.joint_qpos_cos",
    ]
    action_keys = [
        "action.base",
        "action.torso",
        "action.left_arm",
        "action.left_gripper",
        "action.right_arm",
        "action.right_gripper",
    ]
    
    language_keys = ["annotation.human.action.task_description"]

    # 定义了 index 的 horizon
    observation_indices = [0]
    action_indices = list(range(50)) #
    action_dim = 23

    def modality_config(self): # TODO 为什么嗨哟这里的一层？
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # # state transforms @Jinhui 因为原始数据就算了，直接用原始的
            # StateActionToTensor(apply_to=self.state_keys),
            # StateActionSinCosTransform(apply_to=self.state_keys),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            )]
        return ComposedModalityTransform(transforms=transforms)


class LerobotV21RoboTwinDataConfig:
    """RoboTwin dual-arm dataset in Lerobot_v21 format (left/right joints + gripper, 3 cameras)."""

    video_keys = [
        "video.cam_high",
        "video.cam_left_wrist",
        "video.cam_right_wrist",
    ]
    state_keys = [
        "state.left_joints",
        "state.left_gripper",
        "state.right_joints",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_joints",
        "action.left_gripper",
        "action.right_joints",
        "action.right_gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    def transform(self):
        transforms = [
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_joints": "q99",
                    "state.left_gripper": "binary",
                    "state.right_joints": "q99",
                    "state.right_gripper": "binary",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_joints": "q99",
                    "action.left_gripper": "binary",
                    "action.right_joints": "q99",
                    "action.right_gripper": "binary",
                },
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


# the config class for ALOHA datasets, like RDT-yhq
class LerobotV21AlohaDataConfig:
    """ALOHA dual-arm dataset in Lerobot_v21 format (left/right joint + gripper, 3 cameras)."""

    video_keys = [
        "video.image_0",
        "video.image_1",
        "video.image_2",
    ]
    state_keys = [
        "state.left_joint",
        "state.left_gripper",
        "state.right_joint",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_joint",
        "action.left_gripper",
        "action.right_joint",
        "action.right_gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    def transform(self):
        transforms = [
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_joint": "q99",
                    "state.left_gripper": "binary",
                    "state.right_joint": "q99",
                    "state.right_gripper": "binary",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_joint": "q99",
                    "action.left_gripper": "binary",
                    "action.right_joint": "q99",
                    "action.right_gripper": "binary",
                },
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class LerobotDataV21MergedAlohaDataConfig:
    """Merged dual-arm LeRobot dataset with left/right 7-dim joint vectors and task-index annotations."""

    video_keys = [
        "video.front",
        "video.left",
        "video.right",
    ]
    state_keys = [
        "state.left_joint_pos",
        "state.right_joint_pos",
    ]
    action_keys = [
        "action.left_joint_pos",
        "action.right_joint_pos",
    ]
    language_keys = ["annotation.task"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    def transform(self):
        transforms = [
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_joint_pos": "min_max",
                    "state.right_joint_pos": "min_max",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_joint_pos": "min_max",
                    "action.right_joint_pos": "min_max",
                },
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class RoboCOIN_CM14_DataConfig:
    """RoboCOIN Cobot Magic 14-dim (6 joints + 1 gripper per arm, 3 cameras).
    All modality.json files have been pre-populated with left_joint/left_gripper/right_joint/right_gripper
    mapped to the single `action[14]` / `observation.state[14]` vector via indices.
    video: image_0 (cam_front_rgb), image_1 (cam_left_wrist_rgb), image_2 (cam_right_wrist_rgb).
    """

    video_keys = [
        "video.image_0",
        "video.image_1",
        "video.image_2",
    ]
    state_keys = [
        "state.left_joint",
        "state.left_gripper",
        "state.right_joint",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_joint",
        "action.left_gripper",
        "action.right_joint",
        "action.right_gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        transforms = [
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_joint": "q99",
                    "state.left_gripper": "binary",
                    "state.right_joint": "q99",
                    "state.right_gripper": "binary",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_joint": "q99",
                    "action.left_gripper": "binary",
                    "action.right_joint": "q99",
                    "action.right_gripper": "binary",
                },
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class RoboCOIN_CM26_DataConfig:
    """RoboCOIN Cobot Magic / Split Aloha 26-dim, only extracting 14-dim joints+gripper.
    The modality.json left_joint/right_joint already point to indices [0..5] / [7..12] of
    the 26-dim action/state vector (skipping eef dims at [7..12] and [20..25]).
    video: image_0 (cam_high_rgb), image_1 (cam_left_wrist_rgb), image_2 (cam_right_wrist_rgb).
    """

    video_keys = [
        "video.image_0",
        "video.image_1",
        "video.image_2",
    ]
    state_keys = [
        "state.left_joint",
        "state.left_gripper",
        "state.right_joint",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_joint",
        "action.left_gripper",
        "action.right_joint",
        "action.right_gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        transforms = [
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_joint": "q99",
                    "state.left_gripper": "binary",
                    "state.right_joint": "q99",
                    "state.right_gripper": "binary",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_joint": "q99",
                    "action.left_gripper": "binary",
                    "action.right_joint": "q99",
                    "action.right_gripper": "binary",
                },
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class RoboMIND_V1_Agilex_DataConfig:
    """RoboMIND V1.0 Agilex 3RGB, only extracting joint_position 14-dim.
    modality.json maps left_joint => actions.joint_position_left[0:6],
    left_gripper => actions.joint_position_left[6], etc.
    video: image_0 (camera_front), image_1 (camera_left_wrist), image_2 (camera_right_wrist).
    """

    video_keys = [
        "video.image_0",
        "video.image_1",
        "video.image_2",
    ]
    state_keys = [
        "state.left_joint",
        "state.left_gripper",
        "state.right_joint",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_joint",
        "action.left_gripper",
        "action.right_joint",
        "action.right_gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        transforms = [
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_joint": "q99",
                    "state.left_gripper": "binary",
                    "state.right_joint": "q99",
                    "state.right_gripper": "binary",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_joint": "q99",
                    "action.left_gripper": "binary",
                    "action.right_joint": "q99",
                    "action.right_gripper": "binary",
                },
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class RoboMIND_V2_Agilex_DataConfig:
    """RoboMIND V2.0 Agilex, only extracting arm_position + gripper = 14-dim.
    modality.json maps left_joint => actions.arm_left_position_align[0:6],
    left_gripper => actions.end_effector_left_position_align[0], etc.
    video: image_0 (camera_front), image_1 (camera_left), image_2 (camera_right).
    """

    video_keys = [
        "video.image_0",
        "video.image_1",
        "video.image_2",
    ]
    state_keys = [
        "state.left_joint",
        "state.left_gripper",
        "state.right_joint",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_joint",
        "action.left_gripper",
        "action.right_joint",
        "action.right_gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        transforms = [
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_joint": "q99",
                    "state.left_gripper": "binary",
                    "state.right_joint": "q99",
                    "state.right_gripper": "binary",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_joint": "q99",
                    "action.left_gripper": "binary",
                    "action.right_joint": "q99",
                    "action.right_gripper": "binary",
                },
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class RoboMIND_V2_Mobile_DataConfig:
    """RoboMIND V2.0 Agilex Mobile, only extracting arm_position + gripper = 14-dim.
    Same mapping as V2 Agilex (chassis dims ignored by DataConfig — not in action_keys).
    video: image_0 (camera_front), image_1 (camera_left), image_2 (camera_right).
    """

    video_keys = [
        "video.image_0",
        "video.image_1",
        "video.image_2",
    ]
    state_keys = [
        "state.left_joint",
        "state.left_gripper",
        "state.right_joint",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_joint",
        "action.left_gripper",
        "action.right_joint",
        "action.right_gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        transforms = [
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_joint": "q99",
                    "state.left_gripper": "binary",
                    "state.right_joint": "q99",
                    "state.right_gripper": "binary",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_joint": "q99",
                    "action.left_gripper": "binary",
                    "action.right_joint": "q99",
                    "action.right_gripper": "binary",
                },
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


ROBOT_TYPE_CONFIG_MAP = {
    "libero_franka": Libero4in1DataConfig(),
    "oxe_droid": OxeDroidDataConfig(),
    "oxe_bridge": OxeBridgeDataConfig(),
    "oxe_rt1": OxeRT1DataConfig(),
    "SO101": SO101Config(),
    "demo_sim_franka_delta_joints": SingleFrankaRobotiqDeltaJointsDataConfig(),
    "arx_x5": ArxX5DataConfig(),
    "robotwin": AgilexDataConfig(),
    "lerobot_v21_robotwin": LerobotV21RoboTwinDataConfig(),
    "fourier_gr1_arms_waist": FourierGr1ArmsWaistDataConfig(),
    "robocasa365_panda_omron": Robocasa365DataConfig(),
    "R1Pro": BEHAVIORDataConfig(),
    "custom_robot_config": SingleFrankaRobotiqDeltaEefDataConfig(),
    "lerobot_v21_aloha": LerobotV21AlohaDataConfig(),
    "lerobot_data_v21_merged_aloha": LerobotDataV21MergedAlohaDataConfig(),
    "robocoin_cm14": RoboCOIN_CM14_DataConfig(),
    "robocoin_cm26": RoboCOIN_CM26_DataConfig(),
    "robomind_v1_agilex": RoboMIND_V1_Agilex_DataConfig(),
    "robomind_v2_agilex": RoboMIND_V2_Agilex_DataConfig(),
    "robomind_v2_mobile": RoboMIND_V2_Mobile_DataConfig(),
}
