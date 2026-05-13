from pathlib import Path
from typing import Dict, Optional

import cv2 as cv
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from starVLA.model.framework.share_tools import read_mode_config


class PolicyWarper365:
    """Adapter: websocket policy output -> RoboCasa365 action dict."""

    def __init__(
        self,
        policy_ckpt_path: str,
        unnorm_key: Optional[str] = "new_embodiment",
        image_size: list[int] = [224, 224],
        host: str = "127.0.0.1",
        port: int = 5678,
        n_action_steps: int = 8,
    ) -> None:
        self.client = WebsocketClientPolicy(host, port)
        self.image_size = image_size
        self.n_action_steps = n_action_steps
        self.task_description = None
        self.action_norm_stats = self.get_action_stats(unnorm_key, policy_ckpt_path)

    def reset(self, task_description: Optional[str]) -> None:
        self.task_description = task_description

    def step(self, observations, **kwargs):
        task_description = self._extract_task_description(observations)
        if task_description != self.task_description:
            self.reset(task_description)

        images = self._extract_images(observations)
        examples = [
            {
                "image": images,
                "lang": self.task_description or "",
                "robot_tag": "new_embodiment",
            }
        ]

        vla_input = {
            "examples": examples,
            "do_sample": False,
        }
        response = self.client.predict_action(vla_input)

        normalized_actions = response["data"]["normalized_actions"]
        normalized_actions = normalized_actions[:, :, :12]
        raw_actions = self.unnormalize_actions(normalized_actions, self.action_norm_stats)

        raw_action = {
            "action.base_motion": raw_actions[:, : self.n_action_steps, :4],
            "action.control_mode": raw_actions[:, : self.n_action_steps, 4:5],
            "action.end_effector_position": raw_actions[:, : self.n_action_steps, 5:8],
            "action.end_effector_rotation": raw_actions[:, : self.n_action_steps, 8:11],
            "action.gripper_close": raw_actions[:, : self.n_action_steps, 11:12],
        }
        return {"actions": raw_action}

    def _extract_task_description(self, observations) -> str:
        for key in [
            "annotation.human.task_description",
            "annotation.human.action.task_description",
            "annotation.human.coarse_action",
        ]:
            if key in observations:
                value = observations[key]
                if isinstance(value, (list, tuple, np.ndarray)):
                    return str(value[0])
                return str(value)
        return ""

    def _extract_images(self, observations):
        preferred = [
            "video.robot0_eye_in_hand",
            "video.robot0_agentview_left",
            "video.robot0_agentview_right",
        ]
        available = [k for k in preferred if k in observations]
        if not available:
            available = sorted([k for k in observations.keys() if k.startswith("video")])

        images = []
        for key in available:
            frame = observations[key]
            if isinstance(frame, np.ndarray) and frame.ndim >= 4:
                frame = frame[0]
            if isinstance(frame, np.ndarray) and frame.ndim >= 4:
                frame = frame[-1]
            if isinstance(frame, np.ndarray):
                frame = self._resize_image(frame)
            images.append(frame)
        return images

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)

    @staticmethod
    def get_action_stats(unnorm_key: Optional[str], policy_ckpt_path: str) -> dict:
        policy_ckpt_path = Path(policy_ckpt_path)
        _, norm_stats = read_mode_config(policy_ckpt_path)

        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                "Model has multiple normalization keys; please set unnorm_key explicitly. "
                f"Available keys: {list(norm_stats.keys())}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        if unnorm_key not in norm_stats:
            raise KeyError(
                f"unnorm_key={unnorm_key} not found. Available keys: {list(norm_stats.keys())}"
            )
        return norm_stats[unnorm_key]["action"]

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
        action_high = np.array(action_norm_stats["max"])
        action_low = np.array(action_norm_stats["min"])

        normalized_actions = np.clip(normalized_actions, -1, 1)
        actions = np.where(
            mask,
            (normalized_actions + 1) / 2 * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions
