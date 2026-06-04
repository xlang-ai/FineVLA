"""Base VLM interface for RoboFine-Bench evaluation."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from PIL import Image


class BaseVLM(ABC):
    """Abstract base class for Vision-Language Models.

    To evaluate your own model on RoboFine-Bench, subclass this and implement generate().

    Example:
        class MyModel(BaseVLM):
            def __init__(self, model_path):
                self.model = load_model(model_path)

            def generate(self, images, prompt, system_prompt=""):
                # Your inference logic here
                return response_text, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    """

    @abstractmethod
    def generate(
        self,
        images: List[Image.Image],
        prompt: str,
        system_prompt: str = "",
    ) -> Tuple[str, Dict]:
        """Generate a response given video frames and a prompt.

        Args:
            images: List of PIL Images, sampled from the video at a fixed FPS (default 2.0),
                    ordered chronologically. For multi-view inputs, frames from all views
                    are concatenated in order.
            prompt: The user prompt (question for VQA, annotation instruction for Caption).
            system_prompt: Optional system prompt.

        Returns:
            A tuple of (response_text, token_usage_dict).
            token_usage_dict should contain "prompt_tokens", "completion_tokens", "total_tokens"
            (set all to 0 if not available).
        """
        raise NotImplementedError
