"""Example: Evaluate a local HuggingFace model on RoboFine-Bench.

Usage:
    # 1. Pre-extract frames
    python prepare_frames.py --evalsets EvalData/EvalSets.json --video-dir EvalData/Videos --fps 2.0

    # 2. Run your model (modify this script with your model)
    python models/example_local_model.py --frames-dir EvalData/frames --output my_results.jsonl
"""

from models.base_model import BaseVLM
from PIL import Image
from typing import Dict, List, Tuple


class LocalVLMExample(BaseVLM):
    """Example: wrap a local HuggingFace model as a BaseVLM.

    Replace the __init__ and generate methods with your own model's logic.
    """

    def __init__(self, model_path: str):
        # Example: load your model
        # from transformers import AutoModelForVision2Seq, AutoProcessor
        # self.model = AutoModelForVision2Seq.from_pretrained(model_path)
        # self.processor = AutoProcessor.from_pretrained(model_path)
        self.model_path = model_path
        print(f"[Example] Would load model from: {model_path}")

    def generate(
        self,
        images: List[Image.Image],
        prompt: str,
        system_prompt: str = "",
    ) -> Tuple[str, Dict]:
        """Replace this with your model's inference logic.

        Args:
            images: List of PIL Images (video frames at 2 FPS)
            prompt: The evaluation prompt
            system_prompt: System-level instructions

        Returns:
            (response_text, token_usage_dict)
        """
        # Example inference (replace with your actual model call):
        #
        # inputs = self.processor(
        #     text=prompt, images=images, return_tensors="pt"
        # ).to(self.model.device)
        # outputs = self.model.generate(**inputs, max_new_tokens=2048)
        # response = self.processor.decode(outputs[0], skip_special_tokens=True)
        # return response, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        return f"[Example] Got {len(images)} frames, prompt length {len(prompt)}", {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        }


if __name__ == "__main__":
    model = LocalVLMExample("path/to/your/model")
    dummy_images = [Image.new("RGB", (224, 224), color="red") for _ in range(5)]
    response, tokens = model.generate(
        dummy_images,
        "Describe the robot manipulation shown in these frames.",
        "You are a robot manipulation expert.",
    )
    print(f"Response: {response}")
    print(f"Tokens: {tokens}")
