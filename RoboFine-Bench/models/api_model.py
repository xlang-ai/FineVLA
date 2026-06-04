"""Built-in API model implementation using OpenAI-compatible protocol."""

import base64
import io
import logging
import os
import time
from typing import Dict, List, Tuple

import httpx
from openai import OpenAI
from PIL import Image

from .base_model import BaseVLM

logger = logging.getLogger("robofine-bench")

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MAX_RETRIES = 5


def _images_to_base64_parts(images: List[Image.Image], quality: int = 85) -> List[Dict]:
    parts = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    return parts


class APIModel(BaseVLM):
    """OpenAI-compatible API model (supports Qwen, GPT, vLLM, Ollama, etc.).

    Usage:
        model = APIModel("qwen3-vl-plus")
        response, tokens = model.generate(images, prompt, system_prompt)
    """

    def __init__(
        self,
        model_name: str,
        api_key: str = None,
        base_url: str = None,
        enable_thinking: bool = True,
    ):
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")
        base_url = base_url or DEFAULT_BASE_URL
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.Client(timeout=httpx.Timeout(600.0, connect=60.0)),
        )

    def _is_qwen(self) -> bool:
        m = self.model_name.lower()
        return "qwen" in m or "qvq" in m

    def _is_gpt_thinking(self) -> bool:
        m = self.model_name.lower()
        if "." in m:
            m = m.split(".", 1)[-1]
        return m.startswith("o3") or m.startswith("o4") or m.startswith("o1") or m.startswith("gpt-5")

    def generate(
        self,
        images: List[Image.Image],
        prompt: str,
        system_prompt: str = "",
    ) -> Tuple[str, Dict]:
        image_parts = _images_to_base64_parts(images)

        if self._is_qwen():
            combined_text = f"{system_prompt.strip()}\n\n{prompt}" if system_prompt else prompt
            url_list = [p["image_url"]["url"] for p in image_parts]
            if len(url_list) >= 4:
                messages = [{"role": "user", "content": [
                    {"type": "video", "video": url_list},
                    {"type": "text", "text": combined_text},
                ]}]
            else:
                messages = [{"role": "user", "content": [
                    *image_parts,
                    {"type": "text", "text": combined_text},
                ]}]
        elif self._is_gpt_thinking():
            messages = [
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": [*image_parts, {"type": "text", "text": prompt}]},
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [*image_parts, {"type": "text", "text": prompt}]},
            ]

        extra = None
        if self._is_qwen() and self.enable_thinking:
            extra = {"qwen": {"thinking_config": {"enable_thinking": True}}}
        elif self._is_gpt_thinking():
            extra = {"reasoning_effort": "high", "max_completion_tokens": 32768}

        for attempt in range(MAX_RETRIES):
            try:
                kwargs = {"model": self.model_name, "messages": messages}
                if extra:
                    kwargs["extra_body"] = extra
                response = self.client.chat.completions.create(**kwargs)
                if not hasattr(response, "choices") or not response.choices:
                    raise Exception("No choices returned")
                msg = response.choices[0].message
                content = ""
                if isinstance(msg.content, str):
                    content = msg.content.strip()
                elif isinstance(msg.content, list):
                    content = "".join(
                        p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")
                        for p in msg.content
                    ).strip()
                usage = getattr(response, "usage", None)
                token_info = {}
                if usage:
                    token_info = {
                        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                    }
                return content or "", token_info
            except Exception as e:
                wait = min(2 ** attempt, 30) if ("429" in str(e) or "rate" in str(e).lower()) else 2
                logger.warning(f"API attempt {attempt+1}/{MAX_RETRIES}: {str(e)[:200]}")
                time.sleep(wait)

        return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
