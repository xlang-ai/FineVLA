import json
import os
import random
import re
import time
from typing import Dict, List, Optional, Tuple

import httpx

import config as config_module
from config import (
    DEFAULT_MODEL,
    DEFAULT_BASE_URL,
    MAX_RETRIES,
    RATE_LIMIT_DELAY,
    RATE_LIMIT_BACKOFF_CAP,
    logger,
)
from video_utils import _ensure_imports


# =============================================================================
# API Interaction
# =============================================================================

def create_api_client(api_key: str = None, base_url: str = None):
    """Create OpenAI-compatible API client."""
    _ensure_imports()

    api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")

    base_url = base_url or DEFAULT_BASE_URL

    http_client = httpx.Client(
        timeout=httpx.Timeout(600.0, connect=60.0),
    )

    return config_module.OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client
    )


def _build_extra_body(model: str) -> Optional[dict]:
    """Return model-specific extra_body or None."""
    if "qwen" in model.lower():
        return {
            "qwen": {
                "thinking_config": {
                    "enable_thinking": True,
                }
            }
        }
    return None


def _is_qwen_model(model: str) -> bool:
    m = model.lower()
    return "qwen" in m or "qvq" in m


def _build_user_content(
    image_parts: List[Dict], user_prompt: str, model: str,
    system_prompt: str = "",
    fps: float = None,
) -> List[Dict]:
    """Build user message content.

    For Qwen models:
      - frames are wrapped in {"type": "video", "video": [...], "fps": X}
      - system_prompt is prepended to the user text (Qwen best-practice)
    For other models:
      - standard image_url format, system_prompt stays in system message
    """
    combined_text = f"{system_prompt.strip()}\n\n{user_prompt}" if system_prompt else user_prompt

    if _is_qwen_model(model) and image_parts:
        video_urls = []
        for part in image_parts:
            if part.get("type") == "image_url":
                video_urls.append(part["image_url"]["url"])
        if video_urls:
            video_part = {"type": "video", "video": video_urls}
            if fps is not None:
                video_part["fps"] = max(0.1, min(fps, 10.0))
            return [
                video_part,
                {"type": "text", "text": combined_text},
            ]
    if system_prompt:
        return [*image_parts, {"type": "text", "text": combined_text}]
    return [*image_parts, {"type": "text", "text": user_prompt}]


def call_vision_api(
    client,
    image_parts: List[Dict],
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    max_retries: int = MAX_RETRIES,
    target_fps: float = None,
) -> Tuple[str, Dict[str, int]]:
    """
    Call vision API with images and prompts.

    Returns:
        (content, token_usage) where token_usage has keys
        prompt_tokens / completion_tokens / total_tokens.
        On final failure content is "" and token_usage is empty.
    """
    if _is_qwen_model(model):
        messages = [
            {
                "role": "user",
                "content": _build_user_content(
                    image_parts, user_prompt, model,
                    system_prompt=system_prompt, fps=target_fps,
                ),
            }
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": _build_user_content(image_parts, user_prompt, model, fps=target_fps),
            }
        ]

    extra = _build_extra_body(model)
    last_error = None
    attempt = 0
    non_rl_failures = 0

    while True:
        try:
            kwargs = {"model": model, "messages": messages}
            if extra:
                kwargs["extra_body"] = extra
            response = client.chat.completions.create(**kwargs)

            if not hasattr(response, 'choices') or not response.choices:
                raise Exception("Invalid response: no choices returned")

            for i, choice in enumerate(response.choices):
                fr = getattr(choice, 'finish_reason', None)
                if fr not in ("stop", "function_call"):
                    raise Exception(f"Invalid finish_reason '{fr}' for choice {i}")

            msg = response.choices[0].message
            content = ""
            if isinstance(msg.content, str):
                content = msg.content.strip()
            elif isinstance(msg.content, list):
                text_parts = []
                for part in msg.content:
                    p = part if isinstance(part, dict) else part.model_dump()
                    if p.get("type") == "text":
                        text_parts.append(p.get("text", ""))
                content = "".join(text_parts).strip()

            usage = getattr(response, 'usage', None)
            token_info: Dict[str, int] = {}
            if usage:
                token_info = {
                    "prompt_tokens": getattr(usage, 'prompt_tokens', 0) or 0,
                    "completion_tokens": getattr(usage, 'completion_tokens', 0) or 0,
                    "total_tokens": getattr(usage, 'total_tokens', 0) or 0,
                }

            time.sleep(RATE_LIMIT_DELAY)
            return content, token_info

        except Exception as e:
            last_error = e
            attempt += 1
            err = str(e)
            is_rl = "429" in err or "RESOURCE_EXHAUSTED" in err or "rate" in err.lower()

            if is_rl:
                base = min(2 ** min(attempt, 8) * 3, RATE_LIMIT_BACKOFF_CAP)
                backoff = base + random.uniform(0, base * 0.3)
                logger.warning(f"Rate limited (attempt {attempt}, backoff {backoff:.0f}s)")
            else:
                non_rl_failures += 1
                backoff = RATE_LIMIT_DELAY * non_rl_failures
                logger.warning(f"API error [{non_rl_failures}/{max_retries}]: {e}")
                if non_rl_failures >= max_retries:
                    break

            time.sleep(backoff)

    logger.error(f"API failed after {attempt} attempts: {last_error}")
    return "", {}


def _find_first_json_object(text: str) -> Optional[str]:
    """Extract the first balanced JSON object from text (respects strings)."""
    depth = 0
    start = None
    in_string = False
    escape_next = False

    for i, c in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if c == '\\' and in_string:
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            if depth == 0:
                start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i + 1]
    return None


def extract_json_from_response(text: str) -> Optional[Dict]:
    """Extract JSON from model response, handling code fences and <think> blocks."""
    text = text.strip()
    if not text:
        return None

    # Strip <think>...</think> blocks (model reasoning/chain-of-thought)
    if "<think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    if not text:
        return None

    # Try extracting from code fences
    code_blocks = re.findall(r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if code_blocks:
        for block in code_blocks:
            block = block.strip()
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                obj_str = _find_first_json_object(block)
                if obj_str:
                    try:
                        return json.loads(obj_str)
                    except json.JSONDecodeError:
                        continue

    # Try parsing the whole text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try finding first JSON object in text
    obj_str = _find_first_json_object(text)
    if obj_str:
        try:
            return json.loads(obj_str)
        except json.JSONDecodeError:
            pass

    return None
