"""
Unified API calling module for robot manipulation video annotation.

Supports: Qwen, Gemini (native), Doubao (Responses API), GPT (Responses API),
and generic OpenAI-compatible models. All via DashScope gateway.
"""

import json
import logging
import os
import random
import re
import time
from typing import Dict, List, Optional, Tuple

import httpx
from openai import OpenAI

logger = logging.getLogger("annotation")

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MAX_RETRIES = 5
DOUBAO_MAX_FRAMES = 150
GPT_MAX_FRAMES = 250


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def create_api_client(api_key: str = None, base_url: str = None) -> OpenAI:
    api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    base_url = base_url or DEFAULT_BASE_URL
    http_client = httpx.Client(timeout=httpx.Timeout(600.0, connect=60.0))
    return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)


# ---------------------------------------------------------------------------
# Model detection helpers
# ---------------------------------------------------------------------------

def _is_qwen_model(model: str) -> bool:
    m = model.lower()
    return "qwen" in m or "qvq" in m or "robofine" in m


def _is_gemini_native_model(model: str) -> bool:
    m = model.lower()
    if m.startswith("vertex_ai.") or m.startswith("ai_studio."):
        return True
    if "gemini-robotics" in m:
        return True
    bare = m.split(".", 1)[-1] if "." in m else m
    if bare.startswith("gemini-3") or bare.startswith("gemini-4"):
        return True
    return False


def _is_gpt_thinking_model(model: str) -> bool:
    m = model.lower()
    if "." in m:
        m = m.split(".", 1)[-1]
    return m.startswith("o3") or m.startswith("o4") or m.startswith("o1") or m.startswith("gpt-5")


def _is_gpt_responses_api_model(model: str) -> bool:
    """gpt-5.4-pro uses Responses API; gpt-5.4 uses Chat Completions."""
    m = model.lower()
    if "." in m:
        m = m.split(".", 1)[-1]
    return "gpt-5" in m and "pro" in m


def _is_doubao_model(model: str) -> bool:
    return "doubao" in model.lower()


# ---------------------------------------------------------------------------
# Frame subsampling
# ---------------------------------------------------------------------------

def _subsample_parts(image_parts: List[Dict], max_frames: int) -> List[Dict]:
    if len(image_parts) <= max_frames:
        return image_parts
    indices = [int(i * (len(image_parts) - 1) / (max_frames - 1)) for i in range(max_frames)]
    return [image_parts[i] for i in indices]


def _subsample_multiview_parts(image_parts: List[Dict], max_total: int) -> List[Dict]:
    """Subsample multi-view image_parts that have [View: ...] text labels interleaved."""
    frame_count = sum(1 for p in image_parts if p.get("type") == "image_url")
    if frame_count <= max_total:
        return image_parts

    views = []
    cur_label, cur_frames = None, []
    for p in image_parts:
        if p.get("type") == "text" and str(p.get("text", "")).startswith("[View:"):
            if cur_label is not None or cur_frames:
                views.append((cur_label, cur_frames))
            cur_label, cur_frames = p, []
        elif p.get("type") == "image_url":
            cur_frames.append(p)
    if cur_label is not None or cur_frames:
        views.append((cur_label, cur_frames))

    total_orig = sum(len(f) for _, f in views)
    result = []
    for label, frames in views:
        quota = max(1, int(len(frames) / total_orig * max_total))
        sampled = _subsample_parts(frames, quota)
        if label is not None:
            result.append(label)
        result.extend(sampled)
    return result


# ---------------------------------------------------------------------------
# Gemini native protocol helpers
# ---------------------------------------------------------------------------

def _image_parts_to_gemini(image_parts: List[Dict]) -> List[Dict]:
    parts = []
    for p in image_parts:
        if p.get("type") == "image_url":
            url = p["image_url"]["url"]
            if url.startswith("data:"):
                header, b64 = url.split(",", 1)
                mime = header.split(":")[1].split(";")[0]
                parts.append({"inlineData": {"data": b64, "mimeType": mime}})
            else:
                parts.append({"fileData": {"fileUri": url, "mimeType": "image/jpeg"}})
        elif p.get("type") == "text":
            parts.append({"text": p["text"]})
    return parts


def _build_gemini_body(model: str, image_parts: List[Dict],
                       system_prompt: str, user_prompt: str,
                       video_urls: List[str] = None, fps: float = None) -> Dict:
    if video_urls:
        gemini_parts = []
        for vu in video_urls:
            part = {"fileData": {"fileUri": vu, "mimeType": "video/mp4"}}
            if fps is not None:
                part["videoMetadata"] = {"fps": fps}
            gemini_parts.append(part)
        gemini_parts.append({"text": user_prompt})
    else:
        gemini_parts = _image_parts_to_gemini(image_parts)
        gemini_parts.append({"text": user_prompt})

    body: Dict = {
        "model": model,
        "contents": [{"role": "user", "parts": gemini_parts}],
    }
    if not model.lower().startswith("vertex_ai."):
        body["dashscope_extend_params"] = {"using_native_protocol": True}
    if system_prompt:
        body["system_instruction"] = {"parts": [{"text": system_prompt}]}

    suffix = model.lower().split(".", 1)[-1] if "." in model.lower() else model.lower()
    supports_thinking = (
        ("gemini-3" in suffix or "gemini-4" in suffix)
        and "image-preview" not in suffix
    )
    if supports_thinking:
        body["generationConfig"] = {
            "thinkingConfig": {"thinkingLevel": "high", "includeThoughts": False}
        }
    elif "gemini-2.5" in suffix:
        body["generationConfig"] = {
            "thinkingConfig": {"thinkingBudget": -1, "includeThoughts": False}
        }

    return body


# ---------------------------------------------------------------------------
# Core API dispatchers
# ---------------------------------------------------------------------------

def _call_gemini_native(client, image_parts, system_prompt, user_prompt, model,
                        video_urls=None, fps=None):
    body = _build_gemini_body(model, image_parts, system_prompt, user_prompt,
                              video_urls=video_urls, fps=fps)
    api_key = client.api_key
    base_url = str(client.base_url).rstrip("/")
    url = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    http = httpx.Client(timeout=httpx.Timeout(600.0, connect=60.0))

    for attempt in range(MAX_RETRIES):
        try:
            resp = http.post(url, headers=headers, json=body)
            if resp.status_code != 200:
                raise Exception(f"HTTP {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            content, token_info = "", {}
            if "candidates" in data and data["candidates"]:
                parts = data["candidates"][0].get("content", {}).get("parts", [])
                content = "".join(p.get("text", "") for p in parts if "text" in p)
                usage = data.get("usageMetadata", {})
                token_info = {
                    "prompt_tokens": usage.get("promptTokenCount", 0),
                    "completion_tokens": usage.get("candidatesTokenCount", 0),
                    "total_tokens": usage.get("totalTokenCount", 0),
                }
            elif "choices" in data and data["choices"]:
                msg = data["choices"][0].get("message", {})
                c = msg.get("content", "")
                content = "".join(p.get("text", "") for p in c if p.get("type") == "text") if isinstance(c, list) else c
                usage = data.get("usage", {})
                token_info = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
            return content.strip(), token_info
        except Exception as e:
            wait = min(2 ** attempt, 30) if ("429" in str(e) or "rate" in str(e).lower()) else 2
            logger.warning(f"Gemini attempt {attempt+1}: {str(e)[:200]}")
            time.sleep(wait)
    return "", {}


def _doubao_limit_frames(image_parts: List[Dict], max_frames: int) -> List[Dict]:
    """Two-level fallback for Doubao frame limits.

    Level 1: If total frames > max_frames, drop to primary view only (first view group).
    Level 2: If primary view still > max_frames, uniformly subsample.
    """
    has_views = any(
        p.get("type") == "text" and str(p.get("text", "")).startswith("[View:")
        for p in image_parts
    )
    frame_count = sum(1 for p in image_parts if p.get("type") == "image_url")

    if frame_count <= max_frames:
        return image_parts

    if has_views:
        # Level 1: extract primary view only (first view group)
        primary_frames = []
        in_first_view = False
        for p in image_parts:
            if p.get("type") == "text" and str(p.get("text", "")).startswith("[View:"):
                if not in_first_view and not primary_frames:
                    in_first_view = True
                    continue
                else:
                    break
            elif p.get("type") == "image_url" and in_first_view:
                primary_frames.append(p)
        if not primary_frames:
            primary_frames = [p for p in image_parts if p.get("type") == "image_url"]
        logger.warning(
            f"Doubao: {frame_count} frames > {max_frames}, "
            f"fallback to primary view ({len(primary_frames)} frames)"
        )
        # Level 2: subsample if still over limit
        return _subsample_parts(primary_frames, max_frames)

    return _subsample_parts(image_parts, max_frames)


def _call_doubao_responses(client, image_parts, system_prompt, user_prompt, model,
                           video_urls=None):
    api_key = client.api_key
    base_url = str(client.base_url).rstrip("/")
    url = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    http = httpx.Client(timeout=httpx.Timeout(600.0, connect=60.0))

    use_video_mode = bool(video_urls)

    if use_video_mode:
        video_model = model if model.endswith("-completion") else model + "-completion"
        combined_text = f"{system_prompt.strip()}\n\n{user_prompt}" if system_prompt else user_prompt
        user_content = []
        for vu in video_urls:
            user_content.append({"type": "video_url", "video_url": {"url": vu}})
        user_content.append({"type": "text", "text": combined_text})
        body = {
            "model": video_model,
            "messages": [{"role": "user", "content": user_content}],
            "reasoning": {"effort": "high"},
        }
        logger.info(f"Doubao: video_url mode, model={video_model}, {len(video_urls)} video(s)")
    else:
        parts = _doubao_limit_frames(image_parts, DOUBAO_MAX_FRAMES)
        user_content = []
        if system_prompt:
            user_content.append({"type": "input_text", "text": system_prompt.strip()})
        for p in parts:
            if p.get("type") == "image_url":
                user_content.append({"type": "input_image", "image_url": p["image_url"]["url"]})
        user_content.append({"type": "input_text", "text": user_prompt})
        body = {
            "model": model,
            "input": [{"type": "message", "role": "user", "content": user_content}],
            "reasoning": {"effort": "high"},
        }

    for attempt in range(MAX_RETRIES):
        try:
            resp = http.post(url, headers=headers, json=body)
            if resp.status_code != 200:
                err = resp.text[:500]
                if not use_video_mode and ("exceed max" in err.lower() or "max message tokens" in err.lower()):
                    current = sum(1 for c in user_content if c.get("type") == "input_image")
                    new_limit = current // 2
                    if new_limit >= 10:
                        logger.warning(f"Doubao: token limit, reducing {current} -> {new_limit}")
                        reduced = _subsample_parts(parts, new_limit)
                        user_content = []
                        if system_prompt:
                            user_content.append({"type": "input_text", "text": system_prompt.strip()})
                        for rp in reduced:
                            if rp.get("type") == "image_url":
                                user_content.append({"type": "input_image", "image_url": rp["image_url"]["url"]})
                        user_content.append({"type": "input_text", "text": user_prompt})
                        body["input"][0]["content"] = user_content
                        parts = reduced
                        continue
                raise Exception(f"HTTP {resp.status_code}: {err}")
            data = resp.json()
            if data.get("error"):
                raise Exception(f"API error: {data['error'].get('message', '')}")

            content = ""
            if use_video_mode:
                # Chat Completions response format
                choices = data.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    c = msg.get("content", "")
                    if isinstance(c, str):
                        content = c
                    elif isinstance(c, list):
                        content = "".join(p.get("text", "") for p in c if isinstance(p, dict))
                usage = data.get("usage", {})
                token_info = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
            else:
                # Responses API format
                for item in data.get("output", []):
                    if item.get("type") == "message":
                        for c in item.get("content", []):
                            if c.get("type") == "output_text":
                                content += c.get("text", "")
                usage = data.get("usage", {})
                token_info = {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                }
            return content.strip(), token_info
        except Exception as e:
            wait = min(2 ** attempt, 30) if ("429" in str(e) or "rate" in str(e).lower()) else 2
            logger.warning(f"Doubao attempt {attempt+1}: {str(e)[:200]}")
            time.sleep(wait)
    return "", {}


def _call_gpt_responses(client, image_parts, system_prompt, user_prompt, model):
    has_views = any(p.get("type") == "text" and str(p.get("text", "")).startswith("[View:") for p in image_parts)
    if has_views:
        image_parts = _subsample_multiview_parts(image_parts, GPT_MAX_FRAMES)
    else:
        image_parts = _subsample_parts(image_parts, GPT_MAX_FRAMES)

    api_key = client.api_key
    base_url = str(client.base_url).rstrip("/")
    url = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    http = httpx.Client(timeout=httpx.Timeout(600.0, connect=60.0))

    user_content = []
    for p in image_parts:
        if p.get("type") == "image_url":
            user_content.append({"type": "input_image", "image_url": p["image_url"]["url"]})
    user_content.append({"type": "input_text", "text": user_prompt})

    body = {
        "model": model,
        "input": [
            {"role": "developer", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ],
        "reasoning": {"effort": "high"},
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = http.post(url, headers=headers, json=body)
            if resp.status_code != 200:
                raise Exception(f"HTTP {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            if data.get("error") and data.get("status") == "failed":
                raise Exception(f"API error: {data['error'].get('message', '')}")
            content = ""
            for item in data.get("output", []):
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if c.get("type") == "output_text":
                            content += c.get("text", "")
            usage = data.get("usage", {})
            return content.strip(), {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            }
        except Exception as e:
            wait = min(2 ** attempt, 30) if ("429" in str(e) or "rate" in str(e).lower()) else 2
            logger.warning(f"GPT Responses attempt {attempt+1}: {str(e)[:200]}")
            time.sleep(wait)
    return "", {}


def _call_openai_compatible(client, image_parts, system_prompt, user_prompt, model,
                            video_urls=None, fps=None, _ctx_retry=False):
    """OpenAI-compatible path for Qwen, non-native Gemini, and other models."""

    has_views = any(
        p.get("type") == "text" and str(p.get("text", "")).startswith("[View:")
        for p in image_parts
    )
    is_oss = (not has_views and image_parts and
              image_parts[0].get("image_url", {}).get("url", "").startswith("http"))

    if _is_gpt_thinking_model(model) or "gpt" in model.lower():
        if has_views:
            image_parts = _subsample_multiview_parts(image_parts, GPT_MAX_FRAMES)
        else:
            image_parts = _subsample_parts(image_parts, GPT_MAX_FRAMES)

    if video_urls and _is_qwen_model(model):
        combined_text = f"{system_prompt.strip()}\n\n{user_prompt}" if system_prompt else user_prompt
        content = []
        for vu in video_urls:
            vpart = {"type": "video", "video": vu}
            if fps is not None:
                vpart["fps"] = fps
            content.append(vpart)
        content.append({"type": "text", "text": combined_text})
        messages = [{"role": "user", "content": content}]
        logger.info(f"Qwen: video URL mode, {len(video_urls)} video(s), fps={fps}")
    elif _is_qwen_model(model) and has_views:
        combined_text = f"{system_prompt.strip()}\n\n{user_prompt}" if system_prompt else user_prompt
        content = []
        cur_frames = []
        for p in image_parts:
            if p.get("type") == "text":
                if cur_frames:
                    content.append({"type": "video", "video": [f["image_url"]["url"] for f in cur_frames]})
                    cur_frames = []
                content.append(p)
            elif p.get("type") == "image_url":
                cur_frames.append(p)
        if cur_frames:
            content.append({"type": "video", "video": [f["image_url"]["url"] for f in cur_frames]})
        content.append({"type": "text", "text": combined_text})
        messages = [{"role": "user", "content": content}]
    elif _is_qwen_model(model) and is_oss:
        url_list = [p["image_url"]["url"] for p in image_parts if p.get("type") == "image_url"]
        combined_text = f"{system_prompt.strip()}\n\n{user_prompt}" if system_prompt else user_prompt
        messages = [{"role": "user", "content": [
            {"type": "video", "video": url_list},
            {"type": "text", "text": combined_text},
        ]}]
    elif _is_qwen_model(model):
        combined_text = f"{system_prompt.strip()}\n\n{user_prompt}" if system_prompt else user_prompt
        url_list = [p["image_url"]["url"] for p in image_parts if p.get("type") == "image_url"]
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
    elif _is_gpt_thinking_model(model):
        messages = [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": [*image_parts, {"type": "text", "text": user_prompt}]},
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [*image_parts, {"type": "text", "text": user_prompt}]},
        ]

    extra = None
    m = model.lower()
    _base = str(client.base_url)
    if "robofine" in m and ("localhost" in _base or "127.0.0.1" in _base):
        extra = {"chat_template_kwargs": {"enable_thinking": False}}
        mm_kwargs = {
            "min_pixels": 128 * 32 * 32,
            "max_pixels": 1024 * 32 * 32,
            "total_pixels": 224000 * 32 * 32,
        }
        if video_urls:
            mm_kwargs["fps"] = fps or 4
            mm_kwargs["do_sample_frames"] = True
            mm_kwargs["max_frames"] = 512
        extra["mm_processor_kwargs"] = mm_kwargs
    elif "qwen" in m or "qvq" in m:
        extra = {"qwen": {"thinking_config": {"enable_thinking": True}}}
    elif _is_gpt_thinking_model(model) and not _is_gpt_responses_api_model(model):
        extra = {"reasoning_effort": "high", "max_completion_tokens": 32768, "stream": False}

    for attempt in range(MAX_RETRIES):
        try:
            kwargs = {"model": model, "messages": messages}
            if "robofine" in (model or "").lower():
                kwargs["temperature"] = 0.0
                kwargs["top_p"] = 0.95
                kwargs["max_tokens"] = 32768
            if extra:
                kwargs["extra_body"] = extra
            response = client.chat.completions.create(**kwargs)
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
            err_str = str(e)
            ctx_match = re.search(r'Input length \((\d+)\).*maximum context length \((\d+)\)', err_str)
            if ctx_match and not _ctx_retry:
                input_len = int(ctx_match.group(1))
                max_len = int(ctx_match.group(2))
                ratio = max_len / input_len * 0.9
                frame_count = sum(1 for p in image_parts if p.get("type") == "image_url")
                new_count = max(10, int(frame_count * ratio))
                logger.warning(f"Context length exceeded ({input_len} > {max_len}), reducing frames {frame_count} -> {new_count}")
                if has_views:
                    image_parts = _subsample_multiview_parts(image_parts, new_count)
                else:
                    image_parts = _subsample_parts(image_parts, new_count)
                return _call_openai_compatible(client, image_parts, system_prompt, user_prompt, model,
                                               video_urls=video_urls, fps=fps, _ctx_retry=True)
            wait = min(2 ** attempt, 30) if ("429" in err_str or "rate" in err_str.lower()) else 2
            logger.warning(f"OpenAI-compat attempt {attempt+1}: {err_str[:200]}")
            time.sleep(wait)
    return "", {}


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def call_api(
    client,
    image_parts: List[Dict],
    system_prompt: str,
    user_prompt: str,
    model: str,
    video_urls: List[str] = None,
    fps: float = None,
) -> Tuple[str, Dict]:
    """Call VLM API with automatic model-specific dispatch and thinking enabled."""

    if _is_gemini_native_model(model):
        return _call_gemini_native(client, image_parts, system_prompt, user_prompt, model,
                                   video_urls=video_urls, fps=fps)
    if _is_gpt_responses_api_model(model):
        return _call_gpt_responses(client, image_parts, system_prompt, user_prompt, model)
    if _is_doubao_model(model):
        return _call_doubao_responses(client, image_parts, system_prompt, user_prompt, model,
                                      video_urls=video_urls)
    return _call_openai_compatible(client, image_parts, system_prompt, user_prompt, model,
                                   video_urls=video_urls, fps=fps)


def extract_json_from_response(text: str) -> Optional[Dict]:
    """Extract JSON object from model response (handles ```json blocks, <think> tags, etc.)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None
