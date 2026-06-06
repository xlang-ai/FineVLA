#!/usr/bin/env python3
"""
VQA Test: ask VLM questions about robot manipulation videos.

Usage:
    python run_vqa.py --model qwen3-vl-plus                     # video URL mode
    python run_vqa.py --model qwen3-vl-plus --input-type image   # image list mode
    python run_vqa.py --model qwen3-vl-plus --thinking false     # disable thinking
    python run_vqa.py --model qwen3-vl-plus --end 10             # first 10 samples
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
from openai import OpenAI
from tqdm import tqdm

from vqa_prompts import SYSTEM_PROMPT, build_batch_prompt
from vqa_eval import evaluate
from vqa_config import get_view, get_views, get_dataset_dir

# =========================================================================
# Auto-detect project root (works after git clone)
# =========================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../VQAEval
_BASE_DIR = os.path.dirname(_SCRIPT_DIR)                   # .../RoboFine-Bench

# =========================================================================
# Configuration (self-contained, no external config.py dependency)
# =========================================================================

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MAX_FRAMES = 100000
DEFAULT_BATCH_SIZE = 64

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("vqa")

# =========================================================================
# Model detection helpers
# =========================================================================


def _is_gemini_native_model(model: str) -> bool:
    """Detect models that require Gemini native protocol."""
    m = model.lower()
    if m.startswith("vertex_ai.") or m.startswith("ai_studio."):
        return True
    if "gemini-robotics" in m:
        return True
    bare = m.split(".", 1)[-1] if "." in m else m
    if bare.startswith("gemini-3") or bare.startswith("gemini-4"):
        return True
    return False


def _is_qwen_model(model: str) -> bool:
    m = model.lower()
    return "qwen" in m or "qvq" in m


def _is_gpt_thinking_model(model: str) -> bool:
    """Detect GPT thinking/reasoning models (o1, o3, o4-mini, gpt-5, etc.)."""
    m = model.lower()
    if "." in m:
        m = m.split(".", 1)[-1]
    return m.startswith("o3") or m.startswith("o4") or m.startswith("o1") or m.startswith("gpt-5")


# =========================================================================
# Gemini native protocol helpers
# =========================================================================


def _image_parts_to_gemini_parts(image_parts: List[Dict]) -> List[Dict]:
    """Convert OpenAI image_url parts to Gemini native inlineData/fileData parts."""
    parts = []
    for part in image_parts:
        if part.get("type") == "image_url":
            url = part["image_url"]["url"]
            if url.startswith("data:"):
                header, b64_data = url.split(",", 1)
                mime_type = header.split(":")[1].split(";")[0]
                parts.append({"inlineData": {"data": b64_data, "mimeType": mime_type}})
            else:
                parts.append({"fileData": {"fileUri": url, "mimeType": "image/jpeg"}})
    return parts


def _build_gemini_native_request_body(
    model: str, image_parts: List[Dict], system_prompt: str, user_prompt: str,
    video_urls: List[str] = None, fps: float = None,
) -> Dict:
    """Build Gemini native protocol request body for DashScope."""
    if video_urls:
        gemini_parts = []
        for vu in video_urls:
            part = {"fileData": {"fileUri": vu, "mimeType": "video/mp4"}}
            if fps is not None:
                part["videoMetadata"] = {"fps": fps}
            gemini_parts.append(part)
        gemini_parts.append({"text": user_prompt})
    else:
        gemini_parts = _image_parts_to_gemini_parts(image_parts)
        gemini_parts.append({"text": user_prompt})

    body: Dict = {
        "model": model,
        "contents": [{"role": "user", "parts": gemini_parts}],
    }

    if not model.lower().startswith("vertex_ai."):
        body["dashscope_extend_params"] = {"using_native_protocol": True}

    if system_prompt:
        body["system_instruction"] = {"parts": [{"text": system_prompt}]}

    model_suffix = model.lower().split(".", 1)[-1] if "." in model.lower() else model.lower()
    supports_thinking = (
        ("gemini-3" in model_suffix or "gemini-4" in model_suffix)
        and "image-preview" not in model_suffix
    )
    if supports_thinking:
        body["generationConfig"] = {
            "thinkingConfig": {"thinkingLevel": "high", "includeThoughts": False}
        }
    elif "gemini-2.5" in model_suffix:
        body["generationConfig"] = {
            "thinkingConfig": {"thinkingBudget": -1, "includeThoughts": False}
        }
    return body


# =========================================================================
# User content builder (Qwen video format)
# =========================================================================


def _build_user_content(
    image_parts: List[Dict], user_prompt: str, model: str,
    system_prompt: str = "", fps: float = None,
) -> List[Dict]:
    """Build user message content. Qwen models use video format."""
    combined_text = f"{system_prompt.strip()}\n\n{user_prompt}" if system_prompt else user_prompt

    if _is_qwen_model(model) and image_parts:
        video_urls = [p["image_url"]["url"] for p in image_parts if p.get("type") == "image_url"]
        if len(video_urls) >= 4:
            video_part: Dict[str, Any] = {"type": "video", "video": video_urls}
            if fps is not None:
                video_part["fps"] = max(0.1, min(fps, 10.0))
            return [video_part, {"type": "text", "text": combined_text}]
        elif video_urls:
            return [*image_parts, {"type": "text", "text": combined_text}]
    if system_prompt:
        return [*image_parts, {"type": "text", "text": combined_text}]
    return [*image_parts, {"type": "text", "text": user_prompt}]


# =========================================================================
# API client creation
# =========================================================================


def create_api_client(api_key: str = None, base_url: str = None):
    """Create OpenAI-compatible API client."""
    api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    base_url = base_url or DEFAULT_BASE_URL
    http_client = httpx.Client(timeout=httpx.Timeout(600.0, connect=60.0))
    return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)


# =========================================================================
# Self-contained video decode (requires only PyAV + Pillow)
# =========================================================================

import io
import base64
from PIL import Image as PILImage


def _decode_video_to_pil(
    video_path: str, target_fps: float = 2.0, max_frames: int = 500,
) -> List["PILImage.Image"]:
    """Decode a video file to a list of PIL.Image frames."""
    import av
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        total = stream.frames or 0
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        if total <= 0:
            for _ in container.decode(video=0):
                total += 1
            container.seek(0)
        interval = max(1, int(round(fps / target_fps))) if target_fps > 0 else 1
        indices = list(range(0, total, interval))
        if len(indices) > max_frames:
            indices = [int(i * (total - 1) / (max_frames - 1)) for i in range(max_frames)]
        target_set = set(indices)
        frames = []
        for idx, frame in enumerate(container.decode(video=0)):
            if idx in target_set:
                frames.append(PILImage.fromarray(frame.to_ndarray(format="rgb24")))
        return frames
    finally:
        container.close()


def _load_pil_frames(sample: Dict, fps: float = 2.0, frames_dir: str = None) -> Tuple[List, str]:
    """Load video frames as PIL.Image list for custom model inference.

    Returns (frames, description_string).
    """
    import glob as _glob

    sid = sample.get("sample_id", "")
    dataset = sample.get("dataset", "")

    # Priority 1: pre-extracted frames directory
    if frames_dir:
        sample_dir = os.path.join(frames_dir, sid)
        if os.path.isdir(sample_dir):
            files = sorted(_glob.glob(os.path.join(sample_dir, "*.jpg")) +
                           _glob.glob(os.path.join(sample_dir, "*.png")))
            if files:
                frames = [PILImage.open(f).convert("RGB") for f in files]
                return frames, f"frames_dir({len(frames)}f)"

    # Priority 2: local video files (EvalData/Videos/)
    meta = sample.get("meta", {})
    view_names = meta.get("view_names", [])
    if not view_names:
        view_names = get_views(dataset, sample_meta=meta)
    if not view_names:
        view_names = [get_view(dataset) or "video"]

    all_frames = []
    for vname in view_names:
        vpath = os.path.join(VQA_VIDEO_DIR, dataset, sid, f"{vname}.mp4")
        if os.path.exists(vpath):
            try:
                view_frames = _decode_video_to_pil(vpath, target_fps=fps)
                all_frames.extend(view_frames)
            except Exception as e:
                logger.warning(f"  {sid}/{vname}: failed to decode local video: {e}")

    if all_frames:
        return all_frames, f"local({len(all_frames)}f, fps={fps})"

    # Priority 3: download video from EvalSets.json view URLs (fallback)
    views = sample.get("views", {})
    import tempfile
    for vname, vurl in views.items():
        if not isinstance(vurl, str) or not vurl.startswith("http"):
            continue
        try:
            dl = httpx.Client(timeout=httpx.Timeout(120.0, connect=30.0))
            resp = dl.get(vurl, follow_redirects=True)
            resp.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(resp.content)
                tmp_path = tmp.name
            view_frames = _decode_video_to_pil(tmp_path, target_fps=fps)
            all_frames.extend(view_frames)
            os.unlink(tmp_path)
        except Exception as e:
            logger.warning(f"  {sid}/{vname}: failed to download video: {e}")

    if all_frames:
        return all_frames, f"downloaded({len(all_frames)}f, fps={fps})"
    return [], ""


def _decode_video_to_parts(
    video_path: str, target_fps: float = 2.0, max_frames: int = 500,
    resize_width: int = 512, jpeg_quality: int = 75,
) -> List[Dict]:
    """Decode a local or temp video file to base64 image_url parts using PyAV."""
    import av
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        total = stream.frames or 0
        fps = float(stream.average_rate) if stream.average_rate else 30.0

        if total <= 0:
            for _ in container.decode(video=0):
                total += 1
            container.seek(0)

        interval = max(1, int(round(fps / target_fps))) if target_fps > 0 else 1
        indices = list(range(0, total, interval))
        if len(indices) > max_frames:
            indices = [int(i * (total - 1) / (max_frames - 1)) for i in range(max_frames)]
        target_set = set(indices)

        parts = []
        for idx, frame in enumerate(container.decode(video=0)):
            if idx in target_set:
                img = PILImage.fromarray(frame.to_ndarray(format="rgb24"))
                w, h = img.size
                if resize_width and w > resize_width:
                    img = img.resize((resize_width, int(h * resize_width / w)), PILImage.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=jpeg_quality)
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        return parts
    finally:
        container.close()


def _download_video_to_parts(
    video_url: str, target_fps: float = 2.0, max_frames: int = 500,
) -> Tuple[List[Dict], str]:
    """Download video from OSS URL → decode to base64 image_url parts.

    This is the slowest mode but works with just EvalSets.json (no local videos,
    no pre-uploaded frames). Useful for external collaborators after git clone.
    """
    import tempfile
    try:
        dl_client = httpx.Client(timeout=httpx.Timeout(120.0, connect=30.0))
        resp = dl_client.get(video_url, follow_redirects=True)
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"Failed to download video: {e}")
        return [], ""

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name
        parts = _decode_video_to_parts(tmp_path, target_fps=target_fps, max_frames=max_frames)
        return parts, f"url_decode({len(parts)}f)"
    except Exception as e:
        logger.warning(f"Failed to decode downloaded video: {e}")
        return [], ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

# =========================================================================
# Defaults (all relative to _BASE_DIR)
# =========================================================================

DEFAULT_QA_FILE = os.path.join(_BASE_DIR, "EvalData", "QAEvalSets.json")
DEFAULT_INPUT_FILE = os.path.join(_BASE_DIR, "EvalData", "EvalSets.json")
DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "results")
VQA_VIDEO_DIR = os.path.join(_BASE_DIR, "EvalData", "Videos")


# =========================================================================
# Data loading
# =========================================================================

def load_qa(qa_path: str) -> Dict[str, List[Dict]]:
    """Load QA file, return {sample_id: [qa_items]}."""
    with open(qa_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    qa_map = {}
    for entry in data:
        sid = entry.get("sample_id", "")
        if entry.get("status") == "ok" and entry.get("qas"):
            qa_map[sid] = entry["qas"]
    return qa_map


def load_samples(input_path: str) -> Dict[str, Dict]:
    """Load evalsets input (JSON array or JSONL), return {sample_id: sample_dict}."""
    samples = {}
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        # JSON array (EvalSets.json format)
        for s in json.loads(content):
            samples[s.get("sample_id", "")] = s
    else:
        # JSONL format (legacy)
        for line in content.split("\n"):
            line = line.strip()
            if line:
                s = json.loads(line)
                samples[s.get("sample_id", "")] = s
    return samples


def load_completed(output_path: str) -> Tuple[Set[str], Set[str]]:
    """Load question_ids for resume. Returns (success_ids, failed_sample_ids).

    Only questions with call_success=True are considered done.
    Failed samples (call_success=False or missing) will be retried.
    """
    success_ids = set()
    failed_sids = set()
    if not os.path.exists(output_path):
        return success_ids, failed_sids
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("_type") == "summary":
                    continue
                qid = r.get("question_id", "")
                sid = r.get("sample_id", "")
                if not qid:
                    continue
                if r.get("call_success", False):
                    success_ids.add(qid)
                else:
                    failed_sids.add(sid)
            except json.JSONDecodeError:
                continue
    return success_ids, failed_sids


# =========================================================================
# Image parts — video decode to base64
# =========================================================================


def get_multi_view_image_parts(
    sample: Dict, fps: float = 2.0,
) -> Tuple[List[Dict], str]:
    """Get multi-view frame parts with [View: xxx] text labels.

    Priority:
      1. Local video decode from VQA_VIDEO_DIR -> base64
      2. Download video from EvalSets.json view URLs -> decode
    """
    sid = sample.get("sample_id", "")
    dataset = sample.get("dataset", "")
    meta = sample.get("meta", {})
    view_names = meta.get("view_names", [])
    if not view_names:
        view_names = get_views(dataset, sample_meta=meta)
    if not view_names or len(view_names) < 2:
        return [], ""

    # Priority 1: Local video decode -> base64
    all_parts = []
    desc_parts = []
    for view_name in view_names:
        vpath = os.path.join(VQA_VIDEO_DIR, dataset, sid, f"{view_name}.mp4")
        if not os.path.exists(vpath):
            continue
        try:
            frames = _decode_video_to_parts(vpath, target_fps=fps, max_frames=500)
        except Exception as e:
            logger.warning(f"Failed to decode {view_name} for {sid}: {e}")
            continue
        if frames:
            all_parts.append({"type": "text", "text": f"[View: {view_name}]"})
            all_parts.extend(frames)
            desc_parts.append(f"{view_name}({len(frames)}f)")

    if all_parts:
        desc = f"{'+'.join(desc_parts)}(local)"
        return all_parts, desc

    # Priority 2: Download video from EvalSets.json view URLs
    views_map = sample.get("views", {})
    if views_map and isinstance(views_map, dict):
        all_parts = []
        desc_parts = []
        for vname in view_names:
            url = views_map.get(vname, "")
            if not url or not url.startswith("http"):
                continue
            parts, _ = _download_video_to_parts(url, target_fps=fps, max_frames=250)
            if parts:
                all_parts.append({"type": "text", "text": f"[View: {vname}]"})
                all_parts.extend(parts)
                desc_parts.append(f"{vname}({len(parts)}f)")
        if all_parts:
            desc = f"{'+'.join(desc_parts)}(url)"
            return all_parts, desc

    return [], ""


def get_image_parts(
    sample: Dict, fps: float = 2.0,
) -> Tuple[List[Dict], str]:
    """Get image_url parts for a sample — single view.

    Priority:
      1. Local video decode
      2. Download from EvalSets.json view URLs
    """
    sid = sample.get("sample_id", "")
    dataset = sample.get("dataset", "")
    meta = sample.get("meta", {})

    view_name = get_view(dataset)

    # For dynamic-view datasets (RoboMIND etc.), resolve from sample meta
    if not view_name:
        views = get_views(dataset, sample_meta=meta)
        if views:
            view_name = views[0]

    if not view_name:
        return [], ""

    # Priority 1: Local video decode → base64
    video_path = os.path.join(VQA_VIDEO_DIR, dataset, sid, f"{view_name}.mp4")
    if os.path.exists(video_path):
        try:
            parts = _decode_video_to_parts(video_path, target_fps=fps, max_frames=500)
            if parts:
                return parts, f"{view_name}({len(parts)} frames, local)"
        except Exception as e:
            logger.warning(f"Failed to load {view_name} for {sid}: {e}")

    # Priority 2: Download from EvalSets.json view URLs
    views_map = sample.get("views", {})
    if views_map and isinstance(views_map, dict):
        for vname_key in [view_name, *list(views_map.keys())[:1]]:
            url = views_map.get(vname_key, "")
            if url and url.startswith("http"):
                parts, desc = _download_video_to_parts(url, target_fps=fps, max_frames=500)
                if parts:
                    return parts, f"{vname_key}({desc})"

    return [], ""


# =========================================================================
# Per-process API client
# =========================================================================

_process_client = None


def _get_client(api_key: str = None, base_url: str = None):
    global _process_client
    if _process_client is None:
        _process_client = create_api_client(api_key=api_key, base_url=base_url)
    return _process_client


def _worker_init(api_key: str = None, base_url: str = None):
    _get_client(api_key=api_key, base_url=base_url)


# =========================================================================
# Process single question
# =========================================================================

def _parse_json_answers(response: str) -> Dict[str, str]:
    """Extract {question_id: answer} from model JSON response."""
    import re
    # Try to find JSON block in response
    # Look for ```json ... ``` or raw { ... }
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        # Find the outermost { ... }
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            text = json_match.group(0)
        else:
            return {}

    try:
        data = json.loads(text)
        answers = data.get("answers", data)
        if isinstance(answers, dict):
            return {k: str(v).strip() for k, v in answers.items()}
    except json.JSONDecodeError:
        pass
    return {}


def _is_gpt_responses_api_model(model: str) -> bool:
    """Detect GPT models that require Responses API format (input instead of messages).

    gpt-5.4-pro uses Responses API through DashScope, while gpt-5.4 uses Chat Completions.
    """
    m = model.lower()
    if "." in m:
        m = m.split(".", 1)[-1]
    # gpt-5.4-pro, gpt-5.4-pro-xxx etc.
    return "gpt-5" in m and "pro" in m


def _is_doubao_model(model: str) -> bool:
    """Detect Doubao models (doubao.doubao-seed-xxx or doubao-seed-xxx)."""
    m = model.lower()
    return "doubao" in m


def _build_thinking_extra(model: str) -> Optional[Dict]:
    """Build extra_body for thinking/reasoning mode."""
    m = model.lower()
    if "qwen" in m or "qvq" in m:
        return {"qwen": {"thinking_config": {"enable_thinking": True}}}
    if _is_gpt_thinking_model(model) and not _is_gpt_responses_api_model(model):
        return {"reasoning_effort": "high", "max_completion_tokens": 32768, "stream": False}
    # Gemini thinking is handled in native protocol body, not extra_body
    return None


def _subsample_parts(image_parts: List[Dict], max_frames: int) -> List[Dict]:
    """Uniformly subsample image_parts to max_frames if too many."""
    if len(image_parts) <= max_frames:
        return image_parts
    # Keep first and last, uniformly sample the rest
    indices = [int(i * (len(image_parts) - 1) / (max_frames - 1)) for i in range(max_frames)]
    return [image_parts[i] for i in indices]


def _subsample_multiview_parts(image_parts: List[Dict], max_total_frames: int) -> List[Dict]:
    """Uniformly subsample multi-view image_parts (with [View: xxx] text labels).

    Splits parts by view, proportionally subsamples each view so total ≤ max_total_frames,
    then reassembles with view labels preserved.
    """
    frame_count = sum(1 for p in image_parts if p.get("type") == "image_url")
    if frame_count <= max_total_frames:
        return image_parts

    views = []  # list of (label_part, [frame_parts])
    current_label = None
    current_frames = []
    for p in image_parts:
        if p.get("type") == "text" and str(p.get("text", "")).startswith("[View:"):
            if current_label is not None or current_frames:
                views.append((current_label, current_frames))
            current_label = p
            current_frames = []
        elif p.get("type") == "image_url":
            current_frames.append(p)
    if current_label is not None or current_frames:
        views.append((current_label, current_frames))

    total_orig = sum(len(frames) for _, frames in views)
    result = []
    for label, frames in views:
        quota = max(1, int(len(frames) / total_orig * max_total_frames))
        sampled = _subsample_parts(frames, quota)
        if label is not None:
            result.append(label)
        result.extend(sampled)

    logger.info(f"Subsampled multi-view: {frame_count} → "
                f"{sum(1 for p in result if p.get('type') == 'image_url')} frames "
                f"(max {max_total_frames})")
    return result


# Maximum frames for doubao models (image tokens limit ~128K)
DOUBAO_MAX_FRAMES = 100

# Maximum frames for GPT models (API limit: 500 images per request)
GPT_MAX_FRAMES = 490


def _call_doubao_responses_api(
    client, image_parts: List[Dict], system_prompt: str, user_prompt: str,
    model: str, thinking: bool = True, max_retries: int = 5,
    video_urls: List[str] = None,
) -> Tuple[str, Dict]:
    """Call Doubao API (doubao-seed-xxx) via DashScope.

    Two modes:
      - video_urls provided: Chat Completions format with video_url type (lower tokens)
      - no video_urls: Responses API format with per-frame input_image (legacy)
    """
    import httpx

    api_key = client.api_key
    base_url = str(client.base_url).rstrip("/")
    url = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    http_client = httpx.Client(timeout=httpx.Timeout(600.0, connect=60.0))

    use_video_mode = bool(video_urls)

    if use_video_mode:
        # Chat/Completions endpoint: model name needs -completion suffix
        video_model = model if model.endswith("-completion") else model + "-completion"
        combined_text = f"{system_prompt.strip()}\n\n{user_prompt}" if system_prompt else user_prompt
        user_content = []
        for vu in video_urls:
            user_content.append({"type": "video_url", "video_url": {"url": vu}})
        user_content.append({"type": "text", "text": combined_text})
        body = {
            "model": video_model,
            "messages": [{"role": "user", "content": user_content}],
        }
        if thinking:
            body["reasoning"] = {"effort": "high"}
        logger.info(f"Doubao: video_url mode (completion endpoint), model={video_model}, {len(video_urls)} video(s)")
    else:
        _has_views = any(
            p.get("type") == "text" and str(p.get("text", "")).startswith("[View:")
            for p in image_parts
        )
        if _has_views:
            parts = _subsample_multiview_parts(image_parts, DOUBAO_MAX_FRAMES)
        else:
            parts = _subsample_parts(image_parts, DOUBAO_MAX_FRAMES)
        if len(parts) < len(image_parts):
            frame_count = sum(1 for p in parts if p.get("type") == "image_url")
            logger.info(f"Doubao: subsampled {len(image_parts)} → {frame_count} frames")
        user_content = []
        if system_prompt:
            user_content.append({"type": "input_text", "text": system_prompt.strip()})
        for part in parts:
            if part.get("type") == "text":
                user_content.append({"type": "input_text", "text": part["text"]})
            elif part.get("type") == "image_url":
                img_url = part["image_url"]["url"]
                user_content.append({"type": "input_image", "image_url": img_url})
        user_content.append({"type": "input_text", "text": user_prompt})
        body = {
            "model": model,
            "input": [{"type": "message", "role": "user", "content": user_content}],
        }
        if thinking:
            body["reasoning"] = {"effort": "high"}

    for attempt in range(max_retries):
        try:
            resp = http_client.post(url, headers=headers, json=body)
            if resp.status_code != 200:
                err_text = resp.text[:500]
                if not use_video_mode and ("exceed max" in err_text.lower() or "max message tokens" in err_text.lower()):
                    current_frames = sum(1 for c in user_content if c.get("type") == "input_image")
                    new_limit = current_frames // 2
                    if new_limit >= 10:
                        logger.warning(f"Doubao: token limit exceeded, reducing {current_frames} → {new_limit} frames")
                        if _has_views:
                            reduced = _subsample_multiview_parts(parts, new_limit)
                        else:
                            reduced = _subsample_parts(parts, new_limit)
                        user_content = []
                        if system_prompt:
                            user_content.append({"type": "input_text", "text": system_prompt.strip()})
                        for p in reduced:
                            if p.get("type") == "text":
                                user_content.append({"type": "input_text", "text": p["text"]})
                            elif p.get("type") == "image_url":
                                user_content.append({"type": "input_image", "image_url": p["image_url"]["url"]})
                        user_content.append({"type": "input_text", "text": user_prompt})
                        body["input"][0]["content"] = user_content
                        parts = reduced
                        continue
                raise Exception(f"HTTP {resp.status_code}: {err_text}")
            data = resp.json()

            if data.get("error"):
                raise Exception(f"API error: {data['error'].get('message', '')}")

            content = ""
            if use_video_mode:
                # Chat Completions response format
                choices = data.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = "".join(p.get("text", "") for p in content if p.get("type") == "text")
            else:
                # Responses API output format
                for item in data.get("output", []):
                    if item.get("type") == "message":
                        for c in item.get("content", []):
                            if c.get("type") == "output_text":
                                content += c.get("text", "")

            usage = data.get("usage", {})
            return content.strip(), {
                "prompt_tokens": usage.get("input_tokens", usage.get("prompt_tokens", 0)),
                "completion_tokens": usage.get("output_tokens", usage.get("completion_tokens", 0)),
                "total_tokens": usage.get("total_tokens",
                    usage.get("input_tokens", usage.get("prompt_tokens", 0))
                    + usage.get("output_tokens", usage.get("completion_tokens", 0))),
            }
        except Exception as e:
            err = str(e)
            is_rate = "429" in err or "rate" in err.lower()
            wait = min(2 ** attempt, 30) if is_rate else 2
            logger.warning(f"Doubao API error (attempt {attempt+1}): {err[:200]}")
            time.sleep(wait)
    return "", {}


def _call_gpt_responses_api(
    client, image_parts: List[Dict], system_prompt: str, user_prompt: str,
    model: str, thinking: bool = True, max_retries: int = 5,
) -> Tuple[str, Dict]:
    """Call GPT Responses API (gpt-5.4-pro etc.) via DashScope.

    These models require 'input' param instead of 'messages', and use
    'input_text'/'input_image' content types instead of 'text'/'image_url'.
    """
    import httpx

    # Subsample to stay within GPT 500-image limit
    _has_views = any(
        p.get("type") == "text" and str(p.get("text", "")).startswith("[View:")
        for p in image_parts
    )
    if _has_views:
        image_parts = _subsample_multiview_parts(image_parts, GPT_MAX_FRAMES)
    else:
        image_parts = _subsample_parts(image_parts, GPT_MAX_FRAMES)

    api_key = client.api_key
    base_url = str(client.base_url).rstrip("/")
    url = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    http_client = httpx.Client(timeout=httpx.Timeout(600.0, connect=60.0))

    # Build input in Responses API format
    # Convert image_parts from Chat format to Responses format
    user_content = []
    for part in image_parts:
        if part.get("type") == "image_url":
            img_url = part["image_url"]["url"]
            if img_url.startswith("http"):
                user_content.append({"type": "input_image", "image_url": img_url})
            else:
                # base64: data:image/jpeg;base64,...
                user_content.append({"type": "input_image", "image_url": img_url})
    user_content.append({"type": "input_text", "text": user_prompt})

    input_messages = [
        {"role": "developer", "content": [{"type": "input_text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]

    body = {"model": model, "input": input_messages}
    if thinking:
        body["reasoning"] = {"effort": "high"}

    for attempt in range(max_retries):
        try:
            resp = http_client.post(url, headers=headers, json=body)
            if resp.status_code != 200:
                raise Exception(f"HTTP {resp.status_code}: {resp.text[:500]}")
            data = resp.json()

            # Check for API-level error
            if data.get("error") and data.get("status") == "failed":
                raise Exception(f"API error: {data['error'].get('message', '')}")

            # Extract text from Responses API output format
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
            err = str(e)
            is_rate = "429" in err or "rate" in err.lower()
            wait = min(2 ** attempt, 30) if is_rate else 2
            logger.warning(f"GPT Responses API error (attempt {attempt+1}): {err[:200]}")
            time.sleep(wait)
    return "", {}


def _call_vqa_api(
    client, image_parts: List[Dict], system_prompt: str, user_prompt: str,
    model: str, thinking: bool = True, max_retries: int = 5,
    video_urls: List[str] = None, input_type: str = "image", fps: float = 2.0,
) -> Tuple[str, Dict]:
    """Call VLM API with optional thinking/reasoning control.

    Args:
        thinking: if True, enable thinking/reasoning for supported models;
                  if False, call without any thinking config (faster).
        video_urls: optional list of video URLs for models that support video_url type.
    """
    import httpx

    # Gemini native models use their own protocol
    if _is_gemini_native_model(model):
        _gem_video = video_urls if input_type == "video" else None
        body = _build_gemini_native_request_body(
            model, image_parts, system_prompt, user_prompt,
            video_urls=_gem_video, fps=fps,
        )
        if not thinking:
            body.pop("generationConfig", None)  # remove thinkingConfig

        api_key = client.api_key
        base_url = str(client.base_url).rstrip("/")
        url = f"{base_url}/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        http_client = httpx.Client(timeout=httpx.Timeout(600.0, connect=60.0))

        for attempt in range(max_retries):
            try:
                resp = http_client.post(url, headers=headers, json=body)
                if resp.status_code != 200:
                    raise Exception(f"HTTP {resp.status_code}: {resp.text[:500]}")
                data = resp.json()
                content = ""
                if "candidates" in data and data["candidates"]:
                    # Gemini native response: candidates[0].content.parts[].text
                    parts = data["candidates"][0].get("content", {}).get("parts", [])
                    content = "".join(p.get("text", "") for p in parts if "text" in p)
                elif "choices" in data and data["choices"]:
                    # Fallback: OpenAI-compatible format
                    msg = data["choices"][0].get("message", {})
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = "".join(p.get("text", "") for p in content if p.get("type") == "text")
                usage_meta = data.get("usageMetadata", {})
                if usage_meta:
                    token_usage = {
                        "prompt_tokens": usage_meta.get("promptTokenCount", 0),
                        "completion_tokens": usage_meta.get("candidatesTokenCount", 0),
                        "total_tokens": usage_meta.get("totalTokenCount", 0),
                    }
                else:
                    usage = data.get("usage", {})
                    token_usage = {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    }
                return content.strip(), token_usage
            except Exception as e:
                err = str(e)
                is_rate = "429" in err or "rate" in err.lower()
                wait = min(2 ** attempt, 30) if is_rate else 2
                logger.warning(f"Gemini API error (attempt {attempt+1}): {err[:200]}")
                time.sleep(wait)
        return "", {}

    # GPT Responses API models (e.g. gpt-5.4-pro) — uses 'input' instead of 'messages'
    if _is_gpt_responses_api_model(model):
        return _call_gpt_responses_api(
            client, image_parts, system_prompt, user_prompt,
            model=model, thinking=thinking, max_retries=max_retries,
        )

    # Doubao Responses API models (e.g. doubao-seed-2-0-pro-260215)
    if _is_doubao_model(model):
        return _call_doubao_responses_api(
            client, image_parts, system_prompt, user_prompt,
            model=model, thinking=thinking, max_retries=max_retries,
            video_urls=video_urls,
        )

    # OpenAI-compatible path (Qwen, GPT, non-native Gemini)

    # GPT models: enforce 500-image API limit via subsample
    if _is_gpt_thinking_model(model) or "gpt" in model.lower():
        _has_views = any(
            p.get("type") == "text" and str(p.get("text", "")).startswith("[View:")
            for p in image_parts
        )
        if _has_views:
            image_parts = _subsample_multiview_parts(image_parts, GPT_MAX_FRAMES)
        else:
            image_parts = _subsample_parts(image_parts, GPT_MAX_FRAMES)

    # Detect if image_parts already contain multi-view content (text labels interleaved with frames)
    _has_view_labels = any(
        p.get("type") == "text" and str(p.get("text", "")).startswith("[View:")
        for p in image_parts
    )
    # Detect if image_parts are OSS URLs (not base64)
    _is_oss = False
    if not _has_view_labels and image_parts:
        first_url = image_parts[0].get("image_url", {}).get("url", "")
        _is_oss = first_url.startswith("http")

    if input_type == "video" and video_urls and _is_qwen_model(model):
        combined_text = f"{system_prompt.strip()}\n\n{user_prompt}" if system_prompt else user_prompt
        content = []
        for vu in video_urls:
            vpart: Dict[str, Any] = {"type": "video", "video": vu}
            if fps is not None:
                vpart["fps"] = fps
            content.append(vpart)
        content.append({"type": "text", "text": combined_text})
        messages = [{"role": "user", "content": content}]
        logger.info(f"Qwen: video URL mode, {len(video_urls)} video(s), fps={fps}")
    elif _is_qwen_model(model) and _has_view_labels:
        # Multi-view with text labels: group frames per view into video blocks
        combined_text = f"{system_prompt.strip()}\n\n{user_prompt}" if system_prompt else user_prompt
        content = []
        current_frames = []
        for p in image_parts:
            if p.get("type") == "text":
                # Flush previous view's frames as video block
                if current_frames:
                    url_list = [f["image_url"]["url"] for f in current_frames]
                    content.append({"type": "video", "video": url_list})
                    current_frames = []
                content.append(p)  # [View: xxx] label
            elif p.get("type") == "image_url":
                current_frames.append(p)
        # Flush last view
        if current_frames:
            url_list = [f["image_url"]["url"] for f in current_frames]
            content.append({"type": "video", "video": url_list})
        content.append({"type": "text", "text": combined_text})
        messages = [{"role": "user", "content": content}]
    elif _is_qwen_model(model) and _is_oss:
        # Qwen + OSS URLs: use {"type": "video", "video": [url1, url2, ...]} format
        # This avoids 250-frame base64 limit and is much faster
        url_list = [p["image_url"]["url"] for p in image_parts if p.get("type") == "image_url"]
        combined_text = f"{system_prompt.strip()}\n\n{user_prompt}" if system_prompt else user_prompt
        messages = [{"role": "user", "content": [
            {"type": "video", "video": url_list},
            {"type": "text", "text": combined_text},
        ]}]
    elif _is_qwen_model(model):
        # Qwen + base64: use _build_user_content (wraps as video type)
        messages = [{"role": "user", "content": _build_user_content(
            image_parts, user_prompt, model, system_prompt=system_prompt)}]
    elif _is_gpt_thinking_model(model):
        # GPT reasoning models (o1/o3/gpt-5): use "developer" role instead of "system"
        messages = [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": [*image_parts, {"type": "text", "text": user_prompt}]},
        ]
    else:
        # Gemini (non-native) / other models: standard format
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [*image_parts, {"type": "text", "text": user_prompt}]},
        ]

    extra = _build_thinking_extra(model) if thinking else None

    for attempt in range(max_retries):
        try:
            kwargs = {"model": model, "messages": messages}
            if extra:
                kwargs["extra_body"] = extra
            response = client.chat.completions.create(**kwargs)
            if not hasattr(response, 'choices') or not response.choices:
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
            usage = getattr(response, 'usage', None)
            token_info = {}
            if usage:
                token_info = {
                    "prompt_tokens": getattr(usage, 'prompt_tokens', 0) or 0,
                    "completion_tokens": getattr(usage, 'completion_tokens', 0) or 0,
                    "total_tokens": getattr(usage, 'total_tokens', 0) or 0,
                }
            return content or "", token_info
        except Exception as e:
            err = str(e)
            is_rate = "429" in err or "rate" in err.lower()
            wait = min(2 ** attempt, 30) if is_rate else 2
            logger.warning(f"API error (attempt {attempt+1}): {err[:200]}")
            time.sleep(wait)

    return "", {}


def process_sample(
    sid: str,
    qas: List[Dict],
    image_parts: List[Dict],
    model: str,
    base_url: str = None,
    thinking: bool = True,
    video_urls: List[str] = None,
    input_type: str = "image",
    fps: float = 2.0,
    custom_model=None,
    pil_frames: List = None,
) -> List[Dict]:
    """Process all questions for one sample in a single API call.

    Returns list of result dicts (one per question).
    """
    # Build batch prompt with all questions
    prompt, extras_by_qid = build_batch_prompt(qas)

    start = time.time()

    if custom_model is not None and pil_frames is not None:
        response, usage = custom_model.generate(pil_frames, prompt, SYSTEM_PROMPT)
    else:
        client = _get_client(base_url=base_url)
        response, usage = _call_vqa_api(
            client, image_parts, SYSTEM_PROMPT, prompt,
            model=model, thinking=thinking, max_retries=5,
            video_urls=video_urls, input_type=input_type, fps=fps,
        )
    elapsed = time.time() - start

    # Parse JSON answers from response
    parsed_answers = _parse_json_answers(response) if response else {}

    # Remap short keys (Q1, Q2, ...) or bracketed keys ([qid]) → full question_id
    if parsed_answers:
        short_to_qid = {f"Q{i}": qa["question_id"] for i, qa in enumerate(qas, 1)}
        valid_qids = {qa["question_id"] for qa in qas}
        remapped = {}
        for k, v in parsed_answers.items():
            normalized = k.strip().strip("[]")
            if normalized in short_to_qid:
                remapped[short_to_qid[normalized]] = v
            elif normalized in valid_qids:
                remapped[normalized] = v
            else:
                remapped[k.strip()] = v
        parsed_answers = remapped

    # Determine if the API call succeeded (got response and parsed at least 1 answer)
    call_success = bool(response and parsed_answers)

    results = []
    for qa in qas:
        qid = qa["question_id"]
        answer_text = parsed_answers.get(qid, "")
        prompt_extra = extras_by_qid.get(qid, {})

        correct, parsed = evaluate(answer_text, qa, prompt_extra=prompt_extra)

        # GT display
        gt_display = qa["answer"]
        if prompt_extra.get("gt_letter"):
            gt_display = f"{prompt_extra['gt_letter']}. {qa['answer']}"

        results.append({
            "question_id": qid,
            "sample_id": sid,
            "answer_type": qa["answer_type"],
            "capability": qa.get("capability", ""),
            "error_type": qa.get("error_type", ""),
            "question": qa["question"],
            "gt_answer": gt_display,
            "model_answer_raw": answer_text,
            "model_answer_parsed": parsed,
            "correct": correct,
            "call_success": call_success,
            "elapsed": round(elapsed / len(qas), 2),
            "token_usage": usage,
        })

    return results


# =========================================================================
# CLI
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="VQA Test on robot manipulation videos")
    p.add_argument("--qa", default=DEFAULT_QA_FILE, help="QA JSON file")
    p.add_argument("--input", default=DEFAULT_INPUT_FILE, help="Eval set JSONL")
    p.add_argument("--model", default="qwen3-vl-plus")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--output", default=None, help="Output JSONL path (auto-generated if not set)")
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--thinking", default="true", choices=["true", "false"],
                   help="Enable thinking/reasoning mode (default: true)")
    p.add_argument("--round", type=int, default=None,
                   help="Round number for multi-round evaluation (affects output filename)")
    p.add_argument("--input-type", default="image", choices=["image", "video"],
                   help="Input mode: 'image' sends per-frame images; 'video' sends .mp4 URL directly (lower tokens)")
    p.add_argument("--fps", type=float, default=2.0,
                   help="FPS hint for video mode (only used when --input-type=video)")
    p.add_argument("--model-class", default=None,
                   help="Custom model class path, e.g. 'models.example_local_model.LocalVLMExample'")
    p.add_argument("--model-path", default="",
                   help="Path passed to custom model constructor (used with --model-class)")
    p.add_argument("--frames-dir", default=None,
                   help="Pre-extracted frames directory (used with --model-class)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _remove_failed_records(output_path: str, failed_sids: Set[str]):
    """Remove records of failed samples from output file so they can be retried."""
    kept = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("_type") == "summary":
                    continue  # also remove old summary, will be re-generated
                sid = r.get("sample_id", "")
                if sid in failed_sids:
                    continue  # remove failed records
                kept.append(line)
            except json.JSONDecodeError:
                continue
    with open(output_path, "w", encoding="utf-8") as f:
        for line in kept:
            f.write(line + "\n")
    logger.info(f"Removed failed records for {len(failed_sids)} samples, kept {len(kept)} records")


def main():
    args = parse_args()

    # ── Load data ──
    logger.info(f"Loading QA: {args.qa}")
    qa_map = load_qa(args.qa)
    logger.info(f"Loaded QA for {len(qa_map)} samples")

    logger.info(f"Loading samples: {args.input}")
    samples = load_samples(args.input)
    logger.info(f"Loaded {len(samples)} samples")

    # ── Build task list: (sample, question) pairs ──
    sample_ids = sorted(set(qa_map.keys()) & set(samples.keys()))
    end = args.end if args.end is not None else len(sample_ids)
    sample_ids = sample_ids[args.start:end]

    tasks = []
    for sid in sample_ids:
        for qa in qa_map[sid]:
            tasks.append({
                "sample_id": sid,
                "qa": qa,
            })

    # ── Load custom model (if specified) ──
    custom_model = None
    if args.model_class:
        module_path, class_name = args.model_class.rsplit(".", 1)
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        custom_model = cls(args.model_path)
        logger.info(f"Loaded custom model: {args.model_class}({args.model_path})")
        if not args.model or args.model == "qwen3-vl-plus":
            args.model = args.model_class  # use class name as model tag

    # ── Summary ──
    type_counts = Counter(t["qa"]["answer_type"] for t in tasks)
    print(f"\n{'='*60}")
    print(f"  VQA Test: {len(tasks)} questions from {len(sample_ids)} samples")
    print(f"  Model: {args.model}")
    thinking = args.thinking == "true"
    if custom_model:
        mode = f"custom model (fps={args.fps})"
        if args.frames_dir:
            mode += f", frames_dir={args.frames_dir}"
    elif args.input_type == "video":
        mode = f"video URL (fps={args.fps})"
    else:
        mode = f"image list (fps={args.fps})"
    print(f"  Mode: {mode}")
    print(f"  Thinking: {thinking}")
    for t, c in type_counts.most_common():
        print(f"    {t}: {c}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("Dry-run complete.")
        return

    # ── Output path ──
    if args.output:
        output_path = args.output
    else:
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        model_tag = args.model.replace("/", "_").replace(".", "_")
        mode_tag = f"_{args.input_type}"
        if args.input_type == "video":
            mode_tag += f"_fps{args.fps:g}"
        if args.round is not None:
            output_path = os.path.join(DEFAULT_OUTPUT_DIR, f"{model_tag}{mode_tag}_round{args.round}_vqa_result.jsonl")
        else:
            output_path = os.path.join(DEFAULT_OUTPUT_DIR, f"{model_tag}{mode_tag}_vqa_result.jsonl")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output: {output_path}")

    # ── Resume ──
    success_ids, failed_sids = load_completed(output_path)
    if success_ids or failed_sids:
        before = len(tasks)
        # Only skip questions that succeeded; failed samples will be retried
        tasks = [t for t in tasks if t["qa"]["question_id"] not in success_ids]
        logger.info(f"Resume: {len(success_ids)} succeeded, {len(failed_sids)} failed samples to retry, {len(tasks)} remaining")

        # Remove failed records from output file so they can be re-written
        if failed_sids:
            _remove_failed_records(output_path, failed_sids)

    if not tasks:
        logger.info("All questions already completed.")
        _print_report(output_path)
        return

    # ── Group tasks by sample ──
    from collections import defaultdict as _ddict
    tasks_by_sample: Dict[str, List[Dict]] = _ddict(list)
    for t in tasks:
        tasks_by_sample[t["sample_id"]].append(t)

    needed_sids = sorted(tasks_by_sample.keys())
    total_questions = len(tasks)

    # ── Run: parallel API calls, one per sample ──
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _worker_init(base_url=args.base_url)
    num_workers = args.num_workers
    success = 0
    fail = 0
    start_time = time.time()
    write_lock = threading.Lock()

    logger.info(f"Running {total_questions} questions from {len(needed_sids)} samples ({num_workers} workers)...")

    def _process_one_sample(sid: str):
        """Load frames + call API for one sample, return list of result dicts."""
        sample = samples[sid]
        sample_qas = [t["qa"] for t in tasks_by_sample[sid]]

        # Custom model path: load PIL frames, call model.generate()
        if custom_model is not None:
            pil_frames, desc = _load_pil_frames(sample, fps=args.fps, frames_dir=args.frames_dir)
            if not pil_frames:
                return [{
                    "question_id": qa["question_id"], "sample_id": sid,
                    "answer_type": qa["answer_type"], "correct": False,
                    "call_success": False, "error": "no frames for custom model",
                } for qa in sample_qas], desc
            results = process_sample(
                sid, sample_qas, [],
                model=args.model, thinking=thinking,
                custom_model=custom_model, pil_frames=pil_frames,
            )
            return results, desc

        # API model path
        sample_video_urls = [u for u in sample.get("views", {}).values()
                            if isinstance(u, str) and u.startswith("http")]

        input_type = args.input_type

        # Doubao: limit to 1 video (primary view) to avoid 413 Request Entity Too Large
        if input_type == "video" and sample_video_urls and _is_doubao_model(args.model):
            sample_video_urls = sample_video_urls[:1]

        if input_type == "video" and sample_video_urls:
            parts = []
            desc = f"video({len(sample_video_urls)} urls, fps={args.fps})"
        else:
            if input_type == "video":
                logger.warning(f"  {sid}: no video URLs found, falling back to image mode")
            parts, desc = get_multi_view_image_parts(sample, fps=args.fps)
            if not parts:
                parts, desc = get_image_parts(sample, fps=args.fps)
            input_type = "image"

        if not parts and not sample_video_urls:
            return [{
                "question_id": qa["question_id"], "sample_id": sid,
                "answer_type": qa["answer_type"], "correct": False,
                "call_success": False, "error": "no frames",
            } for qa in sample_qas], desc

        results = process_sample(
            sid, sample_qas, parts,
            model=args.model, base_url=args.base_url,
            thinking=thinking,
            video_urls=sample_video_urls or None,
            input_type=input_type, fps=args.fps,
        )
        return results, desc

    pbar = tqdm(total=total_questions, desc="VQA", unit="q")
    with open(output_path, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_process_one_sample, sid): sid for sid in needed_sids}
            for fut in as_completed(futures):
                sid = futures[fut]
                try:
                    results, desc = fut.result()
                    if desc:
                        logger.info(f"  {sid}: {desc}")
                except Exception as e:
                    logger.error(f"  {sid}: error: {e}")
                    sample_qas = [t["qa"] for t in tasks_by_sample[sid]]
                    results = [{
                        "question_id": qa["question_id"], "sample_id": sid,
                        "answer_type": qa["answer_type"], "correct": False,
                        "call_success": False, "error": str(e),
                    } for qa in sample_qas]

                with write_lock:
                    for r in results:
                        out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                        if r.get("correct"):
                            success += 1
                        else:
                            fail += 1
                        pbar.update(1)
                    out_f.flush()

                    total_done = success + fail
                    if total_done > 0:
                        pbar.set_postfix(acc=f"{success/total_done:.1%}")

    pbar.close()
    elapsed = time.time() - start_time
    total_done = success + fail
    print(f"\n{'='*60}")
    print(f"  Done in {elapsed:.1f}s ({len(needed_sids)} API calls, {num_workers} workers)")
    print(f"  Correct: {success}, Wrong: {fail}, Accuracy: {success/total_done:.1%}" if total_done else "  No results")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")

    # Write summary record, full report, and update CSV
    _write_summary(output_path, args.model, elapsed)
    _print_report(output_path)
    _update_score_csv()


def _write_summary(output_path: str, model: str, elapsed: float):
    """Append a summary record to the result JSONL with accuracy breakdowns."""
    from collections import defaultdict

    results = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("_type") == "summary":
                continue  # skip old summary if re-running
            results.append(r)

    total = len(results)
    if total == 0:
        return

    correct = sum(1 for r in results if r.get("correct"))

    # By answer_type
    by_type = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        t = r.get("answer_type", "unknown")
        by_type[t]["total"] += 1
        if r.get("correct"):
            by_type[t]["correct"] += 1
    type_acc = {
        t: round(d["correct"] / d["total"], 4) if d["total"] else 0
        for t, d in by_type.items()
    }

    # By capability
    by_cap = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        cap = r.get("capability", "unknown")
        by_cap[cap]["total"] += 1
        if r.get("correct"):
            by_cap[cap]["correct"] += 1
    cap_acc = {
        cap: round(d["correct"] / d["total"], 4) if d["total"] else 0
        for cap, d in by_cap.items()
    }

    # By error_type
    by_err = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        et = r.get("error_type", "unknown")
        by_err[et]["total"] += 1
        if r.get("correct"):
            by_err[et]["correct"] += 1
    err_acc = {
        et: round(d["correct"] / d["total"], 4) if d["total"] else 0
        for et, d in by_err.items()
    }

    summary = {
        "_type": "summary",
        "model": model,
        "total_questions": total,
        "total_correct": correct,
        "overall_accuracy": round(correct / total, 4),
        "elapsed_seconds": round(elapsed, 1),
        "accuracy_by_answer_type": {
            t: {"correct": by_type[t]["correct"], "total": by_type[t]["total"], "accuracy": type_acc[t]}
            for t in sorted(by_type)
        },
        "accuracy_by_capability": {
            cap: {"correct": by_cap[cap]["correct"], "total": by_cap[cap]["total"], "accuracy": cap_acc[cap]}
            for cap in sorted(by_cap)
        },
        "accuracy_by_error_type": {
            et: {"correct": by_err[et]["correct"], "total": by_err[et]["total"], "accuracy": err_acc[et]}
            for et in sorted(by_err)
        },
    }

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    logger.info(f"Summary written to {output_path}")


def _update_score_csv():
    """Update VQATest_Score.csv with all model results."""
    from vqa_report import update_csv
    update_csv()


def _print_report(output_path: str):
    """Print accuracy breakdown from result JSONL."""
    from vqa_report import print_report
    print_report(output_path)


if __name__ == "__main__":
    main()
