#!/usr/bin/env python3
"""
Single-stage video annotation: call a VLM API to decompose robot manipulation
videos into fine-grained steps.

Supports two input modes:
  - video: send video URLs directly to the model (default, lower tokens)
  - image: decode local videos at target FPS, send as base64 image list

Usage:
    # Video URL mode (default, fps=4)
    python run_annotate.py --model qwen3.5-plus --input-type video --fps 4

    # Image mode (decode local videos)
    python run_annotate.py --model openai.gpt-5.4-2026-03-05 --input-type image --fps 4

    # With frame_index.jsonl (pre-uploaded frame URLs)
    python run_annotate.py --model qwen3.5-plus --input-type image --frame-index EvalData/frame_index.jsonl
"""

import argparse
import base64
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from api_call import call_api, create_api_client, extract_json_from_response
from prompts import SINGLE_VIEW_PROMPT, MULTI_VIEW_PROMPT, classify_view

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("annotation")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
DEFAULT_EVALSETS = os.path.join(_BASE_DIR, "EvalData", "EvalSets.json")
DEFAULT_FRAME_INDEX = os.path.join(_BASE_DIR, "EvalData", "frame_index.jsonl")
DEFAULT_OUTPUT_DIR = os.path.join(_BASE_DIR, "CaptionEval", "CaptionResult")
DEFAULT_VIDEO_DIR = os.path.join(_BASE_DIR, "EvalData", "Videos")

_client_lock = threading.Lock()
_client = None


def _get_client(base_url: str = None):
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = create_api_client(base_url=base_url)
    return _client


# ---------------------------------------------------------------------------
# Video decode (for image mode)
# ---------------------------------------------------------------------------

def _decode_video_to_parts(
    video_path: str, fps: float = 4.0, max_frames: int = 500,
    resize_width: int = 512,
) -> List[Dict]:
    """Decode local video file to base64 image_url parts.

    Uses frame-index sampling (consistent with Annotate_Pipeline):
    interval = max(1, round(native_fps / target_fps)).
    """
    try:
        import av
        import numpy as np
    except ImportError:
        raise ImportError("PyAV is required for image mode: pip install av")

    container = av.open(video_path)
    stream = container.streams.video[0]
    total_frames = stream.frames or 0
    native_fps = float(stream.average_rate) if stream.average_rate else 30.0

    if total_frames <= 0:
        for _ in container.decode(video=0):
            total_frames += 1
        container.seek(0)

    interval = max(1, int(round(native_fps / fps))) if fps > 0 else 1
    indices = list(range(0, total_frames, interval))
    if len(indices) > max_frames:
        indices = np.linspace(0, total_frames - 1, max_frames).astype(int).tolist()
    indices_set = set(indices)

    parts = []
    try:
        for frame_idx, frame in enumerate(container.decode(video=0)):
            if frame_idx not in indices_set:
                continue
            img = frame.to_image()
            if resize_width and img.width > resize_width:
                new_h = int(round(img.height * resize_width / img.width))
                img = img.resize((resize_width, new_h))
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
            if frame_idx >= max(indices):
                break
    finally:
        container.close()

    return parts


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_evalsets(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_frame_index(path: str) -> Dict[str, Dict]:
    """Load frame_index.jsonl -> {sample_id: {view_name: {urls: [...], num_frames: N}}}"""
    index = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            index[d["sample_id"]] = d.get("views", {})
    return index


def load_completed(output_path: str) -> Set[str]:
    """Load already completed sample_ids from output JSONL for resume.

    Also cleans up the file: removes entries with call_success=False so that
    only successful results remain (failed ones will be retried).
    """
    if not os.path.exists(output_path):
        return set()

    keep_lines = []
    done = set()
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                sid = r.get("sample_id", "")
                if r.get("call_success"):
                    done.add(sid)
                    keep_lines.append(line)
            except json.JSONDecodeError:
                continue

    removed = 0
    with open(output_path, "r", encoding="utf-8") as f:
        total = sum(1 for l in f if l.strip())
    removed = total - len(keep_lines)

    if removed > 0:
        with open(output_path, "w", encoding="utf-8") as f:
            for line in keep_lines:
                f.write(line + "\n")
        logger.info(f"  Resume cleanup: removed {removed} failed entries, kept {len(keep_lines)} successful")

    return done


# ---------------------------------------------------------------------------
# Image parts construction
# ---------------------------------------------------------------------------

def _filter_views(view_names: List[str]) -> List[str]:
    """Filter views: keep all views (multi-view mode)."""
    return view_names


def build_image_parts(
    view_names: List[str],
    frame_views: Dict,
) -> Tuple[List[Dict], int, int]:
    """Build image_parts list from frame_index views.

    Always includes [View: label] tags. For 3-view samples, only the main view is used.

    Returns: (image_parts, num_views_used, total_frames)
    """
    view_names = _filter_views(view_names)
    image_parts = []
    total_frames = 0
    views_used = 0

    for i, vn in enumerate(view_names):
        view_data = frame_views.get(vn, {})
        urls = view_data.get("urls", [])
        if not urls:
            continue
        views_used += 1

        label = classify_view(vn, i)
        image_parts.append({"type": "text", "text": f"[View: {label}]"})

        for url in urls:
            image_parts.append({"type": "image_url", "image_url": {"url": url}})
            total_frames += 1

    return image_parts, views_used, total_frames


def build_prompt(view_names: List[str], instruction_raw, no_instruction: bool = False) -> Tuple[str, str]:
    """Return (system_prompt, user_prompt) based on view count.

    system_prompt is empty string (prompt is self-contained).
    user_prompt is the full annotation prompt.
    If no_instruction is True, instruction_raw is not appended.
    """
    if isinstance(instruction_raw, list):
        instr = "; ".join(instruction_raw)
    else:
        instr = str(instruction_raw or "")

    if len(view_names) > 1:
        prompt = MULTI_VIEW_PROMPT
    else:
        prompt = SINGLE_VIEW_PROMPT

    if instr and not no_instruction:
        prompt += f"\n\nTask instruction: {instr}"

    return "", prompt


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _json_to_step_list(parsed) -> List[str]:
    """Convert model JSON response into a flat list of step strings.

    Handles:
      - {"Step1": "...", "Step2": "..."}          → values sorted by numeric key
      - {"steps": ["...", "..."]}                  → list value directly
      - ["...", "..."]                             → already a list
      - {"status": "ok", "steps": [...]}           → find the list-valued key
    """
    if parsed is None:
        return []

    if isinstance(parsed, list):
        return [str(s) for s in parsed if s]

    if not isinstance(parsed, dict):
        return []

    # If any value is a list of strings, use it (e.g. "steps": [...])
    for v in parsed.values():
        if isinstance(v, list) and v and isinstance(v[0], str):
            return [s for s in v if s]

    # Otherwise treat as {"Step1": "...", "Step2": "..."} — sort by numeric key
    import re as _re
    def _sort_key(k):
        m = _re.search(r"(\d+)", str(k))
        return int(m.group(1)) if m else 0

    items = sorted(parsed.items(), key=lambda kv: _sort_key(kv[0]))
    return [str(v) for _, v in items if v and not isinstance(v, (dict, list))]


def build_image_parts_from_video(
    sample: Dict, video_dir: str, fps: float = 4.0,
) -> Tuple[List[Dict], int, int]:
    """Build image_parts by decoding local video files.

    Always includes [View: label] tags. For 3-view samples, only the main view is used.

    Returns: (image_parts, num_views_used, total_frames)
    """
    view_names = _filter_views(sample.get("meta", {}).get("view_names", []))
    dataset = sample.get("dataset", "")
    sid = sample["sample_id"]
    image_parts = []
    total_frames = 0
    views_used = 0

    for i, vn in enumerate(view_names):
        video_path = os.path.join(video_dir, dataset, sid, f"{vn}.mp4")
        if not os.path.exists(video_path):
            continue
        views_used += 1
        parts = _decode_video_to_parts(video_path, fps=fps, max_frames=512)
        if not parts:
            continue

        label = classify_view(vn, i)
        image_parts.append({"type": "text", "text": f"[View: {label}]"})

        image_parts.extend(parts)
        total_frames += len(parts)

    return image_parts, views_used, total_frames


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------

def process_one_sample(
    sample: Dict,
    frame_views: Dict,
    model: str,
    base_url: str = None,
    no_instruction: bool = False,
    input_type: str = "video",
    fps: float = 4.0,
    video_dir: str = None,
) -> Dict:
    """Process a single sample: build image_parts or video_urls, call API, parse response."""
    sid = sample["sample_id"]
    dataset = sample.get("dataset", "")
    view_names_raw = sample.get("meta", {}).get("view_names", [])
    view_names = _filter_views(view_names_raw)
    instruction_raw = sample.get("instruction_raw", "")

    video_urls = None
    image_parts = []
    num_views = 0
    num_frames = 0

    if input_type == "video":
        # Extract video URLs matching filtered view_names (view1, view2, ... in order)
        views_dict = sample.get("views", {})
        sample_video_urls = []
        for i in range(len(view_names)):
            key = f"view{i+1}"
            url = views_dict.get(key, "")
            if isinstance(url, str) and url.startswith("http"):
                sample_video_urls.append(url)
        if sample_video_urls:
            video_urls = sample_video_urls
            num_views = len(video_urls)
        else:
            logger.warning(f"  {sid}: no video URLs, falling back to image mode")
            input_type = "image"

    if input_type == "image":
        if frame_views:
            # Use pre-uploaded frame URLs from frame_index
            image_parts, num_views, num_frames = build_image_parts(view_names, frame_views)
        elif video_dir:
            # Decode from local video files
            image_parts, num_views, num_frames = build_image_parts_from_video(
                sample, video_dir, fps=fps)

    if not image_parts and not video_urls:
        return {
            "sample_id": sid,
            "dataset": dataset,
            "model": model,
            "call_success": False,
            "error": "No frames or video URLs available",
            "timestamp": datetime.now().isoformat(),
        }

    system_prompt, user_prompt = build_prompt(view_names, instruction_raw, no_instruction=no_instruction)
    client = _get_client(base_url)

    start_t = time.time()
    response_text, token_usage = call_api(
        client, image_parts, system_prompt, user_prompt, model,
        video_urls=video_urls, fps=fps,
    )
    elapsed = time.time() - start_t

    parsed = extract_json_from_response(response_text) if response_text else None
    caption_result = _json_to_step_list(parsed)
    success = bool(response_text and caption_result)

    result = {
        "sample_id": sid,
        "dataset": dataset,
        "model": model,
        "instruction_raw": instruction_raw if not isinstance(instruction_raw, list) else "; ".join(instruction_raw),
        "caption_result": caption_result,
        "call_success": success,
        "num_views": num_views,
        "num_frames": num_frames,
        "token_usage": token_usage,
        "elapsed_sec": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Single-stage video annotation")
    parser.add_argument("--model", required=True, help="Model name (e.g. qwen3.5-plus)")
    parser.add_argument("--evalsets", default=DEFAULT_EVALSETS, help="EvalSets.json path")
    parser.add_argument("--frame-index", default=None, help="frame_index.jsonl path (optional, for image mode)")
    parser.add_argument("--input-type", choices=["video", "image"], default="video",
                        help="Input mode: video (send video URL) or image (decode frames)")
    parser.add_argument("--fps", type=float, default=4.0, help="Frames per second (default: 4.0)")
    parser.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR,
                        help="Local video directory (for image mode fallback)")
    parser.add_argument("--output", default=None, help="Output JSONL path (auto-generated if omitted)")
    parser.add_argument("--output-dir", default=None, help="Output directory (overrides default CaptionResult dir)")
    parser.add_argument("--base-url", default=None, help="API base URL")
    parser.add_argument("--num-workers", type=int, default=16, help="Parallel workers")
    parser.add_argument("--start", type=int, default=None, help="Start sample index")
    parser.add_argument("--end", type=int, default=None, help="End sample index (exclusive)")
    parser.add_argument("--no-instruction", action="store_true", help="Do not include instruction_raw in prompt (hard mode)")
    args = parser.parse_args()

    model_tag = args.model.replace("/", "_").replace(".", "_")
    if not args.output:
        out_dir = args.output_dir or DEFAULT_OUTPUT_DIR
        os.makedirs(out_dir, exist_ok=True)
        args.output = os.path.join(out_dir, f"{model_tag}_CaptionResult.jsonl")

    logger.info(f"Model:        {args.model}")
    logger.info(f"Input type:   {args.input_type}")
    logger.info(f"FPS:          {args.fps}")
    logger.info(f"Output:       {args.output}")
    logger.info(f"Workers:      {args.num_workers}")

    # Load data
    logger.info("Loading EvalSets.json ...")
    samples = load_evalsets(args.evalsets)
    logger.info(f"  {len(samples)} samples loaded")

    if args.start is not None or args.end is not None:
        s, e = args.start or 0, args.end or len(samples)
        samples = samples[s:e]
        logger.info(f"  Sliced to [{s}:{e}] -> {len(samples)} samples")

    # Load frame_index (optional)
    frame_index = {}
    fi_path = args.frame_index or DEFAULT_FRAME_INDEX
    if os.path.exists(fi_path):
        logger.info(f"Loading frame_index: {fi_path}")
        frame_index = load_frame_index(fi_path)
        logger.info(f"  {len(frame_index)} entries")
    elif args.input_type == "image":
        logger.info(f"No frame_index found, will decode from local videos ({args.video_dir})")

    # Resume check
    completed = load_completed(args.output)
    if completed:
        before = len(samples)
        samples = [s for s in samples if s["sample_id"] not in completed]
        logger.info(f"  Resume: {before - len(samples)} already done, {len(samples)} remaining")

    if not samples:
        logger.info("All samples already completed!")
        return

    # Init client
    _get_client(args.base_url)

    # Dataset stats
    from collections import Counter
    ds_counter = Counter(s["dataset"] for s in samples)
    view_counter = Counter()
    for s in samples:
        vns = s.get("meta", {}).get("view_names", [])
        view_counter[len(vns)] += 1
    logger.info(f"  Datasets: {dict(ds_counter)}")
    logger.info(f"  View counts: {dict(view_counter)}")

    # Process
    write_lock = threading.Lock()
    success_count = 0
    fail_count = 0

    with open(args.output, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
            futures = {}
            for s in samples:
                sid = s["sample_id"]
                fv = frame_index.get(sid, {})
                fut = pool.submit(
                    process_one_sample, s, fv, args.model,
                    args.base_url, args.no_instruction,
                    input_type=args.input_type, fps=args.fps,
                    video_dir=args.video_dir,
                )
                futures[fut] = sid

            pbar = tqdm(total=len(futures), desc=f"Annotating ({args.model})")
            for fut in as_completed(futures):
                sid = futures[fut]
                try:
                    result = fut.result()
                    with write_lock:
                        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        out_f.flush()
                    if result.get("call_success"):
                        success_count += 1
                    else:
                        fail_count += 1
                        logger.warning(f"FAIL {sid}: {result.get('error', 'empty response')}")
                except Exception as e:
                    fail_count += 1
                    logger.error(f"EXCEPTION {sid}: {e}")
                    err_result = {
                        "sample_id": sid,
                        "model": args.model,
                        "call_success": False,
                        "error": str(e)[:500],
                        "timestamp": datetime.now().isoformat(),
                    }
                    with write_lock:
                        out_f.write(json.dumps(err_result, ensure_ascii=False) + "\n")
                        out_f.flush()
                pbar.update(1)
            pbar.close()

    logger.info(f"\nDone! Success: {success_count}, Failed: {fail_count}")
    logger.info(f"Output: {args.output}")


if __name__ == "__main__":
    main()
