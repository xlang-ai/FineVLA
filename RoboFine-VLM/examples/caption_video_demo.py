#!/usr/bin/env python3
"""Run RoboFine-VLM caption annotation on local demo videos.

This script uses the fixed RoboFine-Bench RB-new prompt and dynamic view labels.
It samples each local video at fps=4, sends each view as one ``video`` part, and
prints normalized Step1/Step2/... JSON.
"""

import argparse
import json
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robofine_caption_prompt import build_caption_prompt, classify_view
from video_utils import sample_video_frames

DEFAULT_CONFIG = ROOT / "config" / "caption_annotation_config.json"
DEFAULT_DEMO_DIR = ROOT / "assets" / "demo_three_view"


def is_local_qwen_like(model: str, base_url: str) -> bool:
    """Match RoboFine-Bench's local Qwen/RoboFine vLLM request branch."""
    m = model.lower()
    is_qwen_like = "qwen" in m or "qvq" in m or "robofine" in m
    is_dashscope = "dashscope" in base_url.lower() or "aliyuncs" in base_url.lower()
    return is_qwen_like and not is_dashscope


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_view_arg(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--view must be role=/path/to/video.mp4")
    role, path = value.split("=", 1)
    role = role.strip()
    path = path.strip()
    if not role or not path:
        raise argparse.ArgumentTypeError("--view role and path must be non-empty")
    return role, path


def extract_json_object(text: str):
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if not match:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1) if match.lastindex else match.group(0))
    except json.JSONDecodeError:
        return None


def normalize_steps(text: str):
    parsed = extract_json_object(text)
    if not isinstance(parsed, dict):
        return None

    def sort_key(item):
        match = re.search(r"(\d+)", str(item[0]))
        return int(match.group(1)) if match else 10**9

    steps = [
        str(value).strip()
        for _, value in sorted(parsed.items(), key=sort_key)
        if isinstance(value, str) and value.strip()
    ]
    if not steps:
        return None
    return {f"Step{i}": step for i, step in enumerate(steps, 1)}


def default_views() -> OrderedDict:
    return OrderedDict([
        ("main", str(DEFAULT_DEMO_DIR / "main.mp4")),
        ("left_wrist", str(DEFAULT_DEMO_DIR / "left_wrist.mp4")),
        ("right_wrist", str(DEFAULT_DEMO_DIR / "right_wrist.mp4")),
    ])


def main():
    cfg = load_config(DEFAULT_CONFIG)
    bench = cfg["benchmark_setting"]
    api_defaults = cfg["api_defaults"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=api_defaults["base_url"])
    parser.add_argument("--model", default=api_defaults["model"])
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", api_defaults["api_key"]))
    parser.add_argument("--mode", choices=bench["supported_modes"], default=bench["default_mode"])
    parser.add_argument("--instruction", default="", help="Task instruction, used only in easy mode")
    parser.add_argument("--view", action="append", type=parse_view_arg,
                        help="role=/path/to/video.mp4; repeat for multi-view input")
    parser.add_argument("--max-frames-per-view", type=int, default=int(bench.get("max_frames_per_view", 512)))
    parser.add_argument("--resize-width", type=int, default=bench.get("resize_width", 512))
    parser.add_argument("--print-request", action="store_true")
    parser.add_argument("--print-raw", action="store_true")
    args = parser.parse_args()

    views = OrderedDict()
    if args.view:
        for role, path in args.view:
            views[role] = path
    else:
        views = default_views()

    fps = float(bench["fps"])
    prompt = build_caption_prompt(args.mode, len(views), args.instruction)

    content = []
    metas = []
    for i, (role, path) in enumerate(views.items()):
        frame_urls, meta = sample_video_frames(
            path,
            fps=fps,
            max_frames=args.max_frames_per_view,
            resize_width=args.resize_width,
        )
        metas.append({"role": role, **meta})
        content.append({"type": "text", "text": f"[View: {classify_view(role, i)}]"})
        content.append({"type": "video", "video": frame_urls, "fps": fps})
    content.append({"type": "text", "text": prompt})

    if args.print_request:
        print(json.dumps({
            "model": args.model,
            "base_url": args.base_url,
            "mode": args.mode,
            "fps": fps,
            "request_format": "RoboFine-Bench image/base64 path: one video part per view, each video is a list of base64 image URLs",
            "temperature": bench["temperature"],
            "top_p": bench["top_p"],
            "max_tokens": bench["max_tokens"],
            "pixel_overrides": bench["pixel_overrides"],
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
            if is_local_qwen_like(args.model, args.base_url) else None,
            "views": metas,
        }, ensure_ascii=False, indent=2), flush=True)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    kwargs = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": float(bench["temperature"]),
        "top_p": float(bench["top_p"]),
        "max_tokens": int(bench["max_tokens"]),
    }
    if is_local_qwen_like(args.model, args.base_url):
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    response = client.chat.completions.create(**kwargs)
    text = response.choices[0].message.content
    normalized = normalize_steps(text)
    if args.print_raw or normalized is None:
        print(text)
    if normalized is not None:
        print(json.dumps(normalized, ensure_ascii=False, indent=2))
    if getattr(response, "usage", None):
        print(json.dumps(response.usage.model_dump(), indent=2), flush=True)


if __name__ == "__main__":
    main()
