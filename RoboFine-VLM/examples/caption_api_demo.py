#!/usr/bin/env python3
"""Minimal RoboFine-VLM caption API demo.

This script mirrors the RoboFine-Bench caption call:
  - all provided views are preserved in order
  - each view is sent as one video part containing sampled frame URLs/base64 images
  - fps=4 is attached to every video part
  - RB new prompt is imported from robofine_caption_prompt.py
  - temperature=0.0, top_p=0.95, max_tokens=32768
  - no pixel overrides are sent

Example:
    python examples/caption_api_demo.py \
        --base-url http://localhost:8000/v1 \
        --model robofine-vlm \
        --mode hard \
        --view main=/path/to/main_000.jpg \
        --view main=/path/to/main_001.jpg \
        --view left_wrist=/path/to/left_000.jpg \
        --view right_wrist=/path/to/right_000.jpg
"""

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "caption_annotation_config.json"
sys.path.insert(0, str(ROOT))

from robofine_caption_prompt import build_caption_prompt, classify_view


def is_local_qwen_like(model: str, base_url: str) -> bool:
    """Match RoboFine-Bench's local Qwen/RoboFine vLLM request branch."""
    m = model.lower()
    is_qwen_like = "qwen" in m or "qvq" in m or "robofine" in m
    is_dashscope = "dashscope" in base_url.lower() or "aliyuncs" in base_url.lower()
    return is_qwen_like and not is_dashscope


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def image_url(path_or_url: str) -> str:
    if path_or_url.startswith(("http://", "https://", "data:")):
        return path_or_url
    path = Path(path_or_url)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def parse_view_arg(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--view must be role=/path/to/image or role=https://...")
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

    steps = []
    for _, value in sorted(parsed.items(), key=sort_key):
        if isinstance(value, str) and value.strip():
            steps.append(value.strip())
    if not steps:
        return None
    return {f"Step{i}": step for i, step in enumerate(steps, 1)}


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
    parser.add_argument("--view", action="append", required=True, type=parse_view_arg,
                        help="role=image_path_or_url; repeat for frames/views")
    parser.add_argument("--input-type", choices=["image", "video"], default="image",
                        help="image: send frames as individual image_url parts (SGLang/broad compat); "
                             "video: group frames as one video part (vLLM)")
    parser.add_argument("--print-request", action="store_true",
                        help="Print request metadata before calling the model")
    parser.add_argument("--print-raw", action="store_true",
                        help="Also print the raw model response before normalized JSON")
    args = parser.parse_args()

    grouped = OrderedDict()
    for role, path in args.view:
        grouped.setdefault(role, []).append(path)

    prompt = build_caption_prompt(args.mode, len(grouped), args.instruction)
    fps = float(bench["fps"])

    content = []
    for i, (role, paths) in enumerate(grouped.items()):
        content.append({"type": "text", "text": f"[View: {classify_view(role, i)}]"})
        if args.input_type == "video":
            content.append({
                "type": "video",
                "video": [image_url(path) for path in paths],
                "fps": fps,
            })
        else:
            for path in paths:
                content.append({"type": "image_url", "image_url": {"url": image_url(path)}})
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
            "views": {role: len(paths) for role, paths in grouped.items()},
        }, indent=2), flush=True)

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
