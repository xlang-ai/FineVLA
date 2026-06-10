#!/usr/bin/env python3
"""Official Caption benchmark launcher.

This wrapper fixes the benchmark setting to all available views, fps=4, RB new
prompt, and no explicit pixel overrides. Hard mode is the default official
setting; Easy mode is also available for optional reporting. Model-specific
logic belongs in caption_eval/adapters/.
"""

import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from caption_eval.annotate import run_annotate

DEFAULT_EVALSETS = os.path.join(ROOT_DIR, "EvalData", "EvalSets.json")
DEFAULT_FRAME_INDEX = os.path.join(ROOT_DIR, "EvalData", "frame_index.jsonl")
DEFAULT_VIDEO_DIR = os.path.join(ROOT_DIR, "EvalData", "Videos")
DEFAULT_OUTPUT_ROOT = os.path.join(ROOT_DIR, "caption_eval", "result", "caption")


def main():
    parser = argparse.ArgumentParser(description="Run the official RoboFine-Bench Caption benchmark")
    parser.add_argument("--model", required=True, help="Model name passed to the selected adapter")
    parser.add_argument("--adapter", default="openai-compatible",
                        help="Caption adapter name (default: openai-compatible)")
    parser.add_argument("--input-type", choices=["video", "image"], default="video",
                        help="Official input track: video or image")
    parser.add_argument("--mode", choices=["hard", "easy"], default="hard",
                        help="Caption mode: hard omits instruction_raw; easy includes it")
    parser.add_argument("--api-key", default=None,
                        help="API key for the selected adapter (default: OPENAI_API_KEY)")
    parser.add_argument("--base-url", default=None, help="API base URL for API adapters")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--output", default=None, help="Output JSONL path")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--evalsets", default=DEFAULT_EVALSETS)
    parser.add_argument("--frame-index", default=DEFAULT_FRAME_INDEX)
    parser.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR)
    args = parser.parse_args()

    track_dir = f"{args.mode}_{args.input_type}"
    output_dir = args.output_dir or os.path.join(DEFAULT_OUTPUT_ROOT, track_dir)

    forwarded = [
        "run_annotate.py",
        "--model", args.model,
        "--adapter", args.adapter,
        "--evalsets", args.evalsets,
        "--input-type", args.input_type,
        "--fps", "4",
        "--output-dir", output_dir,
        "--num-workers", str(args.num_workers),
        "--video-dir", args.video_dir,
    ]
    if args.mode == "hard":
        forwarded.append("--no-instruction")
    if args.input_type == "image":
        forwarded.extend(["--frame-index", args.frame_index])
    if args.api_key:
        forwarded.extend(["--api-key", args.api_key])
    if args.base_url:
        forwarded.extend(["--base-url", args.base_url])
    if args.output:
        forwarded.extend(["--output", args.output])

    sys.argv = forwarded
    run_annotate.main()


if __name__ == "__main__":
    main()
