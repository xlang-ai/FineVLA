#!/usr/bin/env python3
"""Pre-extract video frames at a fixed FPS for offline evaluation.

Usage:
    python prepare_frames.py \\
        --evalsets EvalData/EvalSets.json \\
        --video-dir EvalData/Videos \\
        --output-dir EvalData/frames \\
        --fps 2.0
"""

import argparse
import json
import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm

try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False


def decode_video_frames(video_path: str, fps: float) -> list:
    if not HAS_AV:
        raise ImportError("PyAV is required: pip install av")
    container = av.open(video_path)
    stream = container.streams.video[0]
    total_frames = stream.frames or 0
    duration = float(stream.duration * stream.time_base) if stream.duration else 0
    if duration <= 0 and total_frames > 0 and stream.average_rate:
        duration = total_frames / float(stream.average_rate)

    sample_interval = 1.0 / fps if fps > 0 else 0
    frames = []
    next_sample_time = 0.0
    for frame in container.decode(video=0):
        t = float(frame.pts * stream.time_base) if frame.pts is not None else 0
        if t >= next_sample_time:
            frames.append(frame.to_image())
            next_sample_time = t + sample_interval
    container.close()
    return frames


def main():
    parser = argparse.ArgumentParser(description="Pre-extract video frames for RoboFine-Bench")
    parser.add_argument("--evalsets", required=True, help="Path to EvalSets.json")
    parser.add_argument("--video-dir", required=True, help="Root directory containing video files")
    parser.add_argument("--output-dir", default="EvalData/frames", help="Output directory for extracted frames")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second to sample (default: 2.0)")
    parser.add_argument("--quality", type=int, default=85, help="JPEG quality (default: 85)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    with open(args.evalsets) as f:
        evalsets = json.load(f)

    samples = evalsets[args.start:args.end]
    output_dir = Path(args.output_dir)
    skipped = 0

    for sample in tqdm(samples, desc="Extracting frames"):
        sample_id = sample["sample_id"]
        dataset = sample["dataset"]
        views = sample.get("meta", {}).get("view_names", [])
        videos = sample.get("views", {})

        sample_dir = output_dir / sample_id
        if sample_dir.exists() and any(sample_dir.iterdir()):
            skipped += 1
            continue

        sample_dir.mkdir(parents=True, exist_ok=True)

        for view_name in views:
            view_info = videos.get(view_name, {})
            video_rel = view_info.get("video_path", "")
            if not video_rel:
                continue

            video_path = os.path.join(args.video_dir, dataset, sample_id, f"{view_name}.mp4")
            if not os.path.exists(video_path):
                alt_path = os.path.join(args.video_dir, video_rel)
                if os.path.exists(alt_path):
                    video_path = alt_path
                else:
                    continue

            try:
                frames = decode_video_frames(video_path, args.fps)
                for i, frame in enumerate(frames):
                    frame_path = sample_dir / f"{view_name}_{i:04d}.jpg"
                    frame.save(frame_path, "JPEG", quality=args.quality)
            except Exception as e:
                tqdm.write(f"  [WARN] {sample_id}/{view_name}: {e}")

        meta = {"sample_id": sample_id, "dataset": dataset, "fps": args.fps, "views": {}}
        for view_name in views:
            view_frames = sorted(sample_dir.glob(f"{view_name}_*.jpg"))
            meta["views"][view_name] = len(view_frames)
        with open(sample_dir / "meta.json", "w") as f:
            json.dump(meta, f)

    print(f"\nDone. Extracted to {output_dir}/")
    print(f"  Processed: {len(samples) - skipped}, Skipped (already exist): {skipped}")


if __name__ == "__main__":
    main()
