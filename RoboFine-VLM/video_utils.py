"""Video preprocessing helpers for RoboFine-VLM caption demos.

The benchmark call samples video frames at a fixed FPS, groups frames by view,
and sends each view as one OpenAI-compatible ``video`` part. This module keeps
that preprocessing available inside the RoboFine-VLM directory for users who
want to test their deployment without depending on RoboFine-Bench internals.
"""

import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple


def encode_image_to_data_url(image, quality: int = 85) -> str:
    """Encode a PIL image as a JPEG data URL."""
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def sample_video_frames(
    video_path: str,
    fps: float = 4.0,
    max_frames: int = 512,
    resize_width: int = None,
    jpeg_quality: int = 85,
) -> Tuple[List[str], Dict]:
    """Sample a video into JPEG data URLs.

    Args:
        video_path: Local mp4 path.
        fps: Target sampling FPS. The benchmark setting is 4.0.
        max_frames: Safety cap per view.
        resize_width: Optional width resize. ``None`` keeps original size.
        jpeg_quality: JPEG encoding quality for data URLs.

    Returns:
        (frame_data_urls, metadata)
    """
    try:
        import av
        import numpy as np
    except ImportError as exc:
        raise ImportError("PyAV and NumPy are required: pip install av numpy") from exc

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    container = av.open(str(path))
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
    index_set = set(indices)

    urls: List[str] = []
    try:
        for frame_idx, frame in enumerate(container.decode(video=0)):
            if frame_idx not in index_set:
                continue
            img = frame.to_image()
            if resize_width and img.width > resize_width:
                new_h = int(round(img.height * resize_width / img.width))
                img = img.resize((resize_width, new_h))
            urls.append(encode_image_to_data_url(img, quality=jpeg_quality))
            if len(urls) >= len(indices):
                break
    finally:
        container.close()

    meta = {
        "video_path": str(path),
        "native_fps": native_fps,
        "target_fps": fps,
        "total_frames": total_frames,
        "sampled_frames": len(urls),
        "max_frames": max_frames,
        "resize_width": resize_width,
        "jpeg_quality": jpeg_quality,
    }
    return urls, meta
