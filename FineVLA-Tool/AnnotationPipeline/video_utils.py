import io
import os
import sys
import base64
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import config as config_module
from config import (
    DEFAULT_ANALYSIS_FPS,
    DEFAULT_MAX_FRAMES,
    DEFAULT_RESIZE_WIDTH,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_BATCH_SIZE,
    MIN_API_FRAMES,
    logger,
)


# =============================================================================
# Video Processing Utilities
# =============================================================================

class _DecordStub:
    """Stub so that config_module.decord.VideoReader(...) raises a catchable error
    instead of AttributeError-on-None, letting open_video_reader fall back to pyav."""
    def __getattr__(self, name):
        raise RuntimeError("decord not installed, falling back to pyav")


def _ensure_imports():
    """Lazily import heavy dependencies."""
    if config_module.decord is None:
        try:
            import decord as _decord
            config_module.decord = _decord
        except ImportError:
            config_module.decord = _DecordStub()
    if config_module.OpenAI is None:
        from openai import OpenAI as _OpenAI
        config_module.OpenAI = _OpenAI


def encode_frame_to_base64(
    frame_rgb: np.ndarray,
    quality: int = DEFAULT_JPEG_QUALITY
) -> str:
    """Convert RGB numpy array to JPEG base64 string."""
    img = Image.fromarray(frame_rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def compute_sample_indices(
    total_frames: int,
    video_fps: float,
    target_fps: float,
    max_frames: int
) -> List[int]:
    """Compute frame indices to sample at target FPS."""
    if total_frames <= 0:
        return []
    
    if video_fps <= 0 or target_fps <= 0:
        indices = list(range(total_frames))
    else:
        interval = max(1, int(round(video_fps / target_fps)))
        indices = list(range(0, total_frames, interval))
    
    if len(indices) > max_frames:
        indices = np.linspace(0, total_frames - 1, max_frames).astype(int).tolist()
    
    return sorted(set([i for i in indices if 0 <= i < total_frames]))


def compute_sample_indices_by_ratio(
    total_frames: int,
    ratio: float
) -> List[int]:
    """
    Compute frame indices to sample by ratio of total frames.
    
    Args:
        total_frames: Total number of frames in the video
        ratio: Float in (0, 1], where 1.0 means all frames
    
    Returns:
        List of frame indices, monotonically increasing and in-range
    """
    if total_frames <= 0:
        return []
    
    # Clamp ratio to valid range
    ratio = max(0.0, min(1.0, ratio))
    
    if ratio <= 0:
        return []
    
    if ratio >= 1.0:
        # Return all frames
        return list(range(total_frames))
    
    # Compute number of frames to sample
    n = max(1, round(total_frames * ratio))
    
    if n >= total_frames:
        return list(range(total_frames))
    
    # Use linspace to get evenly distributed indices
    indices = np.linspace(0, total_frames - 1, n).astype(int).tolist()
    
    # Ensure unique, sorted, and in-range
    return sorted(set([i for i in indices if 0 <= i < total_frames]))



def _read_video_pyav(video_path, indices):
    """Fallback: read frames using PyAV (supports AV1 and all ffmpeg codecs)."""
    import av

    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        total_frames = stream.frames or 0
        video_fps = float(stream.average_rate) if stream.average_rate else 30.0

        # If container doesn't report frame count, do a quick count
        if total_frames <= 0:
            for _ in container.decode(video=0):
                total_frames += 1
            container.seek(0)

        indices_set = set(indices)
        max_idx = max(indices) if indices else 0
        collected = {}

        for frame_idx, frame in enumerate(container.decode(video=0)):
            if frame_idx > max_idx:
                break
            if frame_idx in indices_set:
                collected[frame_idx] = frame.to_ndarray(format="rgb24")

        ordered = [collected[i] for i in indices if i in collected]
        return ordered, total_frames, video_fps
    finally:
        container.close()


def _get_video_info_pyav(video_path):
    """Get total_frames and fps via PyAV."""
    import av

    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        total_frames = stream.frames or 0
        video_fps = float(stream.average_rate) if stream.average_rate else 30.0

        if total_frames <= 0:
            for _ in container.decode(video=0):
                total_frames += 1

        return total_frames, video_fps
    finally:
        container.close()



def _encode_frame_jpeg_bytes(
    frame: np.ndarray,
    resize_width: int = DEFAULT_RESIZE_WIDTH,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> bytes:
    """Resize (if needed) and encode a single frame to JPEG bytes.

    Uses OpenCV when available (~6x faster than PIL for resize+encode).
    """
    try:
        import cv2
        h, w = frame.shape[:2]
        if resize_width and w > resize_width:
            new_w = resize_width
            new_h = int(round(h * (new_w / w)))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # cv2 expects BGR
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        return buf.tobytes()
    except ImportError:
        h, w = frame.shape[:2]
        if resize_width and w > resize_width:
            new_w = resize_width
            new_h = int(round(h * (new_w / w)))
            frame = np.array(
                Image.fromarray(frame).resize((new_w, new_h), Image.Resampling.LANCZOS)
            )
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        return buf.getvalue()



def open_video_reader(video_path: str):
    """Open a video and return (vr, total_frames, video_fps, use_pyav).

    ``vr`` is a decord VideoReader when available, else *None* (pyav fallback).
    Caller should ``del vr`` when done to free the file handle.
    """
    _ensure_imports()

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    try:
        vr = config_module.decord.VideoReader(video_path, ctx=config_module.decord.cpu())
        total_frames = len(vr)
        video_fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 30.0
        return vr, total_frames, video_fps, False
    except Exception:
        total_frames, video_fps = _get_video_info_pyav(video_path)
        return None, total_frames, video_fps, True


def read_and_encode_frames(
    video_path: str,
    vr,
    indices: List[int],
    use_pyav: bool,
    batch_size: int = DEFAULT_BATCH_SIZE,
    resize_width: int = DEFAULT_RESIZE_WIDTH,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> Tuple[List[Dict], str]:
    """Read *indices* from an already-opened video, encode, and return image_parts.

    Returns (image_parts, upload_mode) where upload_mode is always 'base64'.
    """
    if not indices:
        return [], "none"

    if use_pyav or vr is None:
        frames_list, _, _ = _read_video_pyav(video_path, indices)
    else:
        try:
            all_frames = []
            for cs in range(0, len(indices), batch_size):
                ce = min(cs + batch_size, len(indices))
                frames = vr.get_batch(indices[cs:ce]).asnumpy()
                all_frames.extend(frames)
                del frames
            frames_list = all_frames
        except Exception:
            frames_list, _, _ = _read_video_pyav(video_path, indices)

    jpeg_list = [_encode_frame_jpeg_bytes(f, resize_width, jpeg_quality) for f in frames_list]
    del frames_list

    image_parts = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(jb).decode('utf-8')}"},
        }
        for jb in jpeg_list
    ]

    del jpeg_list
    return image_parts, "base64"


def load_video_as_image_parts(
    video_path: str,
    target_fps: float = DEFAULT_ANALYSIS_FPS,
    max_frames: int = DEFAULT_MAX_FRAMES,
    resize_width: int = DEFAULT_RESIZE_WIDTH,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    sample_ratio: float = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    frame_start: int = 0,
    frame_end: Optional[int] = None,
    video_reader_cache: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict], Dict]:
    """Load video and convert to image parts for API.

    Delegates to ``open_video_reader`` + ``read_and_encode_frames`` internally.

    If *video_reader_cache* is provided, the opened video reader is stored in
    (and potentially reused from) this dict, keyed by *video_path*.  The caller
    is responsible for cleaning up cached readers when done.
    """
    cached = video_reader_cache.get(video_path) if video_reader_cache is not None else None
    if cached is not None:
        vr, total_frames, video_fps, use_pyav = cached
    else:
        vr, total_frames, video_fps, use_pyav = open_video_reader(video_path)
        if video_reader_cache is not None:
            video_reader_cache[video_path] = (vr, total_frames, video_fps, use_pyav)

    eff_start = max(0, frame_start)
    eff_end = min(total_frames, frame_end + 1) if frame_end is not None else total_frames
    eff_total = max(0, eff_end - eff_start)

    if sample_ratio is not None:
        indices = compute_sample_indices_by_ratio(eff_total, sample_ratio)
        sample_mode = "ratio"
    else:
        indices = compute_sample_indices(eff_total, video_fps, target_fps, max_frames)
        sample_mode = "fps"

    indices = [i + eff_start for i in indices]

    # Enforce API minimum: Qwen requires >= MIN_API_FRAMES frames per request.
    # If sampling produced too few, re-sample with linspace up to MIN_API_FRAMES.
    if 0 < len(indices) < MIN_API_FRAMES and eff_total >= MIN_API_FRAMES:
        indices = sorted(set(
            int(round(x)) for x in np.linspace(eff_start, eff_end - 1, MIN_API_FRAMES)
        ))
        logger.debug(
            f"Frame count below MIN_API_FRAMES={MIN_API_FRAMES}, "
            f"expanded to {len(indices)} frames for {video_path}"
        )

    if not indices:
        if video_reader_cache is None and vr is not None:
            del vr
        raise RuntimeError(f"No frames sampled from {video_path} (range [{eff_start}:{eff_end}))")

    image_parts, upload_mode = read_and_encode_frames(
        video_path, vr, indices, use_pyav,
        batch_size=batch_size,
        resize_width=resize_width,
        jpeg_quality=jpeg_quality,
    )

    # Only release the reader if we're not using a cache
    if video_reader_cache is None and vr is not None:
        del vr

    metadata = {
        "video_fps": video_fps,
        "total_frames": total_frames,
        "sampled_frames": len(indices),
        "sample_mode": sample_mode,
        "sample_ratio": sample_ratio if sample_ratio is not None else None,
        "video_path": video_path,
        "backend": "pyav" if use_pyav else "decord",
        "upload_mode": upload_mode,
        "frame_range": [eff_start, eff_end] if frame_end is not None else None,
    }

    return image_parts, metadata
