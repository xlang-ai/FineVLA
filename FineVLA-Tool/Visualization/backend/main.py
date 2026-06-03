"""
FastAPI backend for LeRobot v2.1 Dataset Visualizer.

Serves the frontend, parses parquet paths, streams video, and returns
structured JSON for chart rendering.
"""

import os
import sys
import mimetypes
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from data_parser import (
    parse_parquet_path,
    load_info_json,
    discover_video_keys,
    build_video_paths,
    load_tasks_map,
    load_episode_stats,
    auto_discover_fields,
    get_field_dim_names,
    compute_y_range,
    load_parquet_data,
)

FILTER_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "CanonicalizeAndClean"))
if FILTER_DIR not in sys.path:
    sys.path.insert(0, FILTER_DIR)
FILTER_UTILS_DIR = os.path.join(FILTER_DIR, "utils")
if FILTER_UTILS_DIR not in sys.path:
    sys.path.insert(0, FILTER_UTILS_DIR)

import config as filter_config
from convert_unified import load_modality, extract_episode_data, build_apply_to
from utils.UnifyJointAction import (
    UnifiedStateActionTransform,
    UNIFIED_STATE_ACTION_INDICES,
)
from cal_distance import compute_episode_similarity

app = FastAPI(title="LeRobot v2.1 Visualizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.get("/")
async def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


class ParseRequest(BaseModel):
    parquet_path: str


@app.post("/api/parse")
async def api_parse(req: ParseRequest):
    """Parse a parquet path and return dataset metadata."""
    try:
        parsed = parse_parquet_path(req.parquet_path)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    info = load_info_json(parsed["dataset_root"])
    video_keys = discover_video_keys(info)
    video_paths = build_video_paths(
        parsed["dataset_root"], parsed["chunk_number"],
        parsed["episode_number"], video_keys
    )

    return {
        "dataset_root": parsed["dataset_root"],
        "chunk_number": parsed["chunk_number"],
        "episode_number": parsed["episode_number"],
        "dataset_family": parsed["dataset_family"],
        "video_keys": list(video_paths.keys()),
        "fps": info.get("fps", 30),
        "total_episodes": info.get("total_episodes", 0),
    }


@app.get("/api/video")
async def api_video(request: Request, dataset_root: str, chunk: int, episode: int, video_key: str):
    """Stream a video file with Range request support for seeking."""
    video_path = os.path.join(
        dataset_root, "videos",
        f"chunk-{chunk:03d}",
        video_key,
        f"episode_{episode:06d}.mp4"
    )
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")

    file_size = os.path.getsize(video_path)
    content_type = mimetypes.guess_type(video_path)[0] or "video/mp4"

    range_header = request.headers.get("range")
    if range_header:
        range_match = range_header.strip().split("=")[1]
        range_parts = range_match.split("-")
        start = int(range_parts[0])
        end = int(range_parts[1]) if range_parts[1] else file_size - 1
        end = min(end, file_size - 1)
        content_length = end - start + 1

        def iter_file():
            with open(video_path, "rb") as f:
                f.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk_size = min(8192, remaining)
                    data = f.read(chunk_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(
            iter_file(),
            status_code=206,
            media_type=content_type,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
            },
        )
    else:
        return FileResponse(video_path, media_type=content_type)


@app.get("/api/data")
async def api_data(dataset_root: str, chunk: int, episode: int):
    """Return all chart data for a given episode."""
    info = load_info_json(dataset_root)

    parquet_path = os.path.join(
        dataset_root, "data",
        f"chunk-{chunk:03d}",
        f"episode_{episode:06d}.parquet"
    )
    if not os.path.exists(parquet_path):
        raise HTTPException(status_code=404, detail=f"Parquet not found: {parquet_path}")

    parsed = parse_parquet_path(parquet_path)
    config = parsed["config"]

    if config:
        state_fields = config["state_fields"]
        action_fields = config["action_fields"]
    else:
        state_fields, action_fields = auto_discover_fields(info)

    # Filter to fields that actually exist in info.json features
    available_features = set(info.get("features", {}).keys())
    state_fields = [f for f in state_fields if f in available_features]
    action_fields = [f for f in action_fields if f in available_features]

    tasks_map = load_tasks_map(dataset_root)
    ep_stats = load_episode_stats(dataset_root, episode)

    raw = load_parquet_data(parquet_path, state_fields, action_fields)

    state_data = {}
    for field, values in raw["state_data"].items():
        dim_names = get_field_dim_names(info, field)
        y_min, y_max = compute_y_range(ep_stats, field)
        state_data[field] = {
            "dim_names": dim_names,
            "values": values,
            "y_min": y_min,
            "y_max": y_max,
        }

    action_data = {}
    for field, values in raw["action_data"].items():
        dim_names = get_field_dim_names(info, field)
        y_min, y_max = compute_y_range(ep_stats, field)
        action_data[field] = {
            "dim_names": dim_names,
            "values": values,
            "y_min": y_min,
            "y_max": y_max,
        }

    tasks_map_str = {str(k): v for k, v in tasks_map.items()}

    return {
        "total_frames": raw["total_frames"],
        "fps": info.get("fps", 30),
        "tasks_map": tasks_map_str,
        "frame_tasks": raw["frame_tasks"],
        "state_data": state_data,
        "action_data": action_data,
    }


DATA_ROOT = os.environ.get("VLA_DATA_ROOT", "/path/to/your/Lerobot_v21")


@app.get("/api/list_subs")
async def api_list_subs(family: str):
    """List sub-datasets for a given family. Scans the filesystem."""
    fam_path = os.path.join(DATA_ROOT, family)
    if not os.path.isdir(fam_path):
        raise HTTPException(status_code=404, detail=f"Family not found: {family}")

    root_info = os.path.join(fam_path, "meta", "info.json")
    if os.path.exists(root_info):
        return {"subs": [""]}

    subs = []
    for item in sorted(os.listdir(fam_path)):
        item_path = os.path.join(fam_path, item)
        if not os.path.isdir(item_path):
            continue
        if os.path.exists(os.path.join(item_path, "meta", "info.json")):
            subs.append(item)
        else:
            for sub2 in sorted(os.listdir(item_path)):
                sub2_path = os.path.join(item_path, sub2)
                if not os.path.isdir(sub2_path):
                    continue
                if os.path.exists(os.path.join(sub2_path, "meta", "info.json")):
                    subs.append(f"{item}/{sub2}")
                else:
                    # 3rd level (e.g. RoboMindV1.0: benchmark/robot_type/task)
                    for sub3 in sorted(os.listdir(sub2_path)):
                        sub3_path = os.path.join(sub2_path, sub3)
                        if os.path.isdir(sub3_path) and os.path.exists(os.path.join(sub3_path, "meta", "info.json")):
                            subs.append(f"{item}/{sub2}/{sub3}")
    return {"subs": subs}


GRIPPER_SLOTS = {"left_gripper", "right_gripper"}

TARGET_JOINT_STATE_TYPE = "abs_joint"
TARGET_EEF_STATE_TYPE = "abs_quat"
TARGET_JOINT_ACTION_TYPE = "abs_joint"
TARGET_EEF_ACTION_TYPE = "abs_quat"


def _resolve_dataset_key(dataset_root: str) -> str:
    """从 dataset_root 推断在 filter_config.STATE_ACTION_COMPARE_SLOTS 中的 key。"""
    norm = os.path.normpath(dataset_root)
    data_root = os.path.normpath(filter_config.DATA_ROOT)
    if norm.startswith(data_root):
        rel = os.path.relpath(norm, data_root)
    else:
        rel = os.path.basename(norm)
    if rel in filter_config.STATE_ACTION_COMPARE_SLOTS:
        return rel
    top = rel.split(os.sep)[0]
    if top in filter_config.STATE_ACTION_COMPARE_SLOTS:
        return top
    return rel


class CalDtwRequest(BaseModel):
    dataset_root: str
    chunk: int
    episode: int


@app.post("/api/cal_dtw")
async def api_cal_dtw(req: CalDtwRequest):
    """对当前 episode 做 unified 转换后计算 state 与 action 的 DTW 距离。"""
    dataset_root = req.dataset_root
    parquet_path = os.path.join(
        dataset_root, "data",
        f"chunk-{req.chunk:03d}",
        f"episode_{req.episode:06d}.parquet"
    )
    if not os.path.exists(parquet_path):
        raise HTTPException(status_code=404, detail=f"Parquet not found: {parquet_path}")

    dataset_key = _resolve_dataset_key(dataset_root)
    slot_cfg = filter_config.STATE_ACTION_COMPARE_SLOTS.get(dataset_key)
    if slot_cfg is None:
        raise HTTPException(
            status_code=400,
            detail=f"数据集 '{dataset_key}' 未配置 state-action 对比 slot（STATE_ACTION_COMPARE_SLOTS 中为 None 或不存在）。"
        )
    state_slot = slot_cfg["state_slot"]
    action_slot = slot_cfg["action_slot"]
    for s in (state_slot, action_slot):
        if s in GRIPPER_SLOTS:
            raise HTTPException(status_code=400, detail=f"slot '{s}' 为 gripper 类型，不支持 DTW 对比。")

    modality_path = os.path.join(dataset_root, "meta", "modality.json")
    if not os.path.exists(modality_path):
        raise HTTPException(status_code=400, detail=f"modality.json not found: {modality_path}")

    modality_cfg = load_modality(dataset_root)
    apply_to = build_apply_to(modality_cfg)
    transform = UnifiedStateActionTransform(
        apply_to=apply_to,
        modality_path=modality_path,
        target_joint_state_type=TARGET_JOINT_STATE_TYPE,
        target_eef_state_type=TARGET_EEF_STATE_TYPE,
        target_joint_action_type=TARGET_JOINT_ACTION_TYPE,
        target_eef_action_type=TARGET_EEF_ACTION_TYPE,
    )

    data, frame_index = extract_episode_data(parquet_path, modality_cfg)
    result = transform.apply(data)

    state_unified = result["state.unified"]   # (T, 80)
    action_unified = result["action.unified"] # (T, 80)
    mask_state = result["mask.state"]         # (T, 80)
    mask_action = result["mask.action"]       # (T, 80)

    if state_slot not in UNIFIED_STATE_ACTION_INDICES:
        raise HTTPException(status_code=400, detail=f"state_slot '{state_slot}' 不在 UNIFIED_STATE_ACTION_INDICES 中。")
    if action_slot not in UNIFIED_STATE_ACTION_INDICES:
        raise HTTPException(status_code=400, detail=f"action_slot '{action_slot}' 不在 UNIFIED_STATE_ACTION_INDICES 中。")

    s_lo, s_hi = UNIFIED_STATE_ACTION_INDICES[state_slot]
    a_lo, a_hi = UNIFIED_STATE_ACTION_INDICES[action_slot]

    state_slice = state_unified[:, s_lo:s_hi]
    action_slice = action_unified[:, a_lo:a_hi]
    mask_s_slice = mask_state[:, s_lo:s_hi]
    mask_a_slice = mask_action[:, a_lo:a_hi]

    valid_dim_mask = (mask_s_slice & mask_a_slice).all(axis=0)
    if not valid_dim_mask.any():
        raise HTTPException(status_code=400, detail="state 与 action 没有共同有效维度（mask 不满足）。")

    state_valid = state_slice[:, valid_dim_mask].astype(np.float64)
    action_valid = action_slice[:, valid_dim_mask].astype(np.float64)

    if state_valid.shape[0] < 2:
        raise HTTPException(status_code=400, detail="有效帧数不足（< 2），无法计算 DTW。")

    sim = compute_episode_similarity(state_valid, action_valid)

    T, D = sim["state_norm"].shape
    dim_names = [f"dim_{i}" for i in range(D)]

    return {
        "dtw_score": sim["dtw_score"],
        "state_slot": state_slot,
        "action_slot": action_slot,
        "valid_dims": D,
        "total_frames": T,
        "dim_names": dim_names,
        "state_norm": sim["state_norm"].tolist(),
        "action_norm": sim["action_norm"].tolist(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
