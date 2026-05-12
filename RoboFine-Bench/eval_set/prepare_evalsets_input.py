#!/usr/bin/env python3
"""
Prepare evalsets_input.jsonl and caption_frame_index.jsonl from EvalSets.json.

Reads instruction_raw from Clustering JSONs, updates EvalSets.json,
and generates the two JSONL files needed by the Caption Pipeline.

Usage:
    python Eval_Set/prepare_evalsets_input.py
"""

import json
import glob
import os
import re
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
EVAL_SET_DIR = SCRIPT_DIR

CLUSTERING_DIR = "/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Annotation/Joint2Action/Clustering"
LEROBOT_DIR = "/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21"

DISPLAY_TO_INTERNAL = {
    "BC-Z": "bc_z",
    "RT-1": "rt_1",
    "BridgeDataV2": "bridge",
    "DROID-Robointer": "droid_robointer",
    "RH20T-RoboInter": "rh20t_robointer",
    "RDT": "rdt",
    "Galaxea": "galaxea_open_world",
    "RoboCoin": "robocoin",
    "RoboMINDV1": "robomind_v1.0",
    "RoboMINDV2": "robomind_v2.0",
}

DISPLAY_TO_DATASET_DIR = {
    "BC-Z": "BC_Z",
    "RT-1": "RT-1",
    "BridgeDataV2": "Bridge",
    "DROID-Robointer": "droid_1.0.1",
    "RH20T-RoboInter": "RH20T-RoboInter",
    "RDT": "RDT-yhq",
    "Galaxea": "Galaxea-Open-World-Dataset",
    "RoboCoin": "RoboCOIN",
    "RoboMINDV1": "RoboMindV1.0",
    "RoboMINDV2": "RoboMindV2.0",
}

VIEW_PRIORITY = ["front", "high", "head", "main", "exterior", "top", "image", "primary"]


def extract_trailing_number(sid):
    m = re.search(r"-(\d+)$", sid)
    return m.group(1) if m else None


def extract_prefix(sid):
    return sid.split("-")[0]


def derive_instruction_from_sid(sid, dataset):
    """Last-resort: derive a rough instruction from the sample_id."""
    if dataset == "Galaxea":
        m = re.match(r"galaxea-(.+?)_\d{8}", sid)
        if m:
            return m.group(1).replace("_", " ").lower()
    if dataset == "RoboMINDV1":
        m = re.search(r"robomindv1-\d+_(.*?)(?:-\d+|__)", sid)
        if m:
            raw = m.group(1)
            spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", raw)
            return spaced.replace("_", " ").lower()
    return ""


def load_clustering_index():
    """Load all Clustering JSONs into exact and numeric indices."""
    exact_index = {}
    numeric_index = {}

    pattern = os.path.join(CLUSTERING_DIR, "cluster_representation_summary_*.json")
    files = sorted(glob.glob(pattern))

    for fpath in files:
        if "droid_1.0.1" in fpath:
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for d in data:
            sid = d["sample_id"]
            exact_index[sid] = d
            prefix = extract_prefix(sid)
            num = extract_trailing_number(sid)
            if num:
                key = (prefix, num)
                if key not in numeric_index:
                    numeric_index[key] = d

    print(f"  Clustering index: {len(exact_index)} exact, {len(numeric_index)} numeric entries")
    return exact_index, numeric_index


def load_lerobot_episodes(dataset_display):
    """Load episode instructions from Lerobot meta/episodes.jsonl."""
    dataset_dir = DISPLAY_TO_DATASET_DIR.get(dataset_display, "")
    ep_path = os.path.join(LEROBOT_DIR, dataset_dir, "meta", "episodes.jsonl")
    if not os.path.exists(ep_path):
        return {}
    episodes = {}
    with open(ep_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            tasks = d.get("tasks", "")
            if isinstance(tasks, list):
                tasks = "; ".join(tasks) if tasks else ""
            episodes[d["episode_index"]] = tasks
    return episodes


def build_instruction_raw(cluster_entry):
    """
    For has_raw_steps=1: return list of step desc strings.
    For has_raw_steps=0: return the instruction_raw string.
    """
    if cluster_entry.get("has_raw_steps") and cluster_entry.get("steps_raw"):
        descs = []
        for step in cluster_entry["steps_raw"]:
            desc = step.get("desc", "")
            if desc:
                descs.append(desc)
        if descs:
            return descs
    return cluster_entry.get("instruction_raw", "")


def pick_primary_view(view_names):
    """Pick the best primary view from a list of view names."""
    for kw in VIEW_PRIORITY:
        for v in view_names:
            if kw in v.lower():
                return v
    return view_names[0] if view_names else ""


def main():
    evalsets_path = os.path.join(EVAL_SET_DIR, "EvalSets.json")
    frame_index_path = os.path.join(EVAL_SET_DIR, "frame_index.jsonl")
    output_jsonl = os.path.join(EVAL_SET_DIR, "evalsets_input.jsonl")
    output_caption_fi = os.path.join(EVAL_SET_DIR, "caption_frame_index.jsonl")

    # ── Load EvalSets ──
    print("Loading EvalSets.json ...")
    with open(evalsets_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    print(f"  {len(eval_data)} samples")

    # ── Load Clustering index ──
    print("Loading Clustering JSONs ...")
    exact_idx, numeric_idx = load_clustering_index()

    # ── Match instruction_raw ──
    print("Matching instruction_raw ...")
    lerobot_cache = {}
    match_stats = Counter()
    instruction_map = {}
    cluster_data_map = {}

    for s in eval_data:
        sid = s["sample_id"]
        ds = s["dataset"]

        if sid in exact_idx:
            cluster_data_map[sid] = exact_idx[sid]
            instruction_map[sid] = build_instruction_raw(exact_idx[sid])
            match_stats["exact"] += 1
            continue

        prefix = extract_prefix(sid)
        num = extract_trailing_number(sid)
        if num and (prefix, num) in numeric_idx:
            cluster_data_map[sid] = numeric_idx[(prefix, num)]
            instruction_map[sid] = build_instruction_raw(numeric_idx[(prefix, num)])
            match_stats["numeric"] += 1
            continue

        if ds not in lerobot_cache:
            lerobot_cache[ds] = load_lerobot_episodes(ds)
        ep_map = lerobot_cache[ds]
        if num and int(num) in ep_map:
            instruction_map[sid] = ep_map[int(num)]
            match_stats["lerobot"] += 1
            continue

        derived = derive_instruction_from_sid(sid, ds)
        if derived:
            instruction_map[sid] = derived
            match_stats["derived"] += 1
        else:
            instruction_map[sid] = ""
            match_stats["missing"] += 1

    print(f"  Match results: {dict(match_stats)}")
    print(f"  Total with instruction: {sum(1 for v in instruction_map.values() if v)}")

    # ── Update EvalSets.json with instruction_raw ──
    # Only fill in missing values; don't overwrite existing instruction_raw
    # (which may have been corrected by fix_instruction_raw.py from parquet/episodes)
    print("Updating EvalSets.json with instruction_raw ...")
    es_updated = 0
    for s in eval_data:
        sid = s["sample_id"]
        existing = s.get("instruction_raw")
        if existing and existing != "":
            continue
        if sid in instruction_map and instruction_map[sid]:
            s["instruction_raw"] = instruction_map[sid]
            es_updated += 1

    with open(evalsets_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
    print(f"  Updated {es_updated} samples in EvalSets.json")

    # ── Generate evalsets_input.jsonl ──
    # Read instruction_raw/has_raw_steps/steps_raw from EvalSets.json directly
    # (already corrected by fix_instruction_raw.py for RT-1, Galaxea, etc.)
    print("Generating evalsets_input.jsonl ...")
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for s in eval_data:
            sid = s["sample_id"]
            ds_display = s["dataset"]
            ds_internal = DISPLAY_TO_INTERNAL.get(ds_display, ds_display.lower())
            ds_dir = ds_display
            view_names = s.get("meta", {}).get("view_names", [])
            robot_type = s.get("robot_type", "")

            instr = s.get("instruction_raw", instruction_map.get(sid, ""))
            if isinstance(instr, list):
                instr_raw = "; ".join(instr)
            else:
                instr_raw = instr

            videos = {}
            for vn in view_names:
                videos[vn] = f"{sid}/{vn}.mp4"

            has_raw_steps = s.get("has_raw_steps", 0) or 0
            steps_raw = s.get("steps_raw")
            if not steps_raw:
                cluster = cluster_data_map.get(sid, {})
                if cluster:
                    has_raw_steps = cluster.get("has_raw_steps", 0) or 0
                    steps_raw = cluster.get("steps_raw")

            record = {
                "sample_id": sid,
                "dataset": ds_internal,
                "dataset_dir": ds_dir,
                "views": view_names,
                "videos": videos,
                "instruction_raw": instr_raw,
                "robot_type": robot_type,
                "has_raw_steps": has_raw_steps,
                "steps_raw": steps_raw,
                "duration_sec": s.get("meta", {}).get("duration_sec", 0),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(eval_data)} records to {output_jsonl}")

    # ── Load frame_index.jsonl ──
    print("Loading frame_index.jsonl ...")
    frame_index = {}
    with open(frame_index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            frame_index[d["sample_id"]] = d["views"]
    print(f"  {len(frame_index)} samples")

    # ── Generate caption_frame_index.jsonl ──
    print("Generating caption_frame_index.jsonl ...")
    stage_names = ["analysis", "refinement"]
    missing_fi = 0
    with open(output_caption_fi, "w", encoding="utf-8") as f:
        for s in eval_data:
            sid = s["sample_id"]
            ds_display = s["dataset"]
            view_names = s.get("meta", {}).get("view_names", [])
            fi_views = frame_index.get(sid, {})

            if not fi_views:
                missing_fi += 1
                continue

            primary = pick_primary_view(view_names)
            primary_urls = []
            if primary in fi_views:
                primary_urls = fi_views[primary].get("urls", [])
            elif fi_views:
                first_view = next(iter(fi_views))
                primary_urls = fi_views[first_view].get("urls", [])

            stages = {}
            for stage in stage_names:
                stages[stage] = {"urls": primary_urls}

            if ds_display == "RDT" and len(view_names) > 1:
                wrist_views = [v for v in view_names if "wrist" in v.lower()]
                if wrist_views:
                    wrist_name = wrist_views[0]
                    if wrist_name in fi_views:
                        stages["detail_refinement"] = {
                            "urls": fi_views[wrist_name].get("urls", [])
                        }

            record = {"sample_id": sid, "stages": stages}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    written = len(eval_data) - missing_fi
    print(f"  Wrote {written} records to {output_caption_fi}")
    if missing_fi:
        print(f"  WARNING: {missing_fi} samples missing from frame_index.jsonl")

    print("\nDone!")


if __name__ == "__main__":
    main()
