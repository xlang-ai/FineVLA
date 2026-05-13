#!/usr/bin/env python3
"""
Create a small test subset of bridge dataset for v20→v21 conversion testing.
Copies first N episodes (default 10) to a test directory.
"""
import json
import shutil
from pathlib import Path

# --- Config ---
SOURCE_ROOT = Path("/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_phu/OXE_LEROBOT_DATASET/bridge_orig_lerobot")
TEST_ROOT = Path("/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21/Bridge_test")
NUM_EPISODES = 10  # 只复制前10个episode，足够测试转换逻辑

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, path):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    print(f"Source: {SOURCE_ROOT}")
    print(f"Test output: {TEST_ROOT}")
    print(f"Copying first {NUM_EPISODES} episodes...")
    
    # Clean and create test dir
    if TEST_ROOT.exists():
        print(f"Removing existing test dir: {TEST_ROOT}")
        shutil.rmtree(TEST_ROOT)
    TEST_ROOT.mkdir(parents=True)
    
    # 1. Load source metadata
    info = json.load(open(SOURCE_ROOT / "meta/info.json"))
    episodes = load_jsonl(SOURCE_ROOT / "meta/episodes.jsonl")
    tasks = load_jsonl(SOURCE_ROOT / "meta/tasks.jsonl")
    
    # 2. Filter episodes (first N)
    test_episodes = episodes[:NUM_EPISODES]
    
    # 3. Collect task indices used by test episodes
    used_tasks = set()
    for ep in test_episodes:
        # v2.0 format: "tasks" is list of strings
        if "tasks" in ep:
            for t in ep["tasks"]:
                used_tasks.add(t)
        if "task_index" in ep:
            used_tasks.add(ep["task_index"])
    
    # 4. Filter tasks (only those referenced, or keep first N for simplicity)
    # For v2.0, tasks.jsonl has {"task_index": i, "task": "..."}, episodes have {"tasks": ["..."]}
    # Keep all tasks whose "task" string is in used_tasks
    test_tasks = [t for t in tasks if t.get("task") in used_tasks]
    if not test_tasks:
        # Fallback: keep first few tasks
        test_tasks = tasks[:max(len(used_tasks), 10)]
    
    # 5. Compute total frames
    total_frames = sum(ep.get("length", 0) for ep in test_episodes)
    
    # 6. Update info.json
    test_info = info.copy()
    test_info["total_episodes"] = len(test_episodes)
    test_info["total_frames"] = total_frames
    test_info["total_tasks"] = len(test_tasks)
    test_info["total_chunks"] = 1  # All test episodes in chunk-000
    test_info["splits"] = {"train": f"0:{len(test_episodes)}"}
    
    # 7. Create meta/
    meta_dir = TEST_ROOT / "meta"
    meta_dir.mkdir()
    with open(meta_dir / "info.json", "w") as f:
        json.dump(test_info, f, indent=2, ensure_ascii=False)
    save_jsonl(test_episodes, meta_dir / "episodes.jsonl")
    save_jsonl(test_tasks, meta_dir / "tasks.jsonl")
    
    # Copy stats.json (needed for v20→v21 check_aggregate_stats)
    if (SOURCE_ROOT / "meta/stats.json").exists():
        shutil.copy(SOURCE_ROOT / "meta/stats.json", meta_dir / "stats.json")
    
    # 8. Copy data/chunk-000 (only selected episodes)
    data_chunk = TEST_ROOT / "data/chunk-000"
    data_chunk.mkdir(parents=True)
    for ep in test_episodes:
        ep_idx = ep["episode_index"]
        src_pq = SOURCE_ROOT / f"data/chunk-000/episode_{ep_idx:06d}.parquet"
        if src_pq.exists():
            shutil.copy(src_pq, data_chunk / f"episode_{ep_idx:06d}.parquet")
            print(f"  Copied: {src_pq.name}")
        else:
            print(f"  Warning: missing {src_pq}")
    
    # 9. Copy videos/chunk-000 (only selected episodes, all video keys)
    # Also track which video features actually exist for fixing info.json
    copied_video_keys = []
    src_video_chunk = SOURCE_ROOT / "videos/chunk-000"
    if src_video_chunk.exists():
        for video_key_dir in src_video_chunk.iterdir():
            if video_key_dir.is_dir():
                # Check if this video key has files for our episodes
                has_files = any(
                    (video_key_dir / f"episode_{ep['episode_index']:06d}.mp4").exists()
                    for ep in test_episodes
                )
                if not has_files:
                    continue
                    
                dst_video_dir = TEST_ROOT / "videos/chunk-000" / video_key_dir.name
                dst_video_dir.mkdir(parents=True, exist_ok=True)
                copied_count = 0
                for ep in test_episodes:
                    ep_idx = ep["episode_index"]
                    src_mp4 = video_key_dir / f"episode_{ep_idx:06d}.mp4"
                    if src_mp4.exists():
                        shutil.copy(src_mp4, dst_video_dir / f"episode_{ep_idx:06d}.mp4")
                        copied_count += 1
                if copied_count > 0:
                    copied_video_keys.append(video_key_dir.name)
                    print(f"  Copied videos: {video_key_dir.name}/ ({copied_count} files)")
    
    # 10. Fix info.json features: remove video features that don't exist in our subset
    print(f"  Video features copied: {copied_video_keys}")
    features = test_info.get("features", {})
    features_to_remove = []
    for feat_name, feat_def in features.items():
        if feat_def.get("dtype") == "video":
            # Video feature name in info.json should match video folder name
            if feat_name not in copied_video_keys:
                features_to_remove.append(feat_name)
    for feat_name in features_to_remove:
        del features[feat_name]
        print(f"  Removed missing video feature from info.json: {feat_name}")
    test_info["features"] = features
    test_info["total_videos"] = len(test_episodes) * len(copied_video_keys)
    
    # Rewrite info.json with fixed features
    with open(meta_dir / "info.json", "w") as f:
        json.dump(test_info, f, indent=2, ensure_ascii=False)
    
    print(f"\nDone! Test dataset created at: {TEST_ROOT}")
    print(f"  Episodes: {len(test_episodes)}")
    print(f"  Tasks: {len(test_tasks)}")
    print(f"  Total frames: {total_frames}")
    print(f"\nNow run conversion:")
    print(f"  1. Update run_bridge_v20_to_v21.sh: DATASET_ROOT=\"{TEST_ROOT}\"")
    print(f"  2. bash VLA_Annotation/Format_Transfer/run_bridge_v20_to_v21.sh")

if __name__ == "__main__":
    main()
