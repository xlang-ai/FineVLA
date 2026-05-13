#!/bin/bash
# =============================================================================
# BC_Z dataset: LeRobot v2.0 → v2.1 conversion runner
# =============================================================================
# Uses existing v20_to_v21 converter. Does NOT modify the converter code.
# Direct conversion: in-place at dataset root (no copy). Optionally move to target.
#
# v2.0 assumptions (source):
#   - meta/info.json (codebase_version "v2.0"), episodes.jsonl, tasks.jsonl
#   - meta/stats.json (aggregate stats for check)
#   - data/chunk-XXX/episode_XXXXXX.parquet, videos/chunk-XXX/<key>/episode_XXXXXX.mp4
#
# v2.1 guarantees (after conversion):
#   - meta/info.json codebase_version set to "v2.1"
#   - meta/episodes_stats.jsonl generated (per-episode stats; chunk layout unchanged)
#   - Episodes/tasks/parquet/videos unchanged; only stats + version bump
# =============================================================================

set -e

# 提高单进程可打开文件数，避免 "Too many open files"（39k+ episodes 会检查大量路径）
ulimit -n 1048576 2>/dev/null || true

# --- Config ---
# 输入: v2.0 数据集路径（原地转换后可选移动到 TARGET_ROOT）
DATASET_ROOT="/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v20/BC_Z"
# 输出: 转换完成后移动到此路径（MOVE_AFTER=1 时生效）
TARGET_ROOT=""
# 1 = 转换完成后 mv 到 TARGET_ROOT（原路径将为空）；0 = 转换后仍在 DATASET_ROOT
MOVE_AFTER=0
REPO_ID="BC_Z"
NUM_WORKERS=12
# 并行后端: "process" (ProcessPoolExecutor) 或 "ray" (Ray Actors, 推荐大数据集)
BACKEND="ray"

# 纯本地转换，禁止访问 HuggingFace Hub（避免 SSL 错误和不必要的网络请求）
export HF_HUB_OFFLINE=1

# Path to existing v20_to_v21 converter (run from this dir so imports work)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V20_TO_V21_DIR="${SCRIPT_DIR}/any4lerobot/ds_version_convert/v20_to_v21"

# --- Checks ---
if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "Error: Dataset root not found: $DATASET_ROOT"
    exit 1
fi
if [[ ! -f "$DATASET_ROOT/meta/info.json" ]]; then
    echo "Error: Missing meta/info.json (not a v2.0 dataset root?)"
    exit 1
fi
if [[ ! -f "$DATASET_ROOT/meta/stats.json" ]]; then
    echo "Error: Missing meta/stats.json (required for check_aggregate_stats)"
    exit 1
fi
if [[ ! -d "$V20_TO_V21_DIR" ]]; then
    echo "Error: Converter dir not found: $V20_TO_V21_DIR"
    exit 1
fi
if [[ "$MOVE_AFTER" == "1" ]] && [[ -e "$TARGET_ROOT" ]]; then
    echo "Error: TARGET_ROOT already exists: $TARGET_ROOT"
    echo "  Remove it first or set MOVE_AFTER=0 to keep result at DATASET_ROOT."
    exit 1
fi

# --- Direct conversion in-place (no copy) ---
echo "Running v20→v21 conversion in-place at: $DATASET_ROOT"
cd "$V20_TO_V21_DIR"

# --skip-aggregate-check: 跳过 episode stats 和 stats.json 的一致性检查（仅子集测试时使用）
python convert_dataset_v20_to_v21.py \
    --repo-id="$REPO_ID" \
    --root="$DATASET_ROOT" \
    --num-workers="$NUM_WORKERS" \
    --backend="$BACKEND"

echo "Conversion finished. Dataset (v2.1) is at: $DATASET_ROOT"

# --- Optional: move result to target (no duplicate data) ---
if [[ "$MOVE_AFTER" == "1" ]]; then
    echo "Moving dataset to: $TARGET_ROOT"
    mkdir -p "$(dirname "$TARGET_ROOT")"
    mv "$DATASET_ROOT" "$TARGET_ROOT"
    echo "Done. Output dataset: $TARGET_ROOT"
    echo "Verify with: python verify_v21_dataset.py --root \"$TARGET_ROOT\" --verbose"
else
    echo "Verify with: python verify_v21_dataset.py --root \"$DATASET_ROOT\" --verbose"
fi
