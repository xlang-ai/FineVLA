#!/usr/bin/env bash
# =============================================================================
# Example: Annotate robot manipulation videos with a VLM
# =============================================================================
#
# Before running, set these environment variables:
#
#   export OPENAI_API_KEY="your-api-key"       # Required: VLM API key
#   export VIDEO_BASE_DIR="/path/to/videos"    # Required: video root directory
#
# Optional overrides:
#   export ANNOTATE_MODEL="qwen-vl-plus"       # Model name
#   export ANNOTATE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
#
# Video files should be at: $VIDEO_BASE_DIR/<dataset_dir>/<video_path>
# For example: /path/to/videos/Bridge/episode_0/video.mp4
# =============================================================================

set -euo pipefail

# --- Step 1: Dry-run to validate input and check video paths ---
echo "=== Dry-run: validating input ==="
python run_annotate.py \
    --input examples/sample_input.jsonl \
    --dry-run

# --- Step 2: Full annotation (uncomment to run) ---
# echo "=== Running annotation ==="
# python run_annotate.py \
#     --input examples/sample_input.jsonl \
#     --output outputs/results.jsonl \
#     --num_workers 4 \
#     --stage-fps analysis=4.0 \
#     --stage-fps refinement=1.0
