# LeRobot Dataset v3.0 to v2.1 Converter

Convert LeRobot datasets from v3.0 (file-based layout) to v2.1 (episode-based layout).

## Key Differences Between v3.0 and v2.1

| Aspect | v3.0 (file-based) | v2.1 (episode-based) |
|--------|-------------------|----------------------|
| Data files | `data/chunk-XXX/file-YYY.parquet` | `data/chunk-XXX/episode_NNNNNN.parquet` |
| Videos | `videos/{key}/chunk-XXX/file-YYY.mp4` (multiple episodes concatenated) | `videos/chunk-XXX/{key}/episode_NNNNNN.mp4` (one per episode) |
| Episodes meta | `meta/episodes/chunk-XXX/file-YYY.parquet` | `meta/episodes.jsonl` |
| Episode stats | Embedded in episodes parquet | `meta/episodes_stats.jsonl` |
| Tasks | `meta/tasks.parquet` | `meta/tasks.jsonl` |

## Prerequisites

### 1. Install dependencies

```bash
# Required Python packages
pip install pyarrow jsonlines tqdm numpy pandas datasets

# Downgrade datasets if you encounter List/Column issues
pip install "datasets<4.0.0"
```

### 2. Install ffmpeg (required for video splitting)

```bash
# CentOS/RHEL
sudo yum install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Via conda
conda install -c conda-forge ffmpeg
```

```bash
git clone https://github.com/huggingface/lerobot.git
cd /mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/lerobot

# 设置跳过 LFS 的 smudge filter
git lfs install --skip-smudge

# 重新 checkout（跳过 LFS 文件）
git reset --hard HEAD


pip install -e .
```
## Quick Start

### Option 1: Use the wrapper script (recommended)

```bash
cd /cpfs04/shared/Group-m6/tongzai.hxt/Qwen_VLA/VLA_Annotation/Format_Transfer/any4lerobot/ds_version_convert/v30_to_v21

# Dry run - list all v3.0 datasets without converting
./run_convert.sh --dry-run

# Convert a single dataset (for testing)
./run_convert.sh --single benchmark1_0_compressed/agilex_3rgb/1_potatooven

# Convert all datasets
./run_convert.sh
```

### Option 2: Use Python scripts directly

```bash
conda activate any4lerobot
cd /cpfs04/shared/Group-m6/tongzai.hxt/Qwen_VLA/VLA_Annotation/Format_Transfer/any4lerobot/ds_version_convert/v30_to_v21

# Batch convert all v3.0 datasets
python batch_convert_v30_to_v21.py \
    --input-root /cpfs04/shared/Group-m6/tongzai.hxt/Qwen_VLA/WQY_VLA_DATA/RoboMIND_lerobot/RoboMIND_lerobot \
    --output-root /cpfs04/shared/Group-m6/tongzai.hxt/Qwen_VLA/WQY_VLA_DATA/RoboMIND_lerobot_v21_converted

# Verify converted datasets
python verify_v21_dataset.py \
    --root /cpfs04/shared/Group-m6/tongzai.hxt/Qwen_VLA/WQY_VLA_DATA/RoboMIND_lerobot_v21_converted \
    --batch \
    --sample-size 3
```

## Scripts

### `batch_convert_v30_to_v21.py`

Batch converter that:
- Recursively finds all v3.0 datasets by checking `meta/info.json`
- Converts each to v2.1 format in a mirrored output directory
- Does NOT modify the original input data

**Arguments:**
- `--input-root`: Root directory containing v3.0 datasets
- `--output-root`: Root directory for v2.1 output (mirrored structure)
- `--dry-run`: List datasets without converting
- `--single RELPATH`: Convert only a single dataset

### `verify_v21_dataset.py`

Verification tool that checks:
- Required files exist (info.json, episodes.jsonl, episodes_stats.jsonl, tasks.jsonl)
- Episode parquet files match episodes.jsonl entries
- Video files exist and frame counts approximately match episode length
- Timestamps are monotonically increasing

**Arguments:**
- `--root`: Path to the v2.1 dataset or root directory
- `--sample-size`: Number of episodes to sample (default: 3)
- `--batch`: Verify all v2.1 datasets under root
- `--verbose`: Show detailed output

### `convert_dataset_v30_to_v21.py`

Original single-dataset converter (from any4lerobot). Requires lerobot installed.

## Output Structure

After conversion, each dataset will have this structure:

```
dataset_name/
├── meta/
│   ├── info.json           # codebase_version = "v2.1"
│   ├── episodes.jsonl      # One JSON object per line per episode
│   ├── episodes_stats.jsonl# Per-episode statistics
│   └── tasks.jsonl         # Task descriptions
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
└── videos/
    └── chunk-000/
        ├── observation.images.camera_front/
        │   ├── episode_000000.mp4
        │   ├── episode_000001.mp4
        │   └── ...
        └── observation.images.camera_left_wrist/
            └── ...
```

## Troubleshooting

### ffmpeg not found
Install ffmpeg (see Prerequisites above).

### Memory issues with large datasets
The converter processes one episode at a time. If you still have memory issues, try reducing the number of workers or processing fewer datasets at once.

### Video frame count mismatch
The verification allows 1-2 frame drift due to ffmpeg's seeking accuracy. Larger mismatches may indicate conversion issues.

### "datasets" version issues
If you see errors about `List` or `Column` types, downgrade the datasets package:
```bash
pip install "datasets<4.0.0"
```

## Performance Notes

- Video splitting is the slowest part (uses ffmpeg per episode)
- Converting a dataset with 99 episodes and 3 video streams takes approximately 5-10 minutes
- The converter skips already-converted datasets (checks if output directory exists)
