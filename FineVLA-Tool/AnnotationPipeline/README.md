# Annotate Pipeline

Automatic annotation pipeline for robot manipulation videos using Vision-Language Models (VLMs). Given a video of a robot performing a task and a coarse instruction (e.g., "pick up the cup"), this tool generates structured, fine-grained action descriptions by analyzing the video frame-by-frame through a VLM.

## What It Does

**Input:** A robot manipulation video + a coarse task instruction

**Output:** Structured action annotations with fine-grained step descriptions

```
Input:  video of a robot + "pick up the red cup and place it on the plate"

Output:
{
  "analysisResult": {
    "action_sequence": ["Grasp", "Pick up", "Move", "Place", "Release"],
    "main_object": "red cup"
  },
  "fineGrainedSteps": [
    "Grasp the red cup by its handle from the right side using a lateral grip.",
    "Pick up the cup vertically, lifting it approximately 15cm above the table.",
    "Move the cup horizontally to the left toward the white plate.",
    "Place the cup upright at the center of the plate.",
    "Release the grip and retract the arm upward to resting position."
  ],
  "refinedInstruction": "Grasp the red cup by its handle from the right side, lift it vertically, move it to the left toward the white plate, place it upright at the center of the plate, and release."
}
```

## How It Works

The pipeline runs in **stages**, each calling a VLM to progressively refine the annotation:

```
Stage 1: Analysis       →  Extract action sequence + identify main object
                              (low FPS sampling, coarse understanding)
                                          │
Stage 2: Refinement     →  Generate fine-grained step descriptions
                              (higher FPS sampling, detailed per-step descriptions)
                                          │
Stage 3 (optional):     →  Polish steps using wrist/close-up camera view
  Detail Refinement          (available for multi-camera setups like RDT)
```

## Prerequisites

- **Python 3.8+**
- **A VLM API** that supports vision (image) inputs with an OpenAI-compatible interface. Tested with:
  - [DashScope](https://dashscope.aliyuncs.com/) (Qwen-VL series) — default
  - Any OpenAI-compatible endpoint (vLLM, Ollama, etc.)
- **Robot manipulation videos** organized in a directory structure

## Installation

```bash
git clone https://github.com/your-org/Annotate_Pipeline.git
cd Annotate_Pipeline
pip install -r requirements.txt
```

**Optional (recommended):**
```bash
pip install av             # PyAV: fallback video decoder (handles AV1, etc.)
pip install opencv-python  # ~6x faster frame encoding than PIL
```

> **Note:** `decord` is the primary video decoder. If it fails to install on your system, the pipeline will automatically fall back to PyAV.

## Configuration

All configuration is done via **environment variables**:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | API key for your VLM service |
| `VIDEO_BASE_DIR` | Yes | `./data/videos` | Root directory containing video files |
| `ANNOTATE_MODEL` | No | `qwen-vl-plus` | Model name to use |
| `ANNOTATE_BASE_URL` | No | `https://dashscope.aliyuncs.com/compatible-mode/v1` | API endpoint URL |

**Example setup:**
```bash
# Using DashScope (Qwen-VL)
export OPENAI_API_KEY="sk-your-dashscope-api-key"
export VIDEO_BASE_DIR="/data/robot_videos"
export ANNOTATE_MODEL="qwen-vl-plus"

# Using a local vLLM server
export OPENAI_API_KEY="dummy"
export ANNOTATE_BASE_URL="http://localhost:8000/v1"
export ANNOTATE_MODEL="your-local-vlm"
```

> **Why `OPENAI_API_KEY`?** The pipeline uses the OpenAI-compatible API protocol. Most VLM services (DashScope, vLLM, Ollama, etc.) support this protocol, so the same environment variable works for all of them.

## Input Format

The input is a **JSONL file** (one JSON object per line). Each line represents one video sample to annotate.

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `sample_id` | string | Unique identifier for this sample |
| `dataset` | string | Dataset name (used to select pipeline config, see [Supported Datasets](#supported-datasets)) |
| `instruction_raw` | string | Coarse task instruction (e.g., "pick up the cup") |
| `views` | list[string] | Available camera view names |
| `videos` | dict | Mapping from view name to relative video path |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `dataset_dir` | string | Subdirectory under `VIDEO_BASE_DIR` for this dataset |
| `robot_type` | string | Robot platform identifier (used for prompt selection) |
| `steps_raw` | list[dict] | Pre-annotated step boundaries with `{i, desc, start, end}` for per-step refinement |

### Example

```jsonl
{"sample_id": "ep_001", "dataset": "bridge", "dataset_dir": "Bridge", "instruction_raw": "pick up the red cup and place it on the plate", "robot_type": "widowx", "views": ["image_0"], "videos": {"image_0": "episode_0/video.mp4"}}
```

### Video Path Resolution

The full path to a video file is constructed as:

```
{VIDEO_BASE_DIR} / {dataset_dir} / {videos[view_name]}
```

For example, with `VIDEO_BASE_DIR=/data/robot_videos`:
```
/data/robot_videos/Bridge/episode_0/video.mp4
```

### Using a Custom Dataset Name

If your dataset is not in the [supported list](#supported-datasets), use any name you like. The pipeline will use `DefaultConfig` which runs a standard 2-stage pipeline (analysis + refinement) and auto-selects the best camera view.

## Usage

### Basic Run

```bash
# Dry-run: validate input format and video paths, no API calls
python run_annotate.py \
    --input your_data.jsonl \
    --dry-run

# Full annotation
python run_annotate.py \
    --input your_data.jsonl \
    --output results.jsonl \
    --num_workers 8
```

### Control Frame Sampling

```bash
# Override FPS per stage
python run_annotate.py \
    --input data.jsonl \
    --output results.jsonl \
    --stage-fps analysis=4.0 \
    --stage-fps refinement=2.0
```

### Filter Samples

```bash
# Only process specific datasets
python run_annotate.py --input data.jsonl --output results.jsonl \
    --dataset bridge,droid_1.0.1

# Process a slice (useful for testing)
python run_annotate.py --input data.jsonl --output results.jsonl \
    --start 0 --end 10
```

### Resume from Interruption

The pipeline writes results incrementally to JSONL. If interrupted, simply re-run the same command — already-completed samples (by `sample_id`) are automatically skipped.

### CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | (required) | Input JSONL file path |
| `--output` | (required) | Output JSONL file path |
| `--video-base-dir` | `$VIDEO_BASE_DIR` | Video root directory |
| `--dataset` | all | Comma-separated dataset names to include |
| `--mode` | `all` | `annotate` (no steps_raw), `review` (has steps_raw), or `all` |
| `--start` / `--end` | 0 / all | Sample index range for slicing |
| `--num_workers` | CPU count | Number of parallel workers |
| `--model` | `$ANNOTATE_MODEL` | VLM model name |
| `--base_url` | `$ANNOTATE_BASE_URL` | API endpoint URL |
| `--stage-fps STAGE=FPS` | per-dataset | Override target FPS for a stage (repeatable) |
| `--stage-ratio STAGE=R` | — | Override sample ratio for a stage (repeatable) |
| `--max_frames` | 100000 | Maximum frames per API call |
| `--dry-run` | — | Validate input only, no API calls |
| `--quiet` | — | Suppress per-sample log output |

## Output Format

Each line in the output JSONL contains the original input fields plus annotation results:

```json
{
  "sample_id": "ep_001",
  "dataset": "bridge",
  "instruction_raw": "pick up the red cup and place it on the plate",
  "analysisResult": {
    "action_sequence": ["Grasp", "Pick up", "Move", "Place", "Release"],
    "main_object": "red cup"
  },
  "refinedInstruction": "Grasp the red cup by its handle from the right...",
  "fineGrainedSteps": [
    "Grasp the red cup by its handle from the right side.",
    "Pick up the cup vertically...",
    "..."
  ],
  "processingMetadata": {
    "success": true,
    "model": "qwen-vl-plus",
    "processingTime": 12.34,
    "tokenUsage": {"prompt_tokens": 5000, "completion_tokens": 500, "total_tokens": 5500}
  }
}
```

## Supported Datasets

The pipeline includes built-in configurations for these robot datasets:

| Dataset | Config | Stages | Key Feature |
|---------|--------|--------|-------------|
| Bridge | `BridgeConfig` | analysis -> refinement | Single view (image_0) |
| BC-Z | `BcZConfig` | analysis -> refinement | Single view (image) |
| RT-1 | `Rt1Config` | analysis -> refinement | 3fps analysis (matches capture rate) |
| DROID | `DroidConfig` | 2-stage or per-step | Adaptive: uses steps_raw when available |
| RDT | `RdtConfig` | 3-stage | Adds wrist-camera detail refinement |
| Galaxea | `GalaxeaConfig` | per-step refinement | Clips video per step, bimanual prompts |
| AgiBotWorld | `AgiBotWorldConfig` | per-step refinement | Head camera, bimanual prompts |
| RoboMind V1/V2 | `RobomindV1/V2Config` | 2-stage or per-step | Random view assignment across stages |
| RH20T | `Rh20tConfig` | 2-stage or per-step | Multi-view random selection |
| RoboCoin | `RobocoinConfig` | 2-stage or per-step | Priority-based view selection, custom prompts |

For unlisted datasets, `DefaultConfig` is used automatically (2-stage pipeline with keyword-based view selection).

## Adding a New Dataset

1. **Create a config file** `dataset_configs/my_dataset.py`:

```python
from .base import BaseDatasetConfig, StageDefinition

class MyDatasetConfig(BaseDatasetConfig):
    dataset_name = "my_dataset"

    stages = [
        StageDefinition(name="analysis", fn_name="analysis", default_fps=4.0),
        StageDefinition(name="refinement", fn_name="refinement",
                        depends_on="analysis", default_fps=1.0),
    ]

    def resolve_stage_views(self, sample, video_base_dir=""):
        """Determine which camera view and video path to use for each stage."""
        videos = sample.get("videos", {})
        base = video_base_dir or "./data/videos"
        dataset_dir = sample.get("dataset_dir", "")

        # Use "front" view if available, otherwise first view
        view = "front" if "front" in videos else list(videos.keys())[0]
        import os
        video_path = os.path.join(base, dataset_dir, videos[view])

        return {
            "analysis": {"view": view, "video_path": video_path},
            "refinement": {"view": view, "video_path": video_path},
        }
```

2. **Register it** in `dataset_configs/__init__.py`:

```python
from .my_dataset import MyDatasetConfig
DATASET_CONFIGS["my_dataset"] = MyDatasetConfig()
```

3. **Use it** — set `"dataset": "my_dataset"` in your input JSONL.

### Customizing Prompts

Override `get_prompt_override()` in your config to use custom system/user prompts:

```python
def get_prompt_override(self, stage, sample=None):
    if stage == "analysis":
        return ("Your custom system prompt...", "Your custom user template...")
    return None  # Use default prompts
```

## Project Structure

```
Annotate_Pipeline/
├── run_annotate.py          # CLI entry point
├── runner.py                # Parallel orchestrator (multiprocessing + resume)
├── stages.py                # Stage functions (analysis, refinement, etc.)
├── api_client.py            # OpenAI-compatible VLM client with retry
├── video_utils.py           # Video decode, frame sampling, JPEG encoding
├── data_types.py            # TrajectoryItem, StageResult, ProcessingResult
├── config.py                # Global defaults (FPS, model, retry settings)
├── prompts/                 # Prompt templates by stage and robot type
│   ├── vocabulary.py        #   Action vocabulary (14 verbs + guidance)
│   ├── analysis.py          #   Analysis prompts
│   ├── refinement.py        #   Refinement prompts
│   ├── detail_refinement.py #   Wrist-view + multi-view prompts
│   ├── bimanual.py          #   Dual-arm robot prompts
│   ├── galaxea_step.py      #   Per-step clip refinement prompts
│   └── robocoin.py          #   RoboCoin-specific prompts
├── dataset_configs/         # Per-dataset pipeline configs
│   ├── base.py              #   BaseDatasetConfig + StageDefinition
│   ├── default.py           #   Fallback config for unknown datasets
│   └── *.py                 #   One file per supported dataset
├── examples/
│   └── sample_input.jsonl   #   Example input format
└── run_example.sh           #   Example run script
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
