# RoboFine-Bench

Evaluation code for **RoboFine-Bench** — a benchmark for fine-grained robotic manipulation video understanding.

- **500 held-out videos** from 10 datasets, **32 embodiments**
- **VQA Track**: 1,030 questions across 3 axes (Grounding, Action & Motion, Interaction & State)
- **Caption Track**: Step-level action description with Consistency, Coverage, and Anti-Hallucination metrics

<p align="center">
  <img src="benchmark_overview.png" alt="RoboFine-Bench Overview" width="100%">
</p>

## 1. Download Data

Benchmark data is hosted on Hugging Face: [xlangai/RoboFine-bench](https://huggingface.co/datasets/xlangai/RoboFine-bench)

```bash
cd RoboFine-Bench/

# Install Git LFS (required for video files)
git lfs install

# Clone the dataset into EvalData/ directory
git clone https://huggingface.co/datasets/xlangai/RoboFine-bench EvalData/
```

After download, your directory should look like:

```
RoboFine-Bench/
├── EvalData/                          # Downloaded from HuggingFace
│   ├── EvalSets.json                  # Evaluation samples with GT annotations
│   ├── QAEvalSets.json                # VQA questions and answers
│   ├── GT_AtomicFacts.jsonl           # Pre-extracted GT atomic facts
│   └── videos/                        # Robot manipulation videos
│       ├── BridgeDataV2/
│       ├── BC-Z/
│       └── ...
├── vqa_eval/                          # VQA evaluation code
└── caption_eval/                      # Caption evaluation code
```

## 2. Installation

```bash
pip install openai httpx tqdm pydantic Pillow av
```

Set your API key (DashScope or OpenAI-compatible endpoint):

```bash
export OPENAI_API_KEY="your-api-key"
```

## 3. VQA Evaluation

The VQA track tests whether VLMs can answer fine-grained questions about robot manipulation videos.

### Single Model

```bash
python RoboFine-Bench/vqa_eval/run_vqa.py \
    --model qwen3-vl-plus \
    --qa EvalData/QAEvalSets.json \
    --input EvalData/EvalSets.json \
    --num-workers 16
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `qwen3-vl-plus` | VLM model name |
| `--qa` | `EvalData/QAEvalSets.json` | VQA questions file |
| `--input` | `EvalData/EvalSets.json` | Evaluation samples file |
| `--base-url` | DashScope URL | API endpoint |
| `--num-workers` | `1` | Parallel API call threads |
| `--thinking` | `true` | Enable model reasoning mode |
| `--round` | `None` | Round number (for multi-round evaluation) |
| `--dry-run` | `False` | Print stats only, no API calls |

**Output:** `vqa_eval/results/{model}_vqa_result.jsonl`

### Multi-Round Batch Evaluation

```bash
# Run 2 rounds for all built-in models
bash RoboFine-Bench/vqa_eval/run_vqa_eval.sh 2

# Run 3 rounds, limit to first 10 samples
bash RoboFine-Bench/vqa_eval/run_vqa_eval.sh 3 10
```

### Generate Score Report

```bash
# Print single model report
python RoboFine-Bench/vqa_eval/vqa_report.py vqa_eval/results/xxx_vqa_result.jsonl

# Update cross-model summary CSV
python RoboFine-Bench/vqa_eval/vqa_report.py --update-csv
```

**Output:** `vqa_eval/results/VQATest_Score.csv` with per-capability and per-answer-type accuracy.

Supports resume: already-evaluated questions are automatically skipped on re-run.

## 4. Caption Evaluation

The Caption track evaluates step-level action description quality through atomic fact alignment. It runs in two stages: **(A) Generate captions** → **(B) Score against GT atomic facts**.

### Stage A: Generate Captions

```bash
python RoboFine-Bench/caption_eval/annotate/run_annotate.py \
    --model qwen3.5-plus \
    --evalsets EvalData/EvalSets.json \
    --frame-index EvalData/frame_index.jsonl \
    --output-dir results/CaptionResult/ \
    --num-workers 16
```

For **hard mode** (no task instruction in prompt):
```bash
python RoboFine-Bench/caption_eval/annotate/run_annotate.py \
    --model qwen3.5-plus \
    --evalsets EvalData/EvalSets.json \
    --frame-index EvalData/frame_index.jsonl \
    --output-dir results/CaptionResult/hard/ \
    --num-workers 16 \
    --no-instruction
```

Or use the batch script:
```bash
bash RoboFine-Bench/caption_eval/annotate/run_annotation_eval.sh easy    # with instruction
bash RoboFine-Bench/caption_eval/annotate/run_annotation_eval.sh hard    # without instruction
```

**Output:** `{output_dir}/{model}_CaptionResult.jsonl`

### Stage B: Score Captions

Score the generated captions against ground-truth atomic facts using Direct Alignment:

```bash
python -m caption_eval.atomic_eval.atomic_eval direct-align \
    --gt-facts EvalData/GT_AtomicFacts.jsonl \
    --caption results/CaptionResult/qwen3_5-plus_CaptionResult.jsonl \
    --output-dir results/AtomicResult/qwen3_5-plus/ \
    --num-workers 8 \
    --enable-thinking
```

**Scoring metrics:**

| Metric | Formula | What it measures |
|--------|---------|------------------|
| Consistency | (Match + 0.5×Partial) / Total Alignments | Precision of matched facts |
| Coverage | (Match + 0.5×Partial) / GT Facts | Recall of GT facts |
| Anti-Hallucination | 1 - Hallucinated / GT_action_sequence | Penalizes fabricated actions |
| **CaptionScore** | 1/3 × (Consistency + Coverage + Anti-Hallucination) | Overall score |

**Output:**
- `scored_results.jsonl` — Per-sample scores
- `dataset_summary.json` / `dataset_summary.csv` — Aggregated scores by dataset and capability

### Cross-Model Summary

```bash
python -m caption_eval.atomic_eval.atomic_eval summary \
    --results-dirs results/AtomicResult/*/ \
    --output results/cross_model_summary.csv
```

Or use the batch script for all models:
```bash
bash RoboFine-Bench/caption_eval/run_direct_align.sh easy
bash RoboFine-Bench/caption_eval/run_direct_align.sh hard
```

## 5. Project Structure

```
RoboFine-Bench/
├── benchmark_overview.png             # Benchmark overview figure
├── prepare_frames.py                  # Pre-extract video frames for offline use
├── models/                            # Model interface for custom models
│   ├── base_model.py                  # BaseVLM abstract class
│   ├── api_model.py                   # Built-in OpenAI-compatible implementation
│   └── example_local_model.py         # Example: local HuggingFace model
├── eval_set/
│   └── prepare_evalsets_input.py      # Data preparation (internal use)
├── vqa_eval/
│   ├── run_vqa.py                     # VQA evaluation runner
│   ├── vqa_eval.py                    # Answer matching logic
│   ├── vqa_config.py                  # Dataset view/FPS configuration
│   ├── vqa_prompts.py                 # VQA prompt templates
│   ├── vqa_report.py                  # Score reporting & CSV
│   └── run_vqa_eval.sh               # Batch multi-round evaluation
└── caption_eval/
    ├── annotate/
    │   ├── run_annotate.py            # Caption generation runner
    │   ├── api_call.py                # Unified API client (Qwen/Gemini/GPT)
    │   ├── prompts.py                 # Caption prompt templates
    │   └── run_annotation_eval.sh     # Batch caption generation
    ├── run_direct_align.sh            # Batch Direct Alignment scoring
    └── atomic_eval/
        ├── run_judge.py               # LLM-as-a-Judge (legacy method)
        ├── prompts/                   # Judge prompt templates
        └── atomic_eval/              # Core evaluation package
            ├── cli.py                 # CLI subcommands
            ├── pipeline.py            # Evaluation pipeline
            ├── scoring.py             # Metric computation
            └── ...
```

## 6. Supported Models

The evaluation scripts support multiple VLM providers out of the box:

| Provider | Model Examples |
|----------|---------------|
| Qwen (DashScope) | `qwen3-vl-plus`, `qwen3.5-plus` |
| Google (Gemini) | `vertex_ai.gemini-3.1-pro-preview` |
| OpenAI | `openai.gpt-5.4-2026-03-05` |
| Doubao | `doubao.doubao-seed-2-0-pro-260215` |

## 7. Evaluate Your Own Model

To evaluate a custom model (local HuggingFace model, vLLM server, etc.), implement the `BaseVLM` interface:

### Step 1: Pre-extract frames

```bash
python prepare_frames.py \
    --evalsets EvalData/EvalSets.json \
    --video-dir EvalData/Videos \
    --output-dir EvalData/frames \
    --fps 2.0
```

### Step 2: Implement BaseVLM

```python
from models.base_model import BaseVLM

class MyModel(BaseVLM):
    def __init__(self, model_path):
        # Load your model
        ...

    def generate(self, images, prompt, system_prompt=""):
        # images: list of PIL.Image (video frames at 2 FPS)
        # Return: (response_text, token_usage_dict)
        ...
        return response, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
```

See `models/example_local_model.py` for a complete example.

### Step 3: Run evaluation

Use your model with the built-in evaluation scripts, or write a simple loop:

```python
from models import APIModel  # or your custom model
from vqa_eval.vqa_eval import evaluate_answer

model = APIModel("qwen3-vl-plus")  # or MyModel("path/to/model")
response, tokens = model.generate(frames, prompt, system_prompt)
```

## Data

Benchmark data is hosted on Hugging Face: [xlangai/RoboFine-bench](https://huggingface.co/datasets/xlangai/RoboFine-bench)

| File | Description |
|------|-------------|
| `EvalSets.json` | 500 evaluation samples with GT step annotations, video URLs, and metadata |
| `QAEvalSets.json` | 1,030 VQA questions with answers, capabilities, and answer types |
| `GT_AtomicFacts.jsonl` | Pre-extracted GT atomic facts across 10 capability dimensions |
| `frame_index.jsonl` | Pre-uploaded frame URLs for efficient Caption generation |
| `Videos/` | Robot manipulation video files organized by dataset |
