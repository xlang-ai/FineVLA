# RoboFine-Bench

Evaluation code for RoboFine-Bench — a benchmark for fine-grained robotic manipulation understanding.

## Setup

```bash
pip install openai tqdm
```

Download the benchmark data from Hugging Face:
```bash
# Or clone directly
git clone https://huggingface.co/datasets/FineVLA/RoboFine-Bench data/
```

## Evaluation

### VQA Evaluation

```bash
# Run all models
bash RoboFine-Bench/vqa_eval/run_vqa_eval.sh

# Single model
python RoboFine-Bench/vqa_eval/run_vqa.py \
    --model <model_name> \
    --num-workers 16
```

### Caption Evaluation

```bash
# Step 1: Generate captions
python RoboFine-Bench/caption_eval/annotate/run_annotate.py \
    --input data/EvalSets.json \
    --model <model_name> \
    --num_workers 16

# Step 2: Atomic fact alignment scoring
bash RoboFine-Bench/caption_eval/atomic_eval/run_atomic_eval.sh
```

## Structure

```
RoboFine-Bench/
├── vqa_eval/                  # VQA evaluation pipeline
│   ├── run_vqa.py             # Main VQA runner
│   ├── vqa_eval.py            # Evaluation logic
│   ├── vqa_config.py          # Model/API configuration
│   ├── vqa_prompts.py         # Prompt templates
│   └── vqa_report.py          # Score reporting
├── caption_eval/
│   ├── annotate/              # Caption generation
│   │   ├── run_annotate.py    # Main caption runner
│   │   ├── stages.py          # Multi-stage pipeline
│   │   ├── dataset_configs/   # Per-dataset configurations
│   │   └── ...
│   └── atomic_eval/           # Atomic fact alignment
│       ├── atomic_eval/       # Core evaluation package
│       ├── prompts/           # Judge prompt templates
│       └── run_atomic_eval.sh
└── statistics/                # Benchmark statistics & plots
```

## Data

Benchmark data is hosted on Hugging Face: [FineVLA/RoboFine-Bench](https://huggingface.co/datasets/FineVLA/RoboFine-Bench)

- `videos/` — Robot manipulation videos from 10 datasets
- `EvalSets.json` — Caption evaluation set with GT annotations
- `QAEvalSets.json` — VQA questions and answers
- `GT_AtomicFacts.jsonl` — Ground truth atomic facts for scoring
