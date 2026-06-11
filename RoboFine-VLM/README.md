# RoboFine-VLM Caption Annotation

This directory is a lightweight, self-contained guide for deploying RoboFine-VLM and using it to generate RoboFine-Bench caption annotations.

The fixed caption annotation parameters live in:

- `config/caption_annotation_config.json`
- `robofine_caption_prompt.py`

These files mirror the parameters currently used in RoboFine-Bench caption evaluation. The RB new prompt is the fixed benchmark prompt and should not be edited when comparing models.

## Fixed Annotation Setting

| Item | Value |
|---|---|
| Views | use all available views, keep original order |
| View labels | generated dynamically from each sample's actual view names |
| FPS | `4.0` |
| Max frames | `512` per view |
| Frame resize | resize width to `512` before base64 encoding |
| Prompt | fixed RB new prompt in `robofine_caption_prompt.py`, exactly matching `../RoboFine-Bench/caption_eval/annotate/prompts.py` |
| Default mode | `hard`, no task instruction |
| Optional mode | `easy`, appends task instruction |
| Temperature | `0.0` |
| Top-p | `0.95` |
| Max tokens | `32768` |
| Pixel overrides | none: do not set `min_pixels`, `max_pixels`, `total_pixels`, or `mm_processor_kwargs` |
| Output | JSON only: `{ "Step1": "...", "Step2": "...", "StepN": "..." }` |

## Deploy RoboFine-VLM

Serve RoboFine-VLM through an OpenAI-compatible vLLM endpoint:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OPENAI_API_KEY=EMPTY

python -m vllm.entrypoints.openai.api_server \
    --model /path/to/RoboFine-VLM \
    --served-model-name robofine-vlm \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code
```

The command used for the current local RoboFine-VLM tests is:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/RoboFine-VLM-opensource \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --max-model-len 262144 \
    --reasoning-parser qwen3 \
    --dtype bfloat16 \
    --enforce-eager \
    --limit-mm-per-prompt '{"image": 1700}' \
    --additional-config '{"gdn_prefill_backend": "triton"}'
```

The service endpoint is:

```text
http://localhost:8000/v1
```

Check the service before calling it:

```bash
curl http://localhost:8000/v1/models
```

## Multi-View Request Format

For RoboFine-VLM, the benchmark-compatible request uses sampled image frames
grouped as OpenAI-compatible `video` parts. Each view is preceded by a dynamic
view label, and each view's `video` field is a list of base64 JPEG image URLs:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "[View: This is the main view of the video.]"},
            {
                "type": "video",
                "video": [
                    "data:image/jpeg;base64,...",
                    "data:image/jpeg;base64,...",
                ],
                "fps": 4.0,
            },
            {"type": "text", "text": "[View: This is the left wrist view of the video.]"},
            {
                "type": "video",
                "video": [
                    "data:image/jpeg;base64,...",
                    "data:image/jpeg;base64,...",
                ],
                "fps": 4.0,
            },
            {"type": "text", "text": "[View: This is the right wrist view of the video.]"},
            {
                "type": "video",
                "video": [
                    "data:image/jpeg;base64,...",
                    "data:image/jpeg;base64,...",
                ],
                "fps": 4.0,
            },
            {"type": "text", "text": RB_NEW_CAPTION_PROMPT},
        ],
    }
]
```

Use these generation parameters:

```python
client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=0.0,
    top_p=0.95,
    max_tokens=32768,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
```

`image_url` parts are also accepted by the local vLLM service, but the default
reported setting is the grouped `type="video"` frame-list format above because
it preserves the video sequence and `fps=4.0` explicitly.

## Run Caption Annotation On RoboFine-Bench

This script reuses the annotation runner in `../RoboFine-Bench` but keeps the RoboFine-VLM-facing parameters here.

Hard mode:

```bash
cd /cpfs01/data/shared/Group-m6/tongzai.hxt/FineVLA/RoboFine-VLM

MODE=hard \
INPUT_TYPE=image \
MODEL=robofine-vlm \
BASE_URL=http://localhost:8000/v1 \
WORKERS=8 \
./scripts/run_robofine_caption_benchmark.sh
```

Easy mode:

```bash
cd /cpfs01/data/shared/Group-m6/tongzai.hxt/FineVLA/RoboFine-VLM

MODE=easy \
INPUT_TYPE=image \
MODEL=robofine-vlm \
BASE_URL=http://localhost:8000/v1 \
WORKERS=8 \
./scripts/run_robofine_caption_benchmark.sh
```

`INPUT_TYPE=image` is recommended for RoboFine-VLM because it is robust across OpenAI-compatible vLLM deployments. Use `INPUT_TYPE=video` only if the service supports direct video parts.

The output is a `CaptionResult.jsonl` file under:

```text
../RoboFine-Bench/caption_eval/result/caption/robofine_${MODE}_${INPUT_TYPE}/
```

## Run The Built-In Three-View Video Demo

This directory includes a local three-view Galaxea demo video:

```text
assets/demo_three_view/
├── main.mp4
├── left_wrist.mp4
└── right_wrist.mp4
```

Run RoboFine-VLM on the demo videos:

```bash
cd /cpfs01/data/shared/Group-m6/tongzai.hxt/FineVLA/RoboFine-VLM

python examples/caption_video_demo.py \
    --base-url http://localhost:8000/v1 \
    --model /cpfs01/data/shared/Group-m6/tongzai.hxt/RoboFine-VLM-opensource \
    --mode hard \
    --print-request
```

The script:

- reads local mp4 files
- samples frames at `fps=4.0`
- keeps up to `512` sampled frames per view by default
- resizes sampled frames to width `512` before encoding
- encodes sampled frames as `data:image/jpeg;base64,...`
- keeps all provided views
- inserts dynamic `[View: ...]` labels
- sends each view as one OpenAI-compatible `video` part whose `video` field is a list of base64 image URLs
- uses `temperature=0.0`, `top_p=0.95`, `max_tokens=32768`
- for local RoboFine/Qwen vLLM endpoints, sends `extra_body={"chat_template_kwargs": {"enable_thinking": false}}`
- sends no pixel overrides

The default `--max-frames-per-view` is `512`, and the default `--resize-width` is `512`, matching the RoboFine-Bench local image/base64 path. Only lower frame count for smoke tests; do not lower it for reported benchmark numbers.

To run your own videos:

```bash
python examples/caption_video_demo.py \
    --base-url http://localhost:8000/v1 \
    --model robofine-vlm \
    --mode easy \
    --instruction "open the drawer and place the item inside" \
    --view main=/path/to/main.mp4 \
    --view left_wrist=/path/to/left_wrist.mp4 \
    --view right_wrist=/path/to/right_wrist.mp4
```

The video preprocessing code is in:

```text
video_utils.py
```

## Run A Single Frame API Call

Use `examples/caption_api_demo.py` to verify the deployed endpoint with already-sampled frames.

```bash
cd /cpfs01/data/shared/Group-m6/tongzai.hxt/FineVLA/RoboFine-VLM

python examples/caption_api_demo.py \
    --base-url http://localhost:8000/v1 \
    --model robofine-vlm \
    --mode hard \
    --view main=/path/to/main_000.jpg \
    --view main=/path/to/main_001.jpg \
    --view left_wrist=/path/to/left_000.jpg \
    --view right_wrist=/path/to/right_000.jpg \
    --print-request
```

The demo sends each actual view as one `video` part containing sampled frame images and attaches `fps=4.0`, matching the benchmark call. View labels are generated dynamically from the role names passed through `--view`.

For Easy mode:

```bash
python examples/caption_api_demo.py \
    --base-url http://localhost:8000/v1 \
    --model robofine-vlm \
    --mode easy \
    --instruction "pick up the cup and place it on the plate" \
    --view main=/path/to/main_000.jpg
```

## View Labels

For multi-view inputs, the request inserts text labels before each actual view. The label is not part of the fixed prompt text; it is generated by the caller according to the view name and inserted into the multimodal input before that view's frames.

```text
[View: This is the main view of the video.]
[View: This is the left wrist view of the video.]
[View: This is the right wrist view of the video.]
```

The first view is always treated as main. Later views are classified by names such as `left_wrist`, `right_wrist`, `wrist`, `left`, or `right`. This supports 1-view, 2-view, 3-view, or other multi-view model inputs without changing the prompt.

## Score Captions

After caption generation, score the output with a judge LLM:

```bash
cd /cpfs01/data/shared/Group-m6/tongzai.hxt/FineVLA/RoboFine-Bench

export OPENAI_API_KEY="your-judge-api-key"

python -m caption_eval.atomic_eval.atomic_eval direct-align \
    --gt-facts EvalData/GT_AtomicFacts.jsonl \
    --caption caption_eval/result/caption/robofine_hard_image/robofine-vlm_CaptionResult.jsonl \
    --output-dir caption_eval/result/DirectAlign/robofine_hard_image \
    --model openai.gpt-5.4-2026-03-05 \
    --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --num-workers 8 \
    --enable-thinking
```

Report the judge model, endpoint, mode (`hard` or `easy`), input type (`image` or `video`), successful caption count, scored sample count, and token usage if available.
