# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fork of StarVLA — a modular VLA (Vision-Language-Action) codebase for robot learning. Focus is on training infrastructure and framework development. Built on PyTorch + Accelerate + DeepSpeed with OmegaConf configuration.

## Common Commands

### Linting & Formatting
```bash
make check       # black + ruff check (read-only)
make autoformat  # black + ruff with auto-fix
```
Black/Ruff: 121 char line length, py310 target.

### Smoke Testing
No pytest. Each submodule is self-testable by running it directly:
```bash
python starVLA/model/framework/QwenGR00T.py
python starVLA/dataloader/lerobot_datasets.py --config_yaml starvla_cotrain_oxe.yaml
```

### Training
```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ./path/to/config.yaml
```
Other entry points: `train_starvla_cotrain.py` (VLA+VLM co-training), `train_starvlm.py` (VLM-only).
CLI overrides work on any config key: `--framework.qwenvl.base_vlm Qwen/Qwen2.5-VL-7B-Instruct`

## Architecture

### `starVLA/model/framework/` — VLA Frameworks
- `base_framework.py`: Base class (extends HuggingFace `PreTrainedModel`) with pretrained loading, action normalization, trainable module discovery
- Implementations: `QwenGR00T.py`, `QwenOFT.py`, `QwenFast.py`, `QwenPI.py`, `QwenAdapter.py`, `QwenOFT_multiRobo.py`
- `__init__.py`: `build_framework(cfg)` factory. `QwenOFT`/`QwenFast` are hardcoded; others use `FRAMEWORK_REGISTRY` auto-registration (`@FRAMEWORK_REGISTRY.register("Name")`)

### `starVLA/model/modules/` — Reusable Components
- `vlm/`: Qwen-VL wrappers (QWen2_5, QWen3)
- `action_model/`: Action heads — DiT (`GR00T_ActionHeader`), Flow Matching (`LayerwiseFM_ActionHeader`), MLP (`MLP_ActionHeader`), FAST tokenizer (`fast_ActionHeader`), Adapter (`VLA_AdapterHeader`)
- `dino_model/`: DINOv2 vision encoders
- `projector/`: Feature projection layers

### `starVLA/dataloader/` — Data Pipeline
- `lerobot_datasets.py`: Main VLA dataset (LeRobot format)
- `vlm_datasets.py`: VLM pretraining data
- `gr00t_lerobot/`: GR00T pipeline — `mixtures.py` (dataset mixtures), `data_config.py` (robot schemas), `embodiment_tags.py` (robot type tags)

### `starVLA/training/` — Training Loop
- `train_starvla.py`: Primary VLA training (native PyTorch + Accelerate + DeepSpeed)
- `train_starvla_cotrain.py`: Joint VLA + VLM co-training
- `train_starvlm.py`: VLM-only pretraining
- `trainer_utils/`: `config_tracker.py` (`AccessTrackedConfig`), `trainer_tools.py` (freezing, per-module LR groups, grad ops)

### `starVLA/config/` — Configuration
- `training/`: YAML configs (`starvla_case.yaml`, `starvla_cotrain_case.yaml`, etc.)
- `deepseeds/`: DeepSpeed configs (ZeRO-2, ZeRO-3)

### Config Structure (OmegaConf YAML)
```yaml
framework:
  name: QwenGR00T          # selects which framework to build
  qwenvl: { ... }          # VLM backbone config
  action_model: { ... }    # action head config
datasets:
  vla_data: { ... }        # VLA robot data
  vlm_data: { ... }        # optional VLM data for co-training
trainer:
  learning_rate:
    base: 2.5e-05           # per-module LRs supported
    action_model: 1.0e-04
```

### Key Design Contracts
- **Dataloader output**: Raw model-agnostic dict `{image: list[PIL.Image], lang: str, action: ndarray[T, D], state: Optional[ndarray]}`. No model-specific preprocessing in the dataloader.
- **Framework API**: `framework.forward()` for training loss, `framework.predict_action()` for inference. Both accept raw inputs.
- **Action normalization**: Models output normalized actions [-1, 1]. Unnormalize via `dataset_statistics.json` at eval time.

### Git Conventions
Commit messages: `[type] description` — e.g., `[train]`, `[framework]`, `[bench]`, `[feat]`, `[fix]`, `[chore]`

## DLC (Cloud Platform) Constraints
- DLC 环境**不可以**使用 `conda activate`，环境已预装好
- DLC 环境**不可以**设置 `ulimit -n 1048576`，没有权限
- DLC 脚本中不要添加这两项，只在本地脚本中使用

## SFT RoboTwin Constraints
- SFT RoboTwin 的训练脚本**只在本地运行**，不使用 DLC
- 本地脚本需要 `ulimit -n 1048576`、`conda activate starVLA`、`--num_processes 8`
- 不需要 DLC 的 `$MASTER_ADDR`/`$MASTER_PORT`/`$RANK`/`$WORLD_SIZE` 环境变量
- SFT 通过 `--trainer.pretrained_checkpoint` 加载预训练权重，`is_resume: false`（不恢复 optimizer 状态）
