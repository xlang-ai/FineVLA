# Repository Guidelines

## Project Structure & Module Organization
`starVLA/` contains the Python package. Core areas are `model/` for frameworks and reusable modules, `dataloader/` for dataset pipelines, `training/` for training entry points, and `config/training/` for OmegaConf YAML configs. `examples/` holds benchmark-specific train and eval scripts such as `examples/LIBERO/` and `examples/Robotwin/`. `deployment/` contains policy-server utilities, `scripts/` contains operational helpers, and `assets/` stores documentation images. Treat `playground/`, `results/`, and `*.log` files as local artifacts, not source.

## Build, Test, and Development Commands
Set up the environment with `pip install -r requirements.txt` and `pip install -e .`.

- `make check` runs `black --check` and `ruff check` without modifying files.
- `make autoformat` applies Black and Ruff fixes in place.
- `python starVLA/model/framework/QwenGR00T.py` is the standard framework smoke test.
- `accelerate launch --num_processes 8 starVLA/training/train_starvla.py --config_yaml starVLA/config/training/starvla_case.yaml` starts baseline VLA training.
- `bash examples/LIBERO/eval_files/eval_libero.sh` runs a benchmark-specific evaluation after its environment variables are configured.

## Coding Style & Naming Conventions
Python 3.10 is the target. Use 4-space indentation, keep lines within 121 characters, and let Black and Ruff define formatting. Follow existing naming patterns: `snake_case` for functions, variables, YAML files, and scripts; `PascalCase` for framework classes such as `QwenGR00T`; and descriptive module names grouped by capability. Keep dataloaders model-agnostic and keep framework-specific logic inside `starVLA/model/framework/`.

## Testing Guidelines
There is no dedicated `pytest` suite in this repository. Validate changes with focused smoke tests by running the affected module directly and, for training changes, launching the smallest relevant config. When adding a new example or config, include a runnable command in the related README or script header. Record any required checkpoints, dataset paths, or environment variables explicitly.

## Commit & Pull Request Guidelines
Recent history uses short, scoped subjects such as `[train] ...`, `[framework] ...`, and `[feat] ...`. Keep commits focused and use the same bracketed prefix style. Pull requests should explain the benchmark or subsystem affected, list the commands used for verification, link any relevant issue or experiment note, and include logs or screenshots when changing evaluation or deployment behavior.
