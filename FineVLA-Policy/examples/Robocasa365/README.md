# RoboCasa365 Training and Evaluation (StarVLA)

This directory integrates the training and simulation evaluation pipeline based on `playground/Datasets/robocasa365`.

## 1) Training

### Data
Your current data directory is:
- `playground/Datasets/robocasa365`

Training configuration is set up with:
- `data_root_dir=playground/Datasets`
- `data_mix=robocasa365_single`

### Start Training
```bash
bash examples/Robocasa365/train_files/run_robocasa365_train.sh
```

Key configuration files:
- `examples/Robocasa365/train_files/starvla_train_robocasa365.yaml`

The default framework is `QwenOFT`, with action dimension configured as 12 (aligned with RoboCasa365 data `meta/modality.json`).

---

## 2) Evaluation

Evaluation requires the RoboCasa environment (recommended to install following the official repository):
- https://github.com/robocasa/robocasa

### Step A: Start the policy server (starVLA environment)
```bash
bash examples/Robocasa365/eval_files/run_policy_server.sh
```

### Step B: Start the simulation evaluation (robocasa environment)
```bash
bash examples/Robocasa365/eval_files/eval_robocasa365.sh
```

Default example task:
- `robocasa/PickPlaceCounterToCabinet`

You can replace `env_name` in `eval_robocasa365.sh` with other RoboCasa365 task names.

---

## 3) Overview of Integration Changes

- Added data mixture: `robocasa365_single`
- Added data schema: `robocasa365_panda_omron`
- Added training scripts and configurations
- Added RoboCasa365 simulation evaluation adapter (websocket policy -> action dict)

If you would like us to further develop a batch evaluation script for all 365 tasks along with a success rate summary table, we can add a `batch_eval_robocasa365.sh` and a results aggregation script to this directory.
