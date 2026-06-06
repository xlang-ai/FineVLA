# StarVLA New Benchmark Training / Testing Protocol (Protocol A)

This document corresponds to the approach you mentioned:
- Refer to `examples/LIBERO/train_files/bar/run_libero_train.sh`
- Abstract what needs to be done for training and testing when integrating a new benchmark

This has already been implemented for RoboCasa365.

---

## 1. Understanding What the Train Script Does (Using LIBERO as a Template)

`run_libero_train.sh` essentially does 4 things:

1. **Environment-level setup**
   - NCCL / distributed training parameters
   - Path parameters (`base_vlm`, `config_yaml`, `data_root`, `run_root_dir`)

2. **Experiment naming and output directory**
   - `run_id`, `output_dir`
   - Copy the script itself to the output directory for traceability

3. **Training entry point invocation**
   - `accelerate launch starVLA/training/train_starvla.py`
   - Override key config fields via CLI (data, batch size, steps, save frequency, etc.)

4. **Logging / tracking**
   - wandb project/entity
   - Save checkpoint + dataset_statistics (needed for action unnormalization later)

---

## 2. Minimum Checklist for Integrating a New Benchmark

### A. Training Side
1. Data directory must be correctly located at `data_root_dir/data_name`
2. Register `data_mix` in `mixtures.py`
3. Register the robot/data key mapping (video/state/action/language) in `data_config.py`
4. Assign a valid tag for the robot_type in `embodiment_tags.py` (at minimum, use `new_embodiment`)
5. Create a benchmark-specific training yaml + run shell script

### B. Evaluation Side
1. Start the websocket policy server (starVLA environment)
2. In the simulation environment, convert observations to `examples=[{"image", "lang", ...}]`
3. Unnormalize the model's normalized action output using `dataset_statistics.json`
4. Split the continuous actions into the action dict fields required by the env
5. Record success rates, videos, and logs

---

## 3. RoboCasa365 Integration Contents

### Code Layer
- `starVLA/dataloader/gr00t_lerobot/mixtures.py`
  - Added `robocasa365_single`
- `starVLA/dataloader/gr00t_lerobot/data_config.py`
  - Added `Robocasa365DataConfig`
- `starVLA/dataloader/gr00t_lerobot/embodiment_tags.py`
  - Added `robocasa365_panda_omron -> new_embodiment`

### Examples Layer
- Training:
  - `examples/Robocasa365/train_files/starvla_train_robocasa365.yaml`
  - `examples/Robocasa365/train_files/run_robocasa365_train.sh`
- Evaluation:
  - `examples/Robocasa365/eval_files/run_policy_server.sh`
  - `examples/Robocasa365/eval_files/model2robocasa365_interface.py`
  - `examples/Robocasa365/eval_files/simulation_env.py`
  - `examples/Robocasa365/eval_files/eval_robocasa365.sh`
- Documentation:
  - `examples/Robocasa365/README.md`

---

## 4. Two Command Chains You Can Run Right Now

### Training
1. Update the paths in `run_robocasa365_train.sh`
2. Run: `bash examples/Robocasa365/train_files/run_robocasa365_train.sh`

### Evaluation
1. Terminal 1 (starVLA environment): `bash examples/Robocasa365/eval_files/run_policy_server.sh`
2. Terminal 2 (robocasa environment): `bash examples/Robocasa365/eval_files/eval_robocasa365.sh`

---

## 5. Important Notes (Common Pitfalls in Practice)

1. `unnorm_key`
   - Currently defaults to `new_embodiment` (from the dataloader tag)
   - If you change the tag later, make sure to update the evaluation scripts accordingly

2. Action dimension
   - RoboCasa365 splits actions into 12 dimensions
   - If you change the data schema, both the training config and evaluation splitting must be updated in sync

3. Task name
   - `env_name` must match the installed version of robocasa
   - It is recommended to smoke test with a single task first, then run batch evaluation
