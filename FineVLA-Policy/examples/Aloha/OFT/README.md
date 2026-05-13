# Aloha OFT Training Notes

This directory contains the OFT training entry points for the Aloha multi-dataset setup.

## Current Training Setup

The OFT scripts in this directory are intended to follow the setup below:

1. Base model: `Qwen3.5-4B`
2. Framework: `QwenOFT`
3. Configured action head type: `DiT-B`
4. Input sequence handling: `max_seq_length` is set to `1024` for the local OFT test entry
5. Learning-rate scheduler: `cosine_with_min_lr`
6. Training data: mixed multi-dataset training via `aloha_multi_mix` or its FG-weighted variants
7. Action/state normalization: `q01/q99` for continuous joint dimensions, `binary` for gripper dimensions

## What This Means

- Training starts from the base VLM path `/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/Pretrained_models/Qwen3.5-4B`
- OFT checkpoints and logs are saved under `/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/StarVLA_YJH/results/Checkpoints/OFT_ALOHA/`
- OFT scripts in this folder are not intended to resume from an older OFT checkpoint unless explicitly modified
- `action_model_type` is currently set to `DiT-B` in the config for consistency with the RawRDT-style OFT setup. In the current `QwenOFT` implementation, this field does not switch to a different action-head implementation; it behaves the same as the earlier `MLPResNet` label.
- For Qwen3.5 input construction, short sequences can be padded to a fixed token length while longer sequences keep their real length without truncation
- `aloha_multi_mix` is a multi-source dataset mixture defined in `starVLA/dataloader/gr00t_lerobot/mixtures.py`

## Current Local Test Entry

The main local smoke-test entry is:

```bash
bash examples/Aloha/OFT/run_qwen35_OFT_aloha_multi_test_local.sh
```

That script currently overrides:

```bash
--framework.qwenvl.max_seq_length 1024
```

## Normalization Note

For the Aloha pipeline, normalization mode is controlled by the robot data config in `starVLA/dataloader/gr00t_lerobot/data_config.py`, not by the training YAML alone. In the Aloha-related configs used by this setup, continuous joint values use `q99` normalization and gripper values use `binary`.
