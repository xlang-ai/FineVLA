# Training Script Comparison

This table summarizes the training scripts and configs that have been checked in this workspace.

| Script / Config | Base VLM | Framework | Action Head Config Label | Data Mix | Data Scope | Normalization | Max Seq Length | Save Root | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `examples/Aloha/OFT/run_qwen35_OFT_aloha_multi_test_local.sh` | `Qwen3.5-4B` | `QwenOFT` | `DiT-B` | `aloha_multi_mix` | Full Aloha mixed dataset | `q01/q99` for joints, `binary` for gripper | `1024` | `results/Checkpoints/OFT_ALOHA` | Local smoke-test entry |
| `examples/Aloha/OFT/run_qwen35_OFT_aloha_multi_dlc.sh` | `Qwen3.5-4B` | `QwenOFT` | `DiT-B` | `aloha_multi_mix` | Full Aloha mixed dataset | `q01/q99` for joints, `binary` for gripper | not set in script | `results/Checkpoints/OFT_ALOHA` | Main RawAloha OFT DLC entry |
| `examples/Aloha/OFT/run_qwen35_OFT_aloha_multi_FGOnly_dlc.sh` | `Qwen3.5-4B` | `QwenOFT` | `DiT-B` | `aloha_multi_FGOnly` | FG-only Aloha episodes | `q01/q99` for joints, `binary` for gripper | `1024` | `results/Checkpoints/OFT_ALOHA` | Uses only `_fg` dataset branch |
| `examples/Aloha/OFT/run_qwen35_OFT_aloha_multi_FG1_1_test_local.sh` | `Qwen3.5-4B` | `QwenOFT` | `DiT-B` | `aloha_multi_FG1_1` | FG + normal mixed, 1:1 | `q01/q99` for joints, `binary` for gripper | `1024` | `results/Checkpoints/OFT_ALOHA` | Local FG 1:1 smoke-test entry |
| `examples/Aloha/OFT/run_qwen35_OFT_aloha_multi_FG1_1_dlc.sh` | `Qwen3.5-4B` | `QwenOFT` | `DiT-B` | `aloha_multi_FG1_1` | FG + normal mixed, 1:1 | `q01/q99` for joints, `binary` for gripper | `1024` | `results/Checkpoints/OFT_ALOHA` | FG branch is episode-level; raw branch still present |
| `examples/Aloha/OFT/run_qwen35_OFT_aloha_multi_FG2_1_dlc.sh` | `Qwen3.5-4B` | `QwenOFT` | `DiT-B` | `aloha_multi_FG2_1` | FG + normal mixed, 2:1 | `q01/q99` for joints, `binary` for gripper | `1024` | `results/Checkpoints/OFT_ALOHA` | FG branch weighted x2 |
| `examples/Aloha/OFT/run_qwen35_OFT_aloha_multi_FG4_1_dlc.sh` | `Qwen3.5-4B` | `QwenOFT` | `DiT-B` | `aloha_multi_FG4_1` | FG + normal mixed, 4:1 | `q01/q99` for joints, `binary` for gripper | `1024` | `results/Checkpoints/OFT_ALOHA` | FG branch weighted x4 |
| `examples/Robotwin/RoboTwinMix/run_qwen35_OFT_robotwinMix_baseline128_local.sh` | `Qwen3.5-4B` | `QwenOFT` | `DiT-B` | `robotwin_mix` | RoboTwin Clean + Randomized | `q01/q99` for joints, `binary` for gripper | not set in script | `results/Checkpoints` | Uses `lerobot_v21_robotwin`; `wandb_project` name currently looks like an Aloha carry-over |
| `Qwen-StarVLA/DLCShell/qwen3.5_OFT_RawRDT.sh` | `Qwen3.5-4B` | `QwenOFT` | `DiT-B` | `rdt_yhq` | Raw RDT-yhq | determined by `lerobot_v21_aloha` config | not set in script | `results/Checkpoints` | Reference script from external repo |

## Notes

- `action_model_type: DiT-B` in current OFT configs is a config label only. In the current `QwenOFT` implementation, the action head still goes through `MLP_ActionHeader`, so the effective behavior is the same as the earlier `MLPResNet` label unless code is changed.
- In the current `StarVLA_YJH` repo, `Qwen3.5` input handling supports fixed padding for short sequences while keeping longer sequences untruncated when `framework.qwenvl.max_seq_length` is set.
