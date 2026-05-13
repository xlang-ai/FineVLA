# Aloha Training Scripts

This directory now contains two parallel training entry sets for the Aloha multi-dataset:

- `examples/Aloha/`: GR00T-based training scripts
- `examples/Aloha/OFT/`: OFT-based training scripts

Both sets use the same dataset roots and `data_mix` naming, but they point to different framework/config combinations.

## Config Mapping

- GR00T uses `starVLA/config/training/starvla_aloha_multi_GR00T.yaml`
- OFT uses `starVLA/config/training/starvla_aloha_multi_OFT.yaml`

OFT is not just a renamed launch script. It switches `framework.name` from `QwenGR00T` to `QwenOFT` and uses an OFT-style action head config instead of the GR00T diffusion head.

## Recommended Entry Points

### GR00T

- Full Aloha local smoke test:
  `bash examples/Aloha/run_qwen35_GR00T_aloha_multi_test_local.sh`
- FG1_1 local smoke test:
  `bash examples/Aloha/run_qwen35_GR00T_aloha_multi_FG1_1_test_local.sh`
- Full Aloha DLC training:
  `bash examples/Aloha/run_qwen35_GR00T_aloha_multi_dlc.sh`

### OFT

- Full Aloha local smoke test:
  `bash examples/Aloha/OFT/run_qwen35_OFT_aloha_multi_test_local.sh`
- FG1_1 local smoke test:
  `bash examples/Aloha/OFT/run_qwen35_OFT_aloha_multi_FG1_1_test_local.sh`
- Full Aloha DLC training:
  `bash examples/Aloha/OFT/run_qwen35_OFT_aloha_multi_dlc.sh`

## Dataset Split Naming

The suffix in the script name maps directly to `datasets.vla_data.data_mix`:

- `aloha_multi_mix`: full multi-dataset
- `aloha_multi_FGOnly`
- `aloha_multi_FG1_1`
- `aloha_multi_FG2_1`
- `aloha_multi_FG4_1`

## Local vs DLC

- `*_test_local.sh`: local smoke tests with `conda activate starVLA` and reduced steps
- `*_dlc.sh`: DLC distributed training scripts using `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT`

## Notes

- Script paths were reorganized only for OFT. `PROJECT_ROOT` remains absolute, so moving the OFT shell scripts into `examples/Aloha/OFT/` does not change runtime behavior.
- If you want to convert an existing GR00T experiment to OFT, start from the matching script under `examples/Aloha/OFT/` rather than editing the GR00T files in place.
