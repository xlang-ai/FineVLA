# RoboCasa365 训练与评测（StarVLA）

本目录集成了基于 `playground/Datasets/robocasa365` 的训练与仿真评测流程。

## 1) 训练

### 数据
你当前数据目录已经是：
- `playground/Datasets/robocasa365`

训练配置已对接：
- `data_root_dir=playground/Datasets`
- `data_mix=robocasa365_single`

### 启动训练
```bash
bash examples/Robocasa365/train_files/run_robocasa365_train.sh
```

关键配置文件：
- `examples/Robocasa365/train_files/starvla_train_robocasa365.yaml`

默认框架是 `QwenOFT`，动作维度配置为 12（对齐 RoboCasa365 数据 `meta/modality.json`）。

---

## 2) 评测

评测依赖 RoboCasa 环境（建议按官方仓库安装）：
- https://github.com/robocasa/robocasa

### Step A: 启动策略服务（starVLA 环境）
```bash
bash examples/Robocasa365/eval_files/run_policy_server.sh
```

### Step B: 启动仿真评测（robocasa 环境）
```bash
bash examples/Robocasa365/eval_files/eval_robocasa365.sh
```

默认示例任务：
- `robocasa/PickPlaceCounterToCabinet`

你可以在 `eval_robocasa365.sh` 里替换 `env_name` 为其他 RoboCasa365 任务名。

---

## 3) 本次集成改动概览

- 新增数据混合项：`robocasa365_single`
- 新增数据 schema：`robocasa365_panda_omron`
- 新增训练脚本与配置
- 新增 RoboCasa365 仿真评测适配器（websocket policy -> action dict）

如果后续你希望我继续做「365任务批量评测脚本 + 汇总成功率表格」，我可以直接在这个目录再补一个 `batch_eval_robocasa365.sh` 和结果聚合脚本。
