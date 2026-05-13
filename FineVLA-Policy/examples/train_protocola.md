# StarVLA 新 Benchmark 训练 / 测试协议（Protocol A）

这个文档对应你提到的思路：
- 参考 `examples/LIBERO/train_files/bar/run_libero_train.sh`
- 抽象出「新 benchmark 接入」时训练与测试分别要做什么

并且已落地到 RoboCasa365。

---

## 一、先理解 train script 在做什么（以 LIBERO 为模板）

`run_libero_train.sh` 本质在做 4 件事：

1. **环境层设置**
   - NCCL / 分布式参数
   - 路径参数（`base_vlm`, `config_yaml`, `data_root`, `run_root_dir`）

2. **实验命名与产物目录**
   - `run_id`、`output_dir`
   - 把脚本自身 copy 到输出目录，保证可追溯

3. **训练入口调用**
   - `accelerate launch starVLA/training/train_starvla.py`
   - 用 CLI 覆盖配置中的关键字段（数据、batch、步数、保存频率等）

4. **日志/追踪**
   - wandb project/entity
   - 保存 checkpoint + dataset_statistics（后续反归一化动作要用）

---

## 二、如果要接入一个新的 benchmark，最小闭环清单

### A. 训练侧
1. 数据目录能被 `data_root_dir/data_name` 正确找到
2. 在 `mixtures.py` 注册 `data_mix`
3. 在 `data_config.py` 注册该机器人/数据的键映射（video/state/action/language）
4. 在 `embodiment_tags.py` 给 robot_type 分配合法 tag（至少可落到 `new_embodiment`）
5. 新建 benchmark 的训练 yaml + run shell

### B. 评测侧
1. 启动 websocket policy server（starVLA 环境）
2. 仿真环境里把 observation 转成 `examples=[{"image", "lang", ...}]`
3. 把模型输出的 normalized action 用 `dataset_statistics.json` 反归一化
4. 把连续动作切分成 env 需要的 action dict 字段
5. 记录成功率、视频和日志

---

## 三、RoboCasa365 已集成内容

### 代码层
- `starVLA/dataloader/gr00t_lerobot/mixtures.py`
  - 新增 `robocasa365_single`
- `starVLA/dataloader/gr00t_lerobot/data_config.py`
  - 新增 `Robocasa365DataConfig`
- `starVLA/dataloader/gr00t_lerobot/embodiment_tags.py`
  - 新增 `robocasa365_panda_omron -> new_embodiment`

### 示例层
- 训练：
  - `examples/Robocasa365/train_files/starvla_train_robocasa365.yaml`
  - `examples/Robocasa365/train_files/run_robocasa365_train.sh`
- 评测：
  - `examples/Robocasa365/eval_files/run_policy_server.sh`
  - `examples/Robocasa365/eval_files/model2robocasa365_interface.py`
  - `examples/Robocasa365/eval_files/simulation_env.py`
  - `examples/Robocasa365/eval_files/eval_robocasa365.sh`
- 说明：
  - `examples/Robocasa365/README.md`

---

## 四、你现在可以直接做的两条命令链

### 训练
1. 改好 `run_robocasa365_train.sh` 里的路径
2. 运行：`bash examples/Robocasa365/train_files/run_robocasa365_train.sh`

### 评测
1. 终端1（starVLA 环境）：`bash examples/Robocasa365/eval_files/run_policy_server.sh`
2. 终端2（robocasa 环境）：`bash examples/Robocasa365/eval_files/eval_robocasa365.sh`

---

## 五、注意点（实战里最容易卡住）

1. `unnorm_key`
   - 当前默认是 `new_embodiment`（来自 dataloader tag）
   - 如果你后续改了 tag，要同步改评测脚本

2. 动作维度
   - RoboCasa365 这里按 12 维切分
   - 如果你改了数据 schema，训练配置和评测切分都要同步

3. 任务名
   - `env_name` 需与 robocasa 安装版本一致
   - 建议先用一个任务 smoke test，再跑批量
