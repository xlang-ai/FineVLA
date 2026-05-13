# Any4LeRobot 环境安装指导

本文档说明如何创建并使用名为 **any4lerobot** 的 Conda 环境，以运行本项目代码。

---

## 一、环境与 Python 版本

- **环境名称**：`any4lerobot`
- **Python**：建议 **3.10** 或 **3.11**（与 [LeRobot](https://github.com/huggingface/lerobot) 兼容）
- **包管理**：Conda + pip

---

## 二、步骤 1：创建 Conda 环境

在终端中执行：

```bash
# 创建环境 any4lerobot，Python 3.10
conda create -n any4lerobot python=3.10 -y

# 激活环境
conda activate any4lerobot
```

若希望使用 Python 3.11：

```bash
conda create -n any4lerobot python=3.11 -y
conda activate any4lerobot
```

---

## 三、步骤 2：安装 LeRobot（核心依赖）

Any4LeRobot 依赖 [Hugging Face LeRobot](https://github.com/huggingface/lerobot)，需先安装：

```bash
conda activate any4lerobot
pip install lerobot
```

安装后可检查：

```bash
lerobot-info
```

---

## 四、步骤 3：安装本项目依赖

进入项目根目录后，用 `requirements.txt` 安装通用依赖：

```bash
cd /cpfs04/shared/Group-m6/tongzai.hxt/Qwen_VLA/any4lerobot
conda activate any4lerobot
pip install -r requirements.txt
```

若只做「核心功能」（数据合并、RoboMIND/AgiBot 转换等），可不用装 TensorFlow 等；若用到 **OpenX→LeRobot** 或 **LeRobot→RLDS**，需再装 TF 相关包（见下文「按需安装」）。

---

## 五、步骤 4：按需安装（取决于你用哪些子模块）

| 子模块 | 用途 | 额外安装 |
|--------|------|----------|
| **robomind2lerobot** / **agibot2lerobot** | RoboMIND / AgiBot → LeRobot | 已包含在 `requirements.txt`（h5py、ray） |
| **openx2lerobot** | Open X-Embodiment → LeRobot | `pip install tensorflow tensorflow-datasets` |
| **lerobot2rlds** | LeRobot → RLDS | `pip install tensorflow tensorflow-datasets apache-beam` |
| **libero2lerobot** | LIBERO → LeRobot | `pip install -U datatrove`，并参考 [LIBERO 安装](https://github.com/Lifelong-Robot-Learning/LIBERO#installation) |
| **dataset_merging** | 合并多个 LeRobot 数据集 | 无需额外包（numpy、pandas 已在 requirements 中） |
| **ds_version_convert** | 数据集版本转换 (v1.6↔v2.0↔v2.1↔v3.0) | 一般只需 lerobot；部分脚本需 `datasets`、`huggingface_hub` 等，缺啥再 `pip install` 即可 |

**示例：若要用 openx2lerobot 和 lerobot2rlds**

```bash
conda activate any4lerobot
pip install tensorflow tensorflow-datasets apache-beam
```

---

## 六、步骤 5：设置项目路径与运行方式

运行各子模块时，需在对应子目录下执行，以便正确解析相对导入（如 `robomind_uitls`、`agibot_utils` 等）。

**示例：RoboMIND → LeRobot**

```bash
conda activate any4lerobot
cd /cpfs04/shared/Group-m6/tongzai.hxt/Qwen_VLA/any4lerobot/robomind2lerobot

# 按需设置环境变量（推荐）
export HDF5_USE_FILE_LOCKING=FALSE
export RAY_DEDUP_LOGS=0

# 按你的路径修改 convert.sh 中的 --src-path / --output-path 后执行
bash convert.sh
```

**示例：数据集合并**

```bash
conda activate any4lerobot
cd /cpfs04/shared/Group-m6/tongzai.hxt/Qwen_VLA/any4lerobot/dataset_merging
python merge_lerobot_dataset.py --sources /path/to/ds1 /path/to/ds2 --output /path/to/merged
```

---

## 七、一键安装脚本示例（仅作参考）

若希望从零一条龙创建并安装「核心 + OpenX/RLDS」支持，可保存为 `setup_any4lerobot.sh` 后执行：

```bash
#!/bin/bash
set -e
ENV_NAME=any4lerobot
PROJECT_ROOT=/cpfs04/shared/Group-m6/tongzai.hxt/Qwen_VLA/any4lerobot

# 1. 创建并激活环境
conda create -n $ENV_NAME python=3.10 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# 2. 安装 LeRobot
pip install lerobot

# 3. 安装本项目 requirements
cd "$PROJECT_ROOT"
pip install -r requirements.txt

# 4. 若需 OpenX / LeRobot2RLDS，取消下面两行注释：
# pip install tensorflow tensorflow-datasets apache-beam

echo "Environment '$ENV_NAME' ready. Run: conda activate $ENV_NAME"
```

使用方式：

```bash
chmod +x setup_any4lerobot.sh
./setup_any4lerobot.sh
```

---

## 八、常见问题

1. **`lerobot-info` 报错或找不到**  
   确认已 `conda activate any4lerobot` 且在该环境中执行 `pip install lerobot`。

2. **导入报错 `No module named 'robomind_uitls'` 等**  
   需要在对应子目录下运行脚本（例如在 `robomind2lerobot/` 下执行 `convert.sh`），不要从 any4lerobot 根目录直接跑子目录里的 `.py`。

3. **RoboMIND 转换时 HDF5 锁或 Ray 相关问题**  
   在运行前执行：  
   `export HDF5_USE_FILE_LOCKING=FALSE`、`export RAY_DEDUP_LOGS=0`，或在 `convert.sh` 里写上这两行。

4. **只想用 dataset_merging**  
   最少需要：创建 `any4lerobot` 环境 → 安装 `lerobot` → `pip install numpy pandas termcolor`；若已按本文安装过 `requirements.txt`，则无需再装。

---

## 九、参考链接

- [LeRobot 官方安装说明](https://huggingface.co/docs/lerobot/installation)
- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- 各子模块用法见对应目录下的 `README.md`（如 `robomind2lerobot/README.md`、`openx2lerobot/README.md` 等）
