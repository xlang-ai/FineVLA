# LeRobot v2.1 Dataset Visualizer

本地 Web 工具，用于可视化 LeRobot v2.1 格式的机器人数据集。

## 功能

- 输入 parquet 文件路径，自动解析数据集信息
- 多视角视频同步播放
- 逐帧 Task 标注显示
- State / Action 时序曲线图（Y 轴范围来自 episodes_stats）
- 帧滑块与视频/曲线联动
- 支持多种数据集（Galaxea、RoboCOIN、RoboMind 等）

## 启动

```bash
cd backend
pip install -r requirements.txt
python main.py
```

然后浏览器打开 http://localhost:8765

## 使用

在输入框中粘贴 parquet 文件的绝对路径，例如：

```
/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21/Galaxea-Open-World-Dataset/Adjust_The_Air_Conditioner_Temperature_20250711_006/data/chunk-000/episode_000000.parquet
```

点击 Load 即可。

## 添加新数据集

编辑 `backend/dataset_config.py`，在 `DATASET_CONFIGS` 中新增条目，指定 `match_keyword` 和对应的 state/action 字段。
如果不配置，系统会自动从 `info.json` 中发现字段（fallback）。
