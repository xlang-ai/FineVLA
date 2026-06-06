# LeRobot v2.1 Dataset Visualizer

A local web tool for visualizing robotics datasets in LeRobot v2.1 format.

## Features

- Enter a parquet file path to automatically parse dataset information
- Multi-view video synchronized playback
- Per-frame task annotation display
- State / Action time-series charts (Y-axis range from episodes_stats)
- Frame slider linked with video and charts
- Supports multiple datasets (Galaxea, RoboCOIN, RoboMind, etc.)

## Getting Started

```bash
cd backend
pip install -r requirements.txt
python main.py
```

Then open http://localhost:8765 in your browser.

## Usage

Paste the absolute path of a parquet file into the input box, for example:

```
$VLA_DATA_ROOT/Galaxea-Open-World-Dataset/Adjust_The_Air_Conditioner_Temperature_20250711_006/data/chunk-000/episode_000000.parquet
```

Click Load to proceed.

## Adding New Datasets

Edit `backend/dataset_config.py` and add a new entry in `DATASET_CONFIGS`, specifying `match_keyword` and the corresponding state/action fields.
If not configured, the system will automatically discover fields from `info.json` (fallback).
