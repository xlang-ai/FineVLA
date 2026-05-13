#!/usr/bin/env python3
"""
根据 error.json 中的失败数据集，从 episodes.jsonl 收集 tasks 并重写 tasks.jsonl。

1. 读取 error.json，获取所有 dataset_name
2. 对每个 dataset：从 episodes.jsonl 按 episode 顺序收集 tasks，保持有序去重
3. 将收集到的 tasks 写入 tasks.jsonl，格式为 {"task_index": i, "task": task_str}
"""

import json
from pathlib import Path

# --- Config ---
ROBOCOIN_ROOT = Path("/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21/RoboCOIN_add1201")
ERROR_JSON = ROBOCOIN_ROOT / "RoboCOIN_add1201_mode2_20260130_123156_error.json"


def load_error_datasets(error_path: Path) -> list[str]:
    """从 error.json 的 error 字段中提取所有 dataset_name（键）"""
    with open(error_path) as f:
        data = json.load(f)
    return list(data.get("error", {}).keys())


def collect_tasks_ordered(episodes_path: Path) -> list[str]:
    """
    按 episode 顺序从 episodes.jsonl 收集 tasks，保持有序去重。
    前面的 episode 的 tasks 先加入，后续 episode 的 tasks 若未出现过则追加。
    """
    collected = []  # 有序列表
    seen = set()    # 用于去重

    with open(episodes_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            tasks = ep.get("tasks", [])
            if not isinstance(tasks, list):
                tasks = [tasks] if tasks else []
            for t in tasks:
                t_str = t if isinstance(t, str) else str(t)
                if t_str not in seen:
                    seen.add(t_str)
                    collected.append(t_str)

    return collected


def write_tasks_jsonl(tasks: list[str], output_path: Path):
    """写入 tasks.jsonl"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, task in enumerate(tasks):
            obj = {"task_index": i, "task": task}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    if not ERROR_JSON.exists():
        print(f"Error: {ERROR_JSON} not found")
        return

    datasets = load_error_datasets(ERROR_JSON)
    print(f"Found {len(datasets)} datasets from error.json")

    for ds_name in datasets:
        ds_path = ROBOCOIN_ROOT / ds_name
        episodes_path = ds_path / "meta" / "episodes.jsonl"
        tasks_path = ds_path / "meta" / "tasks.jsonl"

        if not episodes_path.exists():
            print(f"  Skip {ds_name}: episodes.jsonl not found")
            continue

        collected = collect_tasks_ordered(episodes_path)
        if not collected:
            print(f"  Skip {ds_name}: no tasks in episodes.jsonl")
            continue

        write_tasks_jsonl(collected, tasks_path)
        print(f"  Fixed {ds_name}: {len(collected)} tasks -> {tasks_path}")

    print("Done.")


if __name__ == "__main__":
    main()
