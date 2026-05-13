# -*- coding: utf-8 -*-
"""
修复 episodes.jsonl 中 tasks 为空的问题。

原因：部分子数据集的 episodes.jsonl 中 tasks 字段为空列表 []，
但 parquet 文件中有 task_index 字段，且 meta/tasks.jsonl 中有完整的 task 定义。

修复逻辑：
1. 读取 meta/tasks.jsonl 构建 task_index -> task 映射
2. 只读第一个 parquet 的 task_index 列（同一子数据集内所有 episode 共享同一 task_index）
3. 通过 task_index 查找 task 文本
4. 更新 episodes.jsonl 中的 tasks 字段

用法:
    # 修复单个子数据集
    python fix_episodes_tasks.py /path/to/subdataset

    # 批量修复 RoboCOIN_add1201 下所有 task 为空的子数据集
    python fix_episodes_tasks.py /path/to/RoboCOIN_add1201 --batch

    # 预览（不写入）
    python fix_episodes_tasks.py /path/to/RoboCOIN_add1201 --batch --dry-run
"""

import argparse
import glob
import json
import os
import sys

import pyarrow.parquet as pq


def load_task_map(dataset_path: str) -> dict:
    """加载 tasks.jsonl，返回 {task_index: task_string}。"""
    tasks_file = os.path.join(dataset_path, "meta", "tasks.jsonl")
    task_map = {}
    if not os.path.isfile(tasks_file):
        return task_map
    with open(tasks_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            task_map[obj["task_index"]] = obj["task"]
    return task_map


def get_shared_task_index(dataset_path: str) -> list:
    """只读第一个 parquet 的 task_index 列，返回 [task_index, ...]。
    同一子数据集内所有 episode 通常共享同一 task_index，只需读一个文件。
    """
    parquets = sorted(glob.glob(
        os.path.join(dataset_path, "data", "**", "episode_*.parquet"),
        recursive=True))
    if not parquets:
        return []
    try:
        df = pq.read_table(parquets[0], columns=["task_index"]).to_pandas()
        return sorted(df["task_index"].unique().tolist())
    except Exception as e:
        print(f"  [WARN] 无法读取 {parquets[0]}: {e}")
        return []


def fix_single_dataset(dataset_path: str, dry_run: bool = False) -> int:
    """修复单个数据集的 episodes.jsonl，返回修复的 episode 数量。"""
    name = os.path.basename(dataset_path)
    ep_file = os.path.join(dataset_path, "meta", "episodes.jsonl")

    if not os.path.isfile(ep_file):
        print(f"  [SKIP] {name}: 无 episodes.jsonl")
        return 0

    # 读取现有 episodes.jsonl
    episodes = []
    with open(ep_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))

    # 检查是否有需要修复的 episode
    needs_fix = [ep for ep in episodes
                 if not ep.get("tasks") or (isinstance(ep.get("tasks"), list) and
                     (len(ep["tasks"]) == 0 or all(t.strip() == "" for t in ep["tasks"] if isinstance(t, str))))]

    if not needs_fix:
        print(f"  [OK] {name}: 所有 episode 的 tasks 都非空，无需修复")
        return 0

    # 加载 task_map
    task_map = load_task_map(dataset_path)
    if not task_map:
        print(f"  [SKIP] {name}: 无 tasks.jsonl 或内容为空，无法修复")
        return 0

    # 从第一个 parquet 提取 task_index（同一子数据集共享）
    task_idx_list = get_shared_task_index(dataset_path)
    if not task_idx_list:
        print(f"  [SKIP] {name}: parquet 中无 task_index 信息")
        return 0

    # 解析 task 文本
    resolved_tasks = []
    for ti in task_idx_list:
        if ti in task_map:
            resolved_tasks.append(task_map[ti])
        else:
            print(f"  [WARN] {name}: task_index={ti} 在 tasks.jsonl 中未找到")
    if not resolved_tasks:
        print(f"  [SKIP] {name}: task_index 在 tasks.jsonl 中均未找到")
        return 0

    # 修复
    n_fixed = 0
    for ep in episodes:
        tasks = ep.get("tasks")
        is_empty = (not tasks or
                    (isinstance(tasks, list) and
                     (len(tasks) == 0 or all(t.strip() == "" for t in tasks if isinstance(t, str)))))

        if not is_empty:
            continue

        ep["tasks"] = resolved_tasks
        n_fixed += 1

    if n_fixed == 0:
        print(f"  [SKIP] {name}: 无法从 parquet 解析到有效 task")
        return 0

    # 写回
    if dry_run:
        print(f"  [DRY-RUN] {name}: 将修复 {n_fixed}/{len(episodes)} 个 episode")
        for ep in episodes:
            if ep["episode_index"] < 3:
                print(f"    ep {ep['episode_index']}: tasks={ep.get('tasks')}")
    else:
        # 备份原文件
        backup_path = ep_file + ".bak"
        if not os.path.exists(backup_path):
            import shutil
            shutil.copy2(ep_file, backup_path)

        with open(ep_file, "w", encoding="utf-8") as f:
            for ep in episodes:
                f.write(json.dumps(ep, ensure_ascii=False) + "\n")
        print(f"  [FIXED] {name}: 修复 {n_fixed}/{len(episodes)} 个 episode (备份: {backup_path})")

    return n_fixed


def discover_empty_task_datasets(root_path: str) -> list:
    """查找 root_path 下所有 episodes.jsonl 中 tasks 为空的子数据集。"""
    results = []
    for entry in sorted(os.listdir(root_path)):
        full = os.path.join(root_path, entry)
        if not os.path.isdir(full):
            continue
        ep_file = os.path.join(full, "meta", "episodes.jsonl")
        if not os.path.isfile(ep_file):
            continue
        with open(ep_file, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if not first_line:
                continue
            obj = json.loads(first_line)
            tasks = obj.get("tasks")
            if not tasks or (isinstance(tasks, list) and
                (len(tasks) == 0 or all(t.strip() == "" for t in tasks if isinstance(t, str)))):
                results.append(full)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="修复 episodes.jsonl 中 tasks 为空的问题，从 parquet task_index + tasks.jsonl 恢复。")
    parser.add_argument("path", help="数据集路径（单个子数据集）或父目录（配合 --batch）")
    parser.add_argument("--batch", action="store_true",
                        help="批量模式：扫描 path 下所有 tasks 为空的子数据集并修复")
    parser.add_argument("--dry-run", action="store_true",
                        help="预览模式：只打印将要修复的内容，不写入文件")
    args = parser.parse_args()

    path = args.path.rstrip("/")

    if args.batch:
        datasets = discover_empty_task_datasets(path)
        print(f"发现 {len(datasets)} 个 tasks 为空的子数据集")
        total_fixed = 0
        for ds in datasets:
            n = fix_single_dataset(ds, dry_run=args.dry_run)
            total_fixed += n
        print(f"\n总计修复 {total_fixed} 个 episode" + (" (dry-run)" if args.dry_run else ""))
    else:
        fix_single_dataset(path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
