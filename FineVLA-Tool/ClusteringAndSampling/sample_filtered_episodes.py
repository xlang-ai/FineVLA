#!/usr/bin/env python3
"""
从数据集中读取 *filter_report.json，排除有问题的 episode，
然后随机采样指定数量的 episode，生成与 cluster_representation_summary.json
相同格式的记录，并合并写入输出文件。

用法:
    # droid 数据集（普通），采样 1 万条
    python sample_filtered_episodes.py \
        --dataset_root $VLA_DATA_ROOT/droid_1.0.1 \
        --dataset_label droid_1.0.1 \
        --id_prefix droid \
        --process_type simple \
        --sample_n 10000 \
        --output cluster_representation_summary_droid.json

    # droid_RoboInter 数据集（提取 raw_steps），采样 1 万条
    python sample_filtered_episodes.py \
        --dataset_root $VLA_DATA_ROOT/droid_RoboInter \
        --dataset_label droid_robointer \
        --id_prefix droid_robointer \
        --process_type robointer \
        --sample_n 10000 \
        --output cluster_representation_summary_droid_robointer.json

    # 其它数据集同理，只需修改 --dataset_root 等参数
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

from utils.sample_all import (
    read_jsonl,
    extract_views,
    get_video_paths,
    get_resolution,
    make_record,
)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import pyarrow.parquet as pq_reader
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

import gc
import resource

def _raise_fd_limit():
    """尽可能提高进程的文件描述符上限。"""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            print(f"文件描述符上限: {soft} -> {hard}")
    except Exception:
        pass

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

SEED = 42


def find_filter_reports(dataset_root: str) -> list[str]:
    """在 dataset_root 下查找所有以 filter_report.json 结尾的文件（非递归）。"""
    results = []
    for fname in os.listdir(dataset_root):
        if fname.endswith("filter_report.json") and os.path.isfile(os.path.join(dataset_root, fname)):
            results.append(os.path.join(dataset_root, fname))
    return sorted(results)


def extract_problem_episodes(report_path: str) -> set[int]:
    """从 filter_report.json 中提取有问题的 episode index 集合。

    filter_report.json 格式:
      { "summary": {...}, "<dataset_name>": { "<episode_idx>": [...reasons], ... } }
    排除 "summary" key，其余 key 下的所有 episode index 都是有问题的。
    """
    with open(report_path, encoding="utf-8") as f:
        data = json.load(f)

    problem_indices = set()
    for key, val in data.items():
        if key == "summary":
            continue
        if isinstance(val, dict):
            for ep_idx_str in val.keys():
                try:
                    problem_indices.add(int(ep_idx_str))
                except ValueError:
                    pass
    return problem_indices


def _get_instruction(task_map: dict[int, str], ep: dict) -> str | None:
    """从已加载的 task_map 中获取 instruction，避免每次都读取 tasks.jsonl。"""
    # 优先使用 episode 自己的 tasks 字段
    tasks = ep.get("tasks")
    if tasks:
        for t in tasks:
            if t and t.strip():
                return t

    # 如果没有 tasks，从 task_map 查找
    ti = ep.get("task_index")
    if ti is not None and ti in task_map:
        text = task_map[ti]
        if text and text.strip():
            return text
    return None


def _extract_robointer_steps(dataset_root: Path, info: dict, ep: dict) -> tuple[str | None, list | None, int, str]:
    """从 RoboInter 类型数据集的 parquet 文件提取 raw_steps（annotation.substask + annotation.time_clip）。

    返回: (instruction, steps_raw, has_raw_steps, step_source)
    """
    if not HAS_PANDAS:
        return None, None, 0, "none"

    ei = ep["episode_index"]
    chunks_size = info.get("chunks_size", 1000)
    data_tpl = info.get("data_path", "")
    chunk = ei // chunks_size
    pq_rel = data_tpl.format(episode_chunk=chunk, episode_index=ei)
    pq_path = dataset_root / pq_rel

    instruction = None
    steps_raw = None

    # 使用 try-except 包裹 exists 检查，避免在文件系统问题时崩溃
    try:
        file_exists = pq_path.exists()
    except OSError:
        file_exists = False

    if file_exists:
        try:
            # 使用 pyarrow 显式打开/关闭文件，避免文件描述符泄漏
            if HAS_PYARROW:
                pf = pq_reader.ParquetFile(str(pq_path))
                try:
                    table = pf.read(columns=["frame_index", "annotation.substask", "annotation.time_clip"])
                    df = table.to_pandas()
                finally:
                    # 显式关闭底层文件句柄
                    if hasattr(pf, 'reader') and hasattr(pf.reader, 'close'):
                        pf.reader.close()
                    del pf
            else:
                df = pd.read_parquet(pq_path, columns=["frame_index", "annotation.substask", "annotation.time_clip"])

            if not df.empty and "annotation.substask" in df.columns:
                # 解析 time_clip（存储为字符串，如 "[[0, 286]]"）
                if "annotation.time_clip" in df.columns:
                    time_clip_str = df["annotation.time_clip"].iloc[0]
                    if time_clip_str:
                        try:
                            # 解析 JSON 格式的 time_clip
                            time_clip = json.loads(time_clip_str)
                            if time_clip and isinstance(time_clip, list) and len(time_clip) > 0:
                                # 构造 steps_raw，每个 time_clip 段作为一个 step
                                steps_raw = []
                                for i, clip in enumerate(time_clip):
                                    if isinstance(clip, list) and len(clip) == 2:
                                        start_frame, end_frame = clip[0], clip[1]
                                        # 获取该时间段内的 substask（读取该段第一帧的 substask）
                                        segment_rows = df[df["frame_index"] == start_frame]
                                        if not segment_rows.empty:
                                            substask = segment_rows.iloc[0]["annotation.substask"]
                                        else:
                                            # 如果找不到精确的frame，使用第一个可用的substask
                                            substask = df["annotation.substask"].iloc[0]

                                        steps_raw.append({
                                            "i": i,
                                            "desc": substask,
                                            "start": start_frame,
                                            "end": end_frame,
                                        })

                                # 使用第一个 substask 作为整体 instruction
                                if steps_raw:
                                    instruction = steps_raw[0]["desc"]
                        except (json.JSONDecodeError, ValueError):
                            pass
            del df
        except Exception:
            pass

    has_raw_steps = 1 if steps_raw else 0
    step_source = "parquet.annotation.substask" if steps_raw else "none"
    return instruction, steps_raw, has_raw_steps, step_source


def build_record_for_episode(
    ep: dict,
    info: dict,
    views: list[str],
    ds_dir: str,
    dataset_label: str,
    id_prefix: str,
    task_map: dict[int, str],
    features: dict,
    dataset_root: Path = None,
    process_type: str = "simple",
) -> dict | None:
    """为一个 episode 构建与 cluster_representation_summary.json 相同格式的记录。

    Args:
        process_type: 数据集处理类型
            - "simple": 普通数据集，从 tasks.jsonl 获取 instruction
            - "robointer": RoboInter 类型，从 parquet 提取 annotation.substask 和 time_clip
    """
    ei = ep["episode_index"]

    videos = get_video_paths(info, views, ei)

    duration_sec = None
    if ep.get("length") and info.get("fps"):
        duration_sec = round(ep["length"] / info["fps"], 3)

    # 根据 process_type 提取指令和 raw_steps
    if process_type == "robointer" and dataset_root:
        instruction_from_steps, steps_raw, has_raw_steps, step_source = _extract_robointer_steps(
            dataset_root, info, ep
        )
        # 如果从 parquet 提取失败，使用 task_map 作为后备
        instruction = instruction_from_steps or _get_instruction(task_map, ep)
    else:
        instruction = _get_instruction(task_map, ep)
        steps_raw = None
        has_raw_steps = 0
        step_source = "none"

    return make_record(
        sample_id=f"{id_prefix}-{ei}",
        dataset=dataset_label,
        dataset_dir=ds_dir,
        views=views,
        videos=videos,
        instruction=instruction,
        duration_sec=duration_sec,
        fps=info.get("fps"),
        resolution=get_resolution(features, views),
        robot_type=info.get("robot_type", "unknown"),
        has_raw_steps=has_raw_steps,
        step_source=step_source,
        steps_raw=steps_raw,
    )


def main():
    p = argparse.ArgumentParser(description="从数据集中过滤并采样 episode")
    p.add_argument("--dataset_root", type=str, required=True,
                   help="数据集根目录，例如 .../droid_1.0.1")
    p.add_argument("--dataset_label", type=str, default=None,
                   help="数据集标签 (默认: 目录名)")
    p.add_argument("--id_prefix", type=str, default=None,
                   help="sample_id 前缀 (默认: 目录名)")
    p.add_argument("--process_type", type=str, default="simple",
                   choices=["simple", "robointer"],
                   help="数据集处理类型: simple (普通), robointer (RoboInter类型，提取annotation.substask)")
    p.add_argument("--sample_n", type=int, default=10000,
                   help="采样数量 (default: 10000)")
    p.add_argument("--output", type=str,
                   default="cluster_representation_summary.json",
                   help="输出 JSON 路径")
    p.add_argument("--append", action="store_true", default=True,
                   help="如果输出文件已存在，追加到已有记录 (默认行为)")
    p.add_argument("--no-append", dest="append", action="store_false",
                   help="覆盖输出文件而非追加")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    _raise_fd_limit()

    dataset_root = Path(args.dataset_root).resolve()
    dataset_label = args.dataset_label or dataset_root.name
    id_prefix = args.id_prefix or dataset_root.name
    lerobot_root = Path(os.environ.get("VLA_DATA_ROOT", "/path/to/your/Lerobot_v21"))

    # 1. 查找 filter_report.json
    reports = find_filter_reports(str(dataset_root))
    if not reports:
        print(f"[ERROR] 在 {dataset_root} 下未找到 *filter_report.json")
        sys.exit(1)
    print(f"找到 {len(reports)} 个 filter_report:")
    for r in reports:
        print(f"  {r}")

    # 2. 提取有问题的 episode
    problem_episodes: set[int] = set()
    for rp in reports:
        problems = extract_problem_episodes(rp)
        print(f"  {Path(rp).name}: {len(problems)} 个问题 episode")
        problem_episodes |= problems
    print(f"总计 {len(problem_episodes)} 个问题 episode（去重后）")

    # 3. 读取数据集元信息
    info_path = dataset_root / "meta" / "info.json"
    eps_path = dataset_root / "meta" / "episodes.jsonl"
    if not info_path.exists() or not eps_path.exists():
        print(f"[ERROR] 缺少 meta/info.json 或 meta/episodes.jsonl")
        sys.exit(1)

    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)
    all_episodes = read_jsonl(eps_path)
    print(f"数据集总 episode 数: {len(all_episodes)}")

    # 读取 task_map
    task_map: dict[int, str] = {}
    tp = dataset_root / "meta" / "tasks.jsonl"
    if tp.exists():
        for item in read_jsonl(tp):
            task_map[item["task_index"]] = item["task"]

    # 4. 预计算公共字段
    features = info.get("features", {})
    views = extract_views(features)
    if not views:
        print(f"[ERROR] 数据集无视频 view")
        sys.exit(1)


    try:
        ds_dir = str(dataset_root.relative_to(lerobot_root))
    except ValueError:
        ds_dir = str(dataset_root)

    # 5. 过滤
    valid_episodes = [ep for ep in all_episodes if ep["episode_index"] not in problem_episodes]
    print(f"过滤后剩余: {len(valid_episodes)} 个 episode")

    # 6. 按 task 分组
    print("正在按 task 分组...")
    rng = random.Random(args.seed)
    task_to_eps: dict[str, list[dict]] = defaultdict(list)
    no_task_eps: list[dict] = []
    for ep in valid_episodes:
        instr = _get_instruction(task_map, ep)
        if instr and instr.strip():
            task_to_eps[instr].append(ep)
        else:
            no_task_eps.append(ep)

    print(f"唯一 task 数: {len(task_to_eps)}, 无 task episode: {len(no_task_eps)}（已排除）")

    # 7. 先从 task 层面采样，再为每个 task 随机选一个 episode
    # 这样可以避免需要为所有 task 都选出代表 episode（减少内存和文件访问）
    all_tasks = list(task_to_eps.keys())
    sample_n = min(args.sample_n, len(all_tasks))
    sampled_tasks = rng.sample(all_tasks, sample_n)
    print(f"从 {len(all_tasks)} 个 task 中采样 {sample_n} 个")

    # 为每个采样到的 task 随机选一个 episode
    sampled_episodes = []
    for task_text in sampled_tasks:
        eps = task_to_eps[task_text]
        sampled_episodes.append(rng.choice(eps))

    # 按 episode_index 排序，便于调试
    sampled_episodes.sort(key=lambda x: x["episode_index"])
    print(f"已选出 {len(sampled_episodes)} 个 episode（每个 task 随机选一个）")

    # 8. 构建记录（添加进度条）
    records = []
    skipped = 0
    iterator = tqdm(sampled_episodes, desc="处理 episodes") if HAS_TQDM else sampled_episodes
    for idx, ep in enumerate(iterator):
        rec = build_record_for_episode(
            ep, info, views, ds_dir, dataset_label, id_prefix, task_map, features,
            dataset_root=dataset_root, process_type=args.process_type,
        )
        if rec:
            records.append(rec)
        else:
            skipped += 1

        # 每处理 50 个 episode，做一次垃圾回收以释放残留文件句柄
        if (idx + 1) % 50 == 0:
            gc.collect()
    print(f"\n成功构建 {len(records)} 条记录, 跳过 {skipped} 条")
    if args.process_type == "robointer":
        steps_count = sum(1 for r in records if r.get("has_raw_steps", 0) == 1)
        print(f"  其中 {steps_count} 条包含 raw_steps 标注")

    # 9. 写入输出（支持追加）
    existing = []
    if args.append and os.path.isfile(args.output):
        with open(args.output, encoding="utf-8") as f:
            existing = json.load(f)
        print(f"已有记录 {len(existing)} 条，追加模式")

    all_records = existing + records
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)
    print(f"\n写入 {len(all_records)} 条记录到 {args.output}")
    print(f"  (其中已有 {len(existing)} + 新增 {len(records)})")

    # 最后再做一次垃圾回收
    gc.collect()


if __name__ == "__main__":
    main()
