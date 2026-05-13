"""
批量对 RoboMindV1.0 / RoboMindV2.0 所有子任务运行 convert_unified.py。

工作流:
1. 遍历每个 benchmark/robot_type (V1.0) 或 robot_type (V2.0) 目录
2. 读取 robot_type 顶层的 modality.json
3. 为每个子任务 symlink modality.json 到 task/meta/modality.json (如果不存在)
4. 调用 convert_unified.py 处理该子任务
5. 处理完后清理 symlink

用法:
    # 处理所有 RoboMindV1.0 + V2.0 (跳过已有 unified_output 的)
    python batch_convert_robomind.py --skip-existing

    # 强制重新转换所有
    python batch_convert_robomind.py --force-reconvert

    # dry-run: 只列出要处理的任务，不实际执行
    python batch_convert_robomind.py --list-only

    # 只跑某个版本
    python batch_convert_robomind.py --version v1
    python batch_convert_robomind.py --version v2

    # 只跑某个机器人型号 (支持部分匹配)
    python batch_convert_robomind.py --robot agilex_3rgb

    # 限制每个任务处理的 episode 数
    python batch_convert_robomind.py --episodes 100
"""

import os
import sys
import argparse
import subprocess
import time

DATA_ROOT = "/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONVERT_SCRIPT = os.path.join(SCRIPT_DIR, "convert_unified.py")

SKIP_NAMES = {"video", "videos", "data", "meta", "unified_output", "images"}

# V1.0: benchmark/robot_type/task
V1_BENCHMARKS = [
    "RoboMindV1.0/benchmark1_0_compressed",
    "RoboMindV1.0/benchmark1_1_compressed",
    "RoboMindV1.0/benchmark1_2_compressed",
]

# V2.0: robot_type/task
V2_ROOT = "RoboMindV2.0"


def list_subdirs(path):
    """列出目录下的子目录，排除特殊名称"""
    if not os.path.isdir(path):
        return []
    return sorted([
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d)) and d not in SKIP_NAMES
    ])


def collect_tasks(version_filter=None, robot_filter=None):
    """收集所有待处理的 (robot_type_dir, task_dir, modality_path) 三元组"""
    tasks = []

    # V1.0
    if version_filter in (None, "v1"):
        for bench_rel in V1_BENCHMARKS:
            bench_path = os.path.join(DATA_ROOT, bench_rel)
            if not os.path.isdir(bench_path):
                continue
            for robot in list_subdirs(bench_path):
                if robot_filter and robot_filter not in robot:
                    continue
                robot_dir = os.path.join(bench_path, robot)
                modality = os.path.join(robot_dir, "modality.json")
                if not os.path.isfile(modality):
                    print(f"[跳过] 无 modality.json: {robot_dir}")
                    continue
                for task in list_subdirs(robot_dir):
                    task_dir = os.path.join(robot_dir, task)
                    # 确认有 meta/info.json
                    if not os.path.isfile(os.path.join(task_dir, "meta", "info.json")):
                        continue
                    tasks.append((robot_dir, task_dir, modality, f"V1.0/{os.path.basename(bench_path)}/{robot}/{task}"))

    # V2.0
    if version_filter in (None, "v2"):
        v2_path = os.path.join(DATA_ROOT, V2_ROOT)
        if os.path.isdir(v2_path):
            for robot in list_subdirs(v2_path):
                if robot_filter and robot_filter not in robot:
                    continue
                robot_dir = os.path.join(v2_path, robot)
                modality = os.path.join(robot_dir, "modality.json")
                if not os.path.isfile(modality):
                    print(f"[跳过] 无 modality.json: {robot_dir}")
                    continue
                for task in list_subdirs(robot_dir):
                    task_dir = os.path.join(robot_dir, task)
                    if not os.path.isfile(os.path.join(task_dir, "meta", "info.json")):
                        continue
                    tasks.append((robot_dir, task_dir, modality, f"V2.0/{robot}/{task}"))

    return tasks


def ensure_modality_link(task_dir, modality_src):
    """确保 task/meta/modality.json 存在（symlink 到 robot_type 级别的）。
    返回: (是否新创建, 路径)
    """
    target = os.path.join(task_dir, "meta", "modality.json")
    if os.path.exists(target):
        # 已存在（可能是之前的 symlink 或真实文件）
        return False, target

    os.symlink(modality_src, target)
    return True, target


def run_convert(task_dir, args):
    """运行 convert_unified.py"""
    cmd = [sys.executable, CONVERT_SCRIPT, task_dir]
    if args.episodes:
        cmd += ["--episodes", str(args.episodes)]
    if args.skip_existing:
        cmd += ["--skip-existing"]
    return subprocess.run(cmd, cwd=SCRIPT_DIR)


def main():
    parser = argparse.ArgumentParser(description="批量运行 convert_unified.py")
    parser.add_argument("--version", choices=["v1", "v2"], default=None,
                        help="只处理指定版本 (v1/v2)")
    parser.add_argument("--robot", default=None,
                        help="只处理匹配的机器人型号 (部分匹配)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="每个任务最多处理的 episode 数")
    parser.add_argument("--skip-existing", action="store_true",
                        help="跳过已有 unified parquet 的 episode")
    parser.add_argument("--force-reconvert", action="store_true",
                        help="强制重新转换 (删除已有 unified_output)")
    parser.add_argument("--list-only", action="store_true",
                        help="只列出要处理的任务，不实际执行")
    parser.add_argument("--clean-links", action="store_true",
                        help="只清理之前创建的 modality.json symlink")
    args = parser.parse_args()

    tasks = collect_tasks(version_filter=args.version, robot_filter=args.robot)
    print(f"\n共找到 {len(tasks)} 个子任务待处理\n")

    if args.list_only:
        for _, _, _, label in tasks:
            print(f"  {label}")
        return

    if args.clean_links:
        cleaned = 0
        for _, task_dir, _, label in tasks:
            link = os.path.join(task_dir, "meta", "modality.json")
            if os.path.islink(link):
                os.remove(link)
                cleaned += 1
        print(f"清理了 {cleaned} 个 symlink")
        return

    succeeded = 0
    failed = 0
    skipped = 0
    created_links = []

    for i, (robot_dir, task_dir, modality_src, label) in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(tasks)}] {label}")
        print(f"{'='*60}")

        # 如果 force-reconvert，先删掉已有的 unified_output
        unified_dir = os.path.join(task_dir, "unified_output")
        if args.force_reconvert and os.path.isdir(unified_dir):
            import shutil
            shutil.rmtree(unified_dir)
            print(f"  已删除旧 unified_output")
        elif not args.force_reconvert and not args.skip_existing and os.path.isdir(unified_dir):
            # 如果 unified_output 已存在且未指定 skip-existing 或 force，跳过
            unified_files = [f for f in os.listdir(unified_dir) if f.endswith(".parquet")]
            if unified_files:
                print(f"  unified_output 已存在 ({len(unified_files)} files)，跳过 (用 --force-reconvert 强制重跑)")
                skipped += 1
                continue

        # 确保 modality.json 存在
        is_new, link_path = ensure_modality_link(task_dir, modality_src)
        if is_new:
            created_links.append(link_path)

        # 运行转换
        t0 = time.time()
        result = run_convert(task_dir, args)
        elapsed = time.time() - t0

        if result.returncode == 0:
            succeeded += 1
            print(f"  完成 ({elapsed:.1f}s)")
        else:
            failed += 1
            print(f"  失败 (returncode={result.returncode}, {elapsed:.1f}s)")

    # 清理创建的 symlink
    for link in created_links:
        if os.path.islink(link):
            os.remove(link)

    print(f"\n{'='*60}")
    print(f"批量转换完成: 成功={succeeded}, 失败={failed}, 跳过={skipped}, 总计={len(tasks)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
