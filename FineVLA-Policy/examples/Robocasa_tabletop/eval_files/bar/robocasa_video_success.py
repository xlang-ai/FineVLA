#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计每个 task 的 success rate（success1=成功，success0=失败）。")
    parser.add_argument("--root", type=str, required=True, help="任务目录根路径（下一级是 24 个 task 文件夹）。")
    parser.add_argument("--out", type=str, required=True, help="输出汇总表格文件（建议 .md 或 .log）。")
    parser.add_argument("--expected_tasks", type=int, default=24, help="期望 task 数量，默认 24。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_path = Path(args.out)

    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root}")

    # success1=成功, success0=失败
    pattern = re.compile(r"_success([01])\.mp4$")

    task_rows: list[tuple[str, int, int, float]] = []
    total_success = 0
    total_count = 0

    task_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    for task_dir in task_dirs:
        success = 0
        total = 0
        for video in task_dir.glob("*.mp4"):
            m = pattern.search(video.name)
            if not m:
                continue
            total += 1
            if m.group(1) == "1":
                success += 1

        rate = (success / total) if total > 0 else 0.0
        task_rows.append((task_dir.name, success, total, rate))
        total_success += success
        total_count += total

    # 24 个 task 的平均（宏平均）
    macro_avg = sum(r[3] for r in task_rows) / len(task_rows) if task_rows else 0.0
    # 全部样本的平均（微平均）
    micro_avg = (total_success / total_count) if total_count > 0 else 0.0

    lines: list[str] = []
    lines.append(f"Root: {root}")
    lines.append(f"Detected tasks: {len(task_rows)} (expected: {args.expected_tasks})")
    lines.append("")
    lines.append("| Task | Success | Total | Success Rate |")
    lines.append("|---|---:|---:|---:|")

    for task, success, total, rate in task_rows:
        lines.append(f"| {task} | {success} | {total} | {rate:.4f} |")

    lines.append("")
    lines.append(f"Macro Average (task mean): {macro_avg:.4f}")
    lines.append(f"Micro Average (all videos): {total_success}/{total_count} = {micro_avg:.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved summary: {out_path}")


if __name__ == "__main__":
    main()
