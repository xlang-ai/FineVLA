# -*- coding: utf-8 -*-
"""
绘制各数据集过滤报告的错误类型饼图

从 Lerobot_v21/{dataset}/{dataset}_filter_report.json 读取过滤报告
对其中的各种错误类型进行饼图绘制
一个数据集一个图，RoboCOIN的三个子数据集合并为一个数据集来处理

用法: python Draw_problem_epiosde_pie.py
输出: pie_charts/ 目录下的各数据集饼图
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 工作目录（脚本所在目录，用于输出）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据集根目录（filter_report.json 文件所在的数据集目录）
DATA_ROOT = "/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21"

# 数据集列表
DATASETS = [
    "BC_Z",
    "Bridge",
    "droid_RoboInter",
    "Galaxea-Open-World-Dataset",
    "RDT-yhq",
    "RH20T-RoboInter",
    "RoboCOIN",  # 会合并 RoboCOIN, RoboCOIN_add0130, RoboCOIN_add1201
    "RoboCOIN_add0130",
    "RoboCOIN_add1201",
    "RoboMindV1.0",
    "RoboMindV2.0",
    "RT-1"
]

# RoboCOIN 相关数据集（需要合并）
ROBOCOIN_DATASETS = ["RoboCOIN", "RoboCOIN_add0130", "RoboCOIN_add1201"]

# 错误类型映射（中文标签）
ERROR_LABELS = {
    "parquet_corrupted": "Parquet文件损坏",
    "frame_too_short": "帧数不足",
    "task_empty": "Task为空",
    "l2_abnormal": "L2异常"
}

# 饼图颜色方案
COLORS = {
    "parquet_corrupted": "#9C27B0",  # 紫色
    "frame_too_short": "#FF9800",    # 橙色
    "task_empty": "#2196F3",         # 蓝色
    "l2_abnormal": "#E91E63"         # 粉红色
}


def load_filter_report(dataset_name):
    """加载过滤报告 JSON 文件

    Parameters:
        dataset_name: str, 数据集名称

    Returns:
        dict or None: 报告内容，如果文件不存在则返回 None
    """
    # 报告文件在数据集目录下：{DATA_ROOT}/{dataset_name}/{dataset_name}_filter_report.json
    report_path = os.path.join(DATA_ROOT, dataset_name, f"{dataset_name}_filter_report.json")
    if not os.path.isfile(report_path):
        print(f"  [WARN] 未找到报告文件: {report_path}")
        return None

    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_breakdown(report):
    """从报告中提取错误类型统计

    Parameters:
        report: dict, 报告内容

    Returns:
        dict or None: {error_type: count}，如果报告为空则返回 None
    """
    if report is None:
        return None

    summary = report.get("summary", {})
    breakdown = summary.get("breakdown", {})

    return {
        "parquet_corrupted": breakdown.get("parquet_corrupted", 0),
        "frame_too_short": breakdown.get("frame_too_short", 0),
        "task_empty": breakdown.get("task_empty", 0),
        "l2_abnormal": breakdown.get("l2_abnormal", 0)
    }


def merge_robocoin_breakdown(datasets):
    """合并 RoboCOIN 的三个子数据集统计

    Parameters:
        datasets: list of str, 需要合并的数据集名称列表

    Returns:
        dict or None: 合并后的 breakdown
    """
    merged = {
        "parquet_corrupted": 0,
        "frame_too_short": 0,
        "task_empty": 0,
        "l2_abnormal": 0
    }

    found_any = False
    for ds_name in datasets:
        report = load_filter_report(ds_name)
        breakdown = extract_breakdown(report)
        if breakdown is not None:
            found_any = True
            for key in merged:
                merged[key] += breakdown[key]
            print(f"    已合并: {ds_name}")

    return merged if found_any else None


def plot_pie_chart(breakdown, dataset_name, output_dir):
    """绘制单个数据集的错误类型饼图

    Parameters:
        breakdown: dict, {error_type: count}
        dataset_name: str, 数据集名称
        output_dir: str, 输出目录
    """
    if breakdown is None:
        print(f"  [SKIP] 无数据，跳过绘图")
        return

    # 过滤掉值为 0 的项
    labels = []
    sizes = []
    colors = []

    for error_type in ["parquet_corrupted", "frame_too_short", "task_empty", "l2_abnormal"]:
        count = breakdown.get(error_type, 0)
        if count > 0:
            label = ERROR_LABELS[error_type]
            labels.append(f"{label}\n({count:,})")
            sizes.append(count)
            colors.append(COLORS[error_type])

    if not sizes:
        print(f"  [INFO] 无任何问题 episode，跳过绘图")
        return

    # 创建饼图
    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制饼图
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=[0.05] * len(sizes),  # 突出显示所有扇区
        textprops={'fontsize': 12}
    )

    # 设置百分比文本格式
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')
        autotext.set_fontsize(13)

    # 设置标题
    total = sum(sizes)
    ax.set_title(f'{dataset_name} 问题Episode类型分布\n总计: {total:,} 个问题Episode',
                 fontsize=16, weight='bold', pad=20)

    # 保证饼图为圆形
    ax.axis('equal')

    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_name}_error_pie.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  [OK] 已保存: {output_path}")


def main():
    """主函数"""
    output_dir = os.path.join(SCRIPT_DIR, "pie_charts")
    print(f"输出目录: {output_dir}")
    print(f"{'='*70}")

    processed_count = 0
    skipped_count = 0
    robocoin_merged = False

    for dataset in DATASETS:
        if dataset in ROBOCOIN_DATASETS:
            # 如果是 RoboCOIN，只处理第一个（合并所有三个）
            if not robocoin_merged:
                print(f"\n[{processed_count + 1}] 处理 RoboCOIN (合并三个子数据集)")
                breakdown = merge_robocoin_breakdown(ROBOCOIN_DATASETS)
                plot_pie_chart(breakdown, "RoboCOIN_merged", output_dir)
                robocoin_merged = True
                if breakdown is not None:
                    processed_count += 1
                else:
                    skipped_count += 1
            # 其他两个跳过（已在第一个处理时合并）
            continue
        else:
            # 普通数据集
            print(f"\n[{processed_count + skipped_count + 1}] 处理 {dataset}")
            report = load_filter_report(dataset)
            breakdown = extract_breakdown(report)
            plot_pie_chart(breakdown, dataset, output_dir)
            if breakdown is not None and sum(breakdown.values()) > 0:
                processed_count += 1
            else:
                skipped_count += 1

    print(f"\n{'='*70}")
    print(f"处理完成！")
    print(f"  成功生成饼图: {processed_count} 个")
    print(f"  跳过/无数据: {skipped_count} 个")
    print(f"  输出目录: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
