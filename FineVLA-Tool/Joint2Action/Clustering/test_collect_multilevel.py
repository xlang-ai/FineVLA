#!/usr/bin/env python3
"""
测试 collect_cluster_representation.py 对多级目录结构的支持
"""

import os
from pathlib import Path

def test_find_cluster_results(results_root: str, dataset_name: str):
    """测试递归查找 cluster_results.json"""
    print(f"=" * 70)
    print(f"测试查找 {dataset_name} 的 cluster_results.json")
    print(f"=" * 70)

    dataset_dir = os.path.join(results_root, dataset_name)

    if not os.path.isdir(dataset_dir):
        print(f"✗ 目录不存在: {dataset_dir}")
        return

    print(f"✓ 数据集目录: {dataset_dir}")

    # 递归查找所有 cluster_results.json
    dataset_path = Path(dataset_dir)
    json_paths = []

    for json_path in dataset_path.rglob("cluster_results.json"):
        if json_path.is_file():
            json_paths.append(str(json_path))

    json_paths.sort()

    print(f"\n✓ 找到 {len(json_paths)} 个 cluster_results.json 文件\n")

    if json_paths:
        # 分析目录层级结构
        levels = {}
        for jp in json_paths:
            rel_path = os.path.relpath(jp, dataset_dir)
            depth = rel_path.count(os.sep)
            levels[depth] = levels.get(depth, 0) + 1

        print("目录层级分布:")
        for depth in sorted(levels.keys()):
            print(f"  {depth} 级: {levels[depth]} 个文件")

        # 显示前 10 个示例
        print(f"\n前 10 个文件示例:")
        for jp in json_paths[:10]:
            rel_path = os.path.relpath(jp, dataset_dir)
            print(f"  {rel_path}")

        if len(json_paths) > 10:
            print(f"  ... 还有 {len(json_paths) - 10} 个文件")
    else:
        print("✗ 没有找到 cluster_results.json 文件")

    return json_paths


def test_directory_structure(results_root: str, dataset_name: str):
    """查看数据集的目录结构"""
    print(f"\n" + "=" * 70)
    print(f"查看 {dataset_name} 的目录结构")
    print(f"=" * 70)

    dataset_dir = os.path.join(results_root, dataset_name)

    if not os.path.isdir(dataset_dir):
        print(f"✗ 目录不存在: {dataset_dir}")
        return

    # 显示前 3 级目录结构
    print(f"\n目录结构（前 3 级，每级最多显示 5 个）:\n")

    def show_tree(path, prefix="", max_items=5, max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return

        try:
            items = sorted(os.listdir(path))[:max_items]
            for i, item in enumerate(items):
                item_path = os.path.join(path, item)
                is_last = (i == len(items) - 1)

                connector = "└── " if is_last else "├── "
                print(f"{prefix}{connector}{item}")

                if os.path.isdir(item_path):
                    extension = "    " if is_last else "│   "
                    show_tree(item_path, prefix + extension, max_items, max_depth, current_depth + 1)

            if len(os.listdir(path)) > max_items:
                connector = "└── " if is_last else "├── "
                print(f"{prefix}{'    ' if is_last else '│   '}... ({len(os.listdir(path)) - max_items} more)")
        except PermissionError:
            pass

    print(f"{dataset_name}/")
    show_tree(dataset_dir)


if __name__ == "__main__":
    import sys

    # 默认参数
    results_root = "./results_two_stage"

    if len(sys.argv) > 1:
        results_root = sys.argv[1]

    # 测试 RoboMindV1.0（3级目录）
    print("\n" + "🔍 测试 RoboMindV1.0（3级目录结构）")
    test_directory_structure(results_root, "RoboMindV1.0")
    json_paths_v1 = test_find_cluster_results(results_root, "RoboMindV1.0")

    # 测试 RoboMindV2.0（2级目录）
    print("\n\n" + "🔍 测试 RoboMindV2.0（2级目录结构）")
    test_directory_structure(results_root, "RoboMindV2.0")
    json_paths_v2 = test_find_cluster_results(results_root, "RoboMindV2.0")

    # 测试普通数据集（1级目录）
    print("\n\n" + "🔍 测试 Galaxea（1级目录结构）")
    test_directory_structure(results_root, "Galaxea")
    json_paths_gal = test_find_cluster_results(results_root, "Galaxea")

    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"RoboMindV1.0: {len(json_paths_v1) if json_paths_v1 else 0} 个文件")
    print(f"RoboMindV2.0: {len(json_paths_v2) if json_paths_v2 else 0} 个文件")
    print(f"Galaxea:      {len(json_paths_gal) if json_paths_gal else 0} 个文件")
    print("\n✅ 递归查找功能正常工作！")
