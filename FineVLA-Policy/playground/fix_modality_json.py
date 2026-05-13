"""
Batch fix/create modality.json for all Aloha-like datasets.

For each dataset group:
- RoboCOIN CM14/CM26/Split Aloha: supplement missing video + annotation sections
- RoboMIND V1: copy parent modality.json to each sub-task dir + add video + annotation
- RoboMIND V2/V2-Mobile: copy parent modality.json to each sub-task dir + add video + annotation

Usage:
    python playground/fix_modality_json.py --dry-run   # preview changes
    python playground/fix_modality_json.py              # apply changes
"""

import json
import os
import sys
import shutil
from pathlib import Path
from copy import deepcopy

DRY_RUN = "--dry-run" in sys.argv

LEROBOT_V21 = Path("/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21")
ROBOMIND_V2 = Path("/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/RoboMindV2.0-Lerobot")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    if DRY_RUN:
        print(f"  [DRY-RUN] Would write: {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def get_task_dirs(parent_dir, exclude_files=True):
    """List sub-directories (task dirs) under parent_dir, skip files."""
    result = []
    for name in sorted(os.listdir(parent_dir)):
        full = parent_dir / name
        if full.is_dir():
            result.append(full)
    return result


def detect_camera_keys(info_json):
    """Auto-detect camera keys from info.json features."""
    cameras = []
    for k in info_json.get("features", {}):
        if "images" in k or "image" in k:
            if info_json["features"][k].get("dtype") == "video":
                cameras.append(k)
    return sorted(cameras)


def build_video_section(camera_keys):
    """Build video section for modality.json.
    Maps camera keys to generic names: image_0, image_1, image_2.
    """
    video = {}
    for i, cam_key in enumerate(camera_keys):
        video[f"image_{i}"] = {"original_key": cam_key}
    return video


def build_annotation_section():
    """Build annotation section for modality.json."""
    return {
        "human.action.task_description": {
            "original_key": "task_index"
        }
    }


def fix_robocoin_tasks(data_root, task_names):
    """Fix RoboCOIN tasks: add video + annotation to existing modality.json."""
    fixed = 0
    skipped = 0
    for task_name in task_names:
        task_dir = data_root / task_name
        if not task_dir.exists():
            print(f"  [WARN] Task dir not found: {task_dir}")
            continue

        modality_path = task_dir / "meta" / "modality.json"
        info_path = task_dir / "meta" / "info.json"

        if not modality_path.exists():
            print(f"  [WARN] No modality.json: {modality_path}")
            continue

        modality = load_json(modality_path)
        info = load_json(info_path)

        if "video" in modality and modality["video"]:
            skipped += 1
            continue

        camera_keys = detect_camera_keys(info)
        if not camera_keys:
            print(f"  [WARN] No camera keys found in {info_path}")
            continue

        modality["video"] = build_video_section(camera_keys)
        modality["annotation"] = build_annotation_section()
        save_json(modality_path, modality)
        fixed += 1

    return fixed, skipped


def fix_robomind_tasks(data_root, parent_modality_path, sub_task_dirs):
    """Fix RoboMIND tasks: copy parent modality + add video + annotation."""
    if not parent_modality_path.exists():
        print(f"  [ERROR] Parent modality.json not found: {parent_modality_path}")
        return 0, 0

    parent_modality = load_json(parent_modality_path)
    fixed = 0
    skipped = 0

    for task_dir in sub_task_dirs:
        modality_path = task_dir / "meta" / "modality.json"
        info_path = task_dir / "meta" / "info.json"

        if not info_path.exists():
            print(f"  [WARN] No info.json: {info_path}")
            continue

        # Check if already complete
        if modality_path.exists():
            existing = load_json(modality_path)
            if "video" in existing and existing["video"]:
                skipped += 1
                continue

        info = load_json(info_path)
        camera_keys = detect_camera_keys(info)
        if not camera_keys:
            print(f"  [WARN] No camera keys found in {info_path}")
            continue

        new_modality = deepcopy(parent_modality)
        new_modality["video"] = build_video_section(camera_keys)
        new_modality["annotation"] = build_annotation_section()
        save_json(modality_path, new_modality)
        fixed += 1

    return fixed, skipped


def main():
    print(f"{'='*60}")
    print(f"Batch fix modality.json for Aloha multi-dataset training")
    print(f"Mode: {'DRY-RUN' if DRY_RUN else 'APPLY'}")
    print(f"{'='*60}\n")

    total_fixed = 0
    total_skipped = 0

    # ===== Group 2: RoboCOIN CM14 =====
    print("[Group 2] RoboCOIN CM14 (14-dim, from RoboCOIN + RoboCOIN_add1201)")
    cm14_tasks = [
        "Cobot_Magic_box_storage_chopsticks", "Cobot_Magic_catch_the_ball",
        "Cobot_Magic_classification_of_fruits_and_vegetables", "Cobot_Magic_classification_of_tableware",
        "Cobot_Magic_clean_up_the_tableware", "Cobot_Magic_clear_the_desktop",
        "Cobot_Magic_close_book", "Cobot_Magic_close_button",
        "Cobot_Magic_fold_clothes", "Cobot_Magic_fold_the_towel",
        "Cobot_Magic_fold_towel_a", "Cobot_Magic_move_the_bread",
        "Cobot_Magic_move_the_cup", "Cobot_Magic_move_the_plate",
        "Cobot_Magic_move_the_small_ball", "Cobot_Magic_movethe_position_of_the_bluetooth",
        "Cobot_Magic_place_the_test_tube", "Cobot_Magic_plate_storage_apple",
        "Cobot_Magic_plate_storage_bread", "Cobot_Magic_plate_storaje_baozi",
        "Cobot_Magic_pot_storage_steamer", "Cobot_Magic_put_in_the_pear",
        "Cobot_Magic_put_the_building_block_on_the_table", "Cobot_Magic_steamer_storage_dumpling",
        "Cobot_Magic_storage_plate", "Cobot_Magic_take_out_a_pen_from_the_pen_holder",
        "Cobot_Magic_take_out_the_bread", "Cobot_Magic_take_the_shoes_off_the_shelf",
        "Cobot_Magic_the_box_stores_table_tennis_balls", "Cobot_Magic_the_plate_holds_the_fruit",
        "Cobot_Magic_the_plate_holds_the_vegetables", "Cobot_Magic_turn_off_the_desk_lamp",
        "Cobot_Magic_turn_on_the_bulb", "Cobot_Magic_turn_on_the_desk_lamp",
    ]
    cm14_add1201 = [
        "Cobot_Magic_cap_the_pen_a", "Cobot_Magic_classification_of_fruits_and_vegetables_a",
        "Cobot_Magic_drawer_storage_mineral_water", "Cobot_Magic_open_the_shoebox",
        "Cobot_Magic_plate_storage_toy", "Cobot_Magic_vase_storage_flower",
    ]
    f, s = fix_robocoin_tasks(LEROBOT_V21 / "RoboCOIN", cm14_tasks)
    total_fixed += f; total_skipped += s
    print(f"  RoboCOIN: fixed={f}, skipped={s}")
    f, s = fix_robocoin_tasks(LEROBOT_V21 / "RoboCOIN_add1201", cm14_add1201)
    total_fixed += f; total_skipped += s
    print(f"  RoboCOIN_add1201: fixed={f}, skipped={s}")

    # ===== Group 3: RoboCOIN CM26 =====
    print("\n[Group 3] RoboCOIN CM26 (26-dim)")
    cm26_tasks = [
        "Cobot_Magic_clean_blackboard", "Cobot_Magic_cube_reset",
        "Cobot_Magic_mobile_cube", "Cobot_Magic_mobile_cube_blackboard",
        "Cobot_Magic_move_beverage", "Cobot_Magic_move_plate",
        "Cobot_Magic_move_the_ball", "Cobot_Magic_move_the_ball_interference",
        "Cobot_Magic_place_square_pyramid", "Cobot_Magic_pour_drink",
        "Cobot_Magic_pour_water_bottle", "Cobot_Magic_prepare_breakfast",
        "Cobot_Magic_pull_zipper", "Cobot_Magic_pushing_magnet",
        "Cobot_Magic_twist_bottle_cap", "Cobot_Magic_water_bottle_storage",
    ]
    cm26_add1201 = [
        "Cobot_Magic_cut_banana", "Cobot_Magic_desktop_organization",
        "Cobot_Magic_food_packaging", "Cobot_Magic_make_fruit_salad",
        "Cobot_Magic_make_hamburger", "Cobot_Magic_move_the_ball_and_the_cube_block",
        "Cobot_Magic_place_the_cube_block", "Cobot_Magic_pour_water_a",
        "Cobot_Magic_take_down_the_cube_block",
    ]
    f, s = fix_robocoin_tasks(LEROBOT_V21 / "RoboCOIN", cm26_tasks)
    total_fixed += f; total_skipped += s
    print(f"  RoboCOIN: fixed={f}, skipped={s}")
    f, s = fix_robocoin_tasks(LEROBOT_V21 / "RoboCOIN_add1201", cm26_add1201)
    total_fixed += f; total_skipped += s
    print(f"  RoboCOIN_add1201: fixed={f}, skipped={s}")

    # ===== Group 4: Split Aloha =====
    print("\n[Group 4] RoboCOIN Split Aloha (26-dim)")
    split_aloha_tasks = [
        "Split_aloha_basket_storage_banana", "Split_aloha_basket_storage_bread",
        "Split_aloha_basket_storage_egg_yolk_pastry", "Split_aloha_basket_storage_long_bread",
        "Split_aloha_basket_storage_orange", "Split_aloha_basket_storage_peach",
        "Split_aloha_plate_storage", "Split_aloha_pour_rice",
        "Split_aloha_scoop_coffee_beans", "Split_aloha_stack_baskets",
        "Split_aloha_stir_coffee", "Split_aloha_wipe_table",
        "Split_aloha_wipe_the_table", "Split_aloha_zip_up_the_document_bag",
    ]
    split_aloha_add1201 = [
        "Split_aloha_fold_the_pants", "Split_aloha_pour_tea",
    ]
    f, s = fix_robocoin_tasks(LEROBOT_V21 / "RoboCOIN", split_aloha_tasks)
    total_fixed += f; total_skipped += s
    print(f"  RoboCOIN: fixed={f}, skipped={s}")
    f, s = fix_robocoin_tasks(LEROBOT_V21 / "RoboCOIN_add1201", split_aloha_add1201)
    total_fixed += f; total_skipped += s
    print(f"  RoboCOIN_add1201: fixed={f}, skipped={s}")

    # ===== Group 5: RoboMIND V1 Agilex =====
    print("\n[Group 5] RoboMIND V1.0 Agilex 3RGB")
    for bench_dir in sorted((LEROBOT_V21 / "RoboMindV1.0").glob("benchmark1_*_compressed")):
        agilex_dir = bench_dir / "agilex_3rgb"
        if not agilex_dir.exists():
            continue
        parent_modality = agilex_dir / "modality.json"
        task_dirs = get_task_dirs(agilex_dir)
        f, s = fix_robomind_tasks(agilex_dir, parent_modality, task_dirs)
        total_fixed += f; total_skipped += s
        print(f"  {bench_dir.name}/agilex_3rgb: fixed={f}, skipped={s}")

    # ===== Group 6: RoboMIND V2 Agilex =====
    print("\n[Group 6] RoboMIND V2.0 Agilex")
    agilex_dir = ROBOMIND_V2 / "agilex"
    parent_modality = agilex_dir / "modality.json"
    task_dirs = get_task_dirs(agilex_dir)
    f, s = fix_robomind_tasks(agilex_dir, parent_modality, task_dirs)
    total_fixed += f; total_skipped += s
    print(f"  agilex: fixed={f}, skipped={s}")

    # ===== Group 7: RoboMIND V2 Agilex Mobile =====
    print("\n[Group 7] RoboMIND V2.0 Agilex Mobile")
    mobile_dir = ROBOMIND_V2 / "agilex_mobile"
    parent_modality = mobile_dir / "modality.json"
    task_dirs = get_task_dirs(mobile_dir)
    f, s = fix_robomind_tasks(mobile_dir, parent_modality, task_dirs)
    total_fixed += f; total_skipped += s
    print(f"  agilex_mobile: fixed={f}, skipped={s}")

    print(f"\n{'='*60}")
    print(f"Total: fixed={total_fixed}, skipped={total_skipped}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
