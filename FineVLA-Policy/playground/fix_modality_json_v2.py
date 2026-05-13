"""
V2: Rewrite modality.json for all Aloha datasets using schema-compatible start/end format.

For each dataset group, creates modality.json with:
- state/action: left_joint, left_gripper, right_joint, right_gripper (14-dim total)
- video: image_0, image_1, image_2 (3 cameras)
- annotation: human.action.task_description

Uses the correct original_key and start/end indices based on actual parquet column structure.
"""

import json
import os
import sys
from pathlib import Path

DRY_RUN = "--dry-run" in sys.argv

LEROBOT_V21 = Path("/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/VLA_Data/Lerobot_v21")
ROBOMIND_V2 = Path("/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/RoboMindV2.0-Lerobot")


def save_json(path, data):
    if DRY_RUN:
        print(f"  [DRY-RUN] Would write: {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def build_modality(state_action_spec, video_spec):
    """Build a complete modality.json dict.

    state_action_spec: dict with keys for each sub-modality, each containing:
        original_key, start, end
    video_spec: list of (generic_name, original_key) tuples
    """
    modality = {"state": {}, "action": {}, "video": {}, "annotation": {}}
    for name, spec in state_action_spec.items():
        modality["state"][name] = {
            "original_key": spec["state_key"],
            "start": spec["start"],
            "end": spec["end"],
        }
        modality["action"][name] = {
            "original_key": spec["action_key"],
            "start": spec["start"],
            "end": spec["end"],
        }
    for generic_name, orig_key in video_spec:
        modality["video"][generic_name] = {"original_key": orig_key}
    modality["annotation"]["human.action.task_description"] = {"original_key": "task_index"}
    return modality


# === Modality specs per dataset group ===

# CM14: observation.state[14], action[14] — continuous [0,14]
CM14_SPEC = {
    "left_joint":   {"state_key": "observation.state", "action_key": "action", "start": 0, "end": 6},
    "left_gripper":  {"state_key": "observation.state", "action_key": "action", "start": 6, "end": 7},
    "right_joint":  {"state_key": "observation.state", "action_key": "action", "start": 7, "end": 13},
    "right_gripper": {"state_key": "observation.state", "action_key": "action", "start": 13, "end": 14},
}
CM14_VIDEO = [
    ("image_0", "observation.images.cam_front_rgb"),
    ("image_1", "observation.images.cam_left_wrist_rgb"),
    ("image_2", "observation.images.cam_right_wrist_rgb"),
]

# CM26 / Split Aloha: observation.state[26], action[26]
# Layout: left_joint[0:6], left_gripper[6], left_eef[7:13], right_joint[13:19], right_gripper[19], right_eef[20:26]
CM26_SPEC = {
    "left_joint":   {"state_key": "observation.state", "action_key": "action", "start": 0, "end": 6},
    "left_gripper":  {"state_key": "observation.state", "action_key": "action", "start": 6, "end": 7},
    "right_joint":  {"state_key": "observation.state", "action_key": "action", "start": 13, "end": 19},
    "right_gripper": {"state_key": "observation.state", "action_key": "action", "start": 19, "end": 20},
}
CM26_VIDEO_ROBOCOIN = [
    ("image_0", "observation.images.cam_high_rgb"),
    ("image_1", "observation.images.cam_left_wrist_rgb"),
    ("image_2", "observation.images.cam_right_wrist_rgb"),
]
CM26_VIDEO_SPLIT = [
    ("image_0", "observation.images.cam_high_rgb"),
    ("image_1", "observation.images.cam_left_wrist_rgb"),
    ("image_2", "observation.images.cam_right_wrist_rgb"),
]

# RoboMIND V1: each column is a 7-dim vector (joint_position_left/right[7])
# joint[0:6] = arm joints, [6] = gripper
RMV1_SPEC = {
    "left_joint":   {"state_key": "observation.states.joint_position_left",  "action_key": "actions.joint_position_left",  "start": 0, "end": 6},
    "left_gripper":  {"state_key": "observation.states.joint_position_left",  "action_key": "actions.joint_position_left",  "start": 6, "end": 7},
    "right_joint":  {"state_key": "observation.states.joint_position_right", "action_key": "actions.joint_position_right", "start": 0, "end": 6},
    "right_gripper": {"state_key": "observation.states.joint_position_right", "action_key": "actions.joint_position_right", "start": 6, "end": 7},
}

# RoboMIND V2 Agilex: arm_*_position_align[6] + end_effector_*_position_align (scalar → [1])
RMV2_SPEC = {
    "left_joint":   {"state_key": "observation.states.arm_left_position_align",  "action_key": "actions.arm_left_position_align",  "start": 0, "end": 6},
    "left_gripper":  {"state_key": "observation.states.end_effector_left_position_align",  "action_key": "actions.end_effector_left_position_align",  "start": 0, "end": 1},
    "right_joint":  {"state_key": "observation.states.arm_right_position_align", "action_key": "actions.arm_right_position_align", "start": 0, "end": 6},
    "right_gripper": {"state_key": "observation.states.end_effector_right_position_align", "action_key": "actions.end_effector_right_position_align", "start": 0, "end": 1},
}

# RoboMIND V2 Mobile: same structure for arms, ignore chassis
RMV2M_SPEC = RMV2_SPEC  # Same arm structure


def detect_cameras_from_info(info_path):
    """Detect camera key names from info.json."""
    with open(info_path) as f:
        info = json.load(f)
    cameras = []
    for k, v in info.get("features", {}).items():
        if v.get("dtype") == "video":
            cameras.append(k)
    return sorted(cameras)


def write_modality_for_tasks(task_dirs, spec, default_video, label):
    """Write modality.json for a list of task directories."""
    fixed = 0
    for task_dir in task_dirs:
        if isinstance(task_dir, str):
            task_dir = Path(task_dir)
        if not task_dir.exists():
            print(f"  [WARN] not found: {task_dir}")
            continue

        info_path = task_dir / "meta" / "info.json"
        modality_path = task_dir / "meta" / "modality.json"

        video_spec = default_video
        if info_path.exists():
            cams = detect_cameras_from_info(info_path)
            if cams:
                video_spec = [(f"image_{i}", cam) for i, cam in enumerate(cams)]

        modality = build_modality(spec, video_spec)
        save_json(str(modality_path), modality)
        fixed += 1

    print(f"  {label}: fixed={fixed}")
    return fixed


def get_task_dirs(parent_dir):
    """List sub-directories (task dirs) under parent_dir."""
    parent = Path(parent_dir)
    if not parent.exists():
        return []
    return sorted([d for d in parent.iterdir() if d.is_dir()])


def main():
    print(f"{'='*60}")
    print(f"V2: Rewrite modality.json with start/end format")
    print(f"Mode: {'DRY-RUN' if DRY_RUN else 'APPLY'}")
    print(f"{'='*60}\n")

    total = 0

    # Group 2: RoboCOIN CM14
    print("[Group 2] RoboCOIN CM14")
    cm14_robocoin = [
        LEROBOT_V21 / "RoboCOIN" / t for t in [
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
    ]
    cm14_add1201 = [
        LEROBOT_V21 / "RoboCOIN_add1201" / t for t in [
            "Cobot_Magic_cap_the_pen_a", "Cobot_Magic_classification_of_fruits_and_vegetables_a",
            "Cobot_Magic_drawer_storage_mineral_water", "Cobot_Magic_open_the_shoebox",
            "Cobot_Magic_plate_storage_toy", "Cobot_Magic_vase_storage_flower",
        ]
    ]
    total += write_modality_for_tasks(cm14_robocoin + cm14_add1201, CM14_SPEC, CM14_VIDEO, "CM14")

    # Group 3: RoboCOIN CM26
    print("\n[Group 3] RoboCOIN CM26")
    cm26_robocoin = [
        LEROBOT_V21 / "RoboCOIN" / t for t in [
            "Cobot_Magic_clean_blackboard", "Cobot_Magic_cube_reset",
            "Cobot_Magic_mobile_cube", "Cobot_Magic_mobile_cube_blackboard",
            "Cobot_Magic_move_beverage", "Cobot_Magic_move_plate",
            "Cobot_Magic_move_the_ball", "Cobot_Magic_move_the_ball_interference",
            "Cobot_Magic_place_square_pyramid", "Cobot_Magic_pour_drink",
            "Cobot_Magic_pour_water_bottle", "Cobot_Magic_prepare_breakfast",
            "Cobot_Magic_pull_zipper", "Cobot_Magic_pushing_magnet",
            "Cobot_Magic_twist_bottle_cap", "Cobot_Magic_water_bottle_storage",
        ]
    ]
    cm26_add1201 = [
        LEROBOT_V21 / "RoboCOIN_add1201" / t for t in [
            "Cobot_Magic_cut_banana", "Cobot_Magic_desktop_organization",
            "Cobot_Magic_food_packaging", "Cobot_Magic_make_fruit_salad",
            "Cobot_Magic_make_hamburger", "Cobot_Magic_move_the_ball_and_the_cube_block",
            "Cobot_Magic_place_the_cube_block", "Cobot_Magic_pour_water_a",
            "Cobot_Magic_take_down_the_cube_block",
        ]
    ]
    total += write_modality_for_tasks(cm26_robocoin + cm26_add1201, CM26_SPEC, CM26_VIDEO_ROBOCOIN, "CM26")

    # Group 4: Split Aloha (same 26-dim layout)
    print("\n[Group 4] RoboCOIN Split Aloha")
    split_tasks = [
        LEROBOT_V21 / "RoboCOIN" / t for t in [
            "Split_aloha_basket_storage_banana", "Split_aloha_basket_storage_bread",
            "Split_aloha_basket_storage_egg_yolk_pastry", "Split_aloha_basket_storage_long_bread",
            "Split_aloha_basket_storage_orange", "Split_aloha_basket_storage_peach",
            "Split_aloha_plate_storage", "Split_aloha_pour_rice",
            "Split_aloha_scoop_coffee_beans", "Split_aloha_stack_baskets",
            "Split_aloha_stir_coffee", "Split_aloha_wipe_table",
            "Split_aloha_wipe_the_table", "Split_aloha_zip_up_the_document_bag",
        ]
    ] + [
        LEROBOT_V21 / "RoboCOIN_add1201" / t for t in [
            "Split_aloha_fold_the_pants", "Split_aloha_pour_tea",
        ]
    ]
    total += write_modality_for_tasks(split_tasks, CM26_SPEC, CM26_VIDEO_SPLIT, "Split Aloha")

    # Group 5: RoboMIND V1.0 Agilex 3RGB
    print("\n[Group 5] RoboMIND V1.0 Agilex")
    rmv1_video = [
        ("image_0", "observation.images.camera_front"),
        ("image_1", "observation.images.camera_left_wrist"),
        ("image_2", "observation.images.camera_right_wrist"),
    ]
    for bench_dir in sorted((LEROBOT_V21 / "RoboMindV1.0").glob("benchmark1_*_compressed")):
        agilex_dir = bench_dir / "agilex_3rgb"
        if not agilex_dir.exists():
            continue
        tasks = get_task_dirs(agilex_dir)
        total += write_modality_for_tasks(tasks, RMV1_SPEC, rmv1_video, f"{bench_dir.name}/agilex_3rgb")

    # Group 6: RoboMIND V2.0 Agilex
    print("\n[Group 6] RoboMIND V2.0 Agilex")
    rmv2_video = [
        ("image_0", "observation.images.camera_front"),
        ("image_1", "observation.images.camera_left"),
        ("image_2", "observation.images.camera_right"),
    ]
    rmv2_tasks = get_task_dirs(ROBOMIND_V2 / "agilex")
    total += write_modality_for_tasks(rmv2_tasks, RMV2_SPEC, rmv2_video, "agilex")

    # Group 7: RoboMIND V2.0 Mobile
    print("\n[Group 7] RoboMIND V2.0 Agilex Mobile")
    rmv2m_tasks = get_task_dirs(ROBOMIND_V2 / "agilex_mobile")
    total += write_modality_for_tasks(rmv2m_tasks, RMV2M_SPEC, rmv2_video, "agilex_mobile")

    print(f"\n{'='*60}")
    print(f"Total: {total} modality.json files written")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
