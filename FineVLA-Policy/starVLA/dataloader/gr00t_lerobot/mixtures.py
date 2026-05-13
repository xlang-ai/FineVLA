"""
mixtures.py

Defines a registry of dataset mixtures and weights for the Open-X Embodiment Datasets. Each dataset is associated with
a float "sampling weight"
"""

import os
from typing import Dict, List, Tuple


def _scan_subdatasets(parent_dir, robot_type, weight=1.0, prefix=None):
    """Scan all subdirectories under parent_dir and return mixture entries.

    When *prefix* is given, each task name becomes ``prefix/task`` so that
    ``data_root_dir / prefix / task`` resolves to the correct dataset path.
    """
    if not os.path.isdir(parent_dir):
        return []
    tasks = sorted([d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))])
    if prefix:
        return [(f"{prefix}/{task}", weight, robot_type) for task in tasks]
    return [(task, weight, robot_type) for task in tasks]


# Dataset mixture name mapped to a list of tuples containing:
## {nakename: [(data_name, sampling_weight, robot_type)] }
## move this to config file later
DATASET_NAMED_MIXTURES = {

    "custom_dataset": [
        ("custom_dataset_name", 1.0, "custom_robot_config"),
    ],
    "custom_dataset_2": [
        ("custom_dataset_name_1", 1.0, "custom_robot_config"),
        ("custom_dataset_name_2", 1.0, "custom_robot_config"),
    ],

    "libero_all": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
                # ("libero_90_no_noops_lerobot", 1.0, "libero_franka"),
    ],
    "bridge": [
        ("bridge_orig_1.0.0_lerobot", 1.0, "oxe_bridge"),
    ],
    "bridge_rt_1": [
        ("bridge_orig_1.0.0_lerobot", 1.0, "oxe_bridge"),
        ("fractal20220817_data_0.1.0_lerobot", 0.5, "oxe_rt1"),
    ],

    "demo_sim_pick_place": [
        ("sim_pick_place", 1.0, "demo_sim_franka_delta_joints"),
    ],

    "custom_dataset": [
        ("custom_dataset_name", 1.0, "custom_robot_config"),
    ],
    "custom_dataset_2": [
        ("custom_dataset_name_1", 1.0, "custom_robot_config"),
        ("custom_dataset_name_2", 1.0, "custom_robot_config"),
    ],

    "fourier_gr1_unified_1000": [
        ("gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
    ],

    "robocasa365_single": [
        ("robocasa365", 1.0, "robocasa365_panda_omron"),
    ],

    "BEHAVIOR_challenge": [
        ("BEHAVIOR_challenge", 1.0, "R1Pro"),
    ],


    "SO101_pick": [
        ("pick_dataset_name", 1.0, "SO101"),
    ],

    "arx_x5": [
        ("arx_x5", 1.0, "arx_x5"),
    ],

    "robotwin": [
        ("adjust_bottle", 1.0, "robotwin"),
        ("beat_block_hammer", 1.0, "robotwin"),
        ("blocks_ranking_rgb", 1.0, "robotwin"),
        ("blocks_ranking_size", 1.0, "robotwin"),
        ("click_alarmclock", 1.0, "robotwin"),
        ("click_bell", 1.0, "robotwin"),
        ("dump_bin_bigbin", 1.0, "robotwin"),
        ("grab_roller", 1.0, "robotwin"),
        ("handover_block", 1.0, "robotwin"),
        ("handover_mic", 1.0, "robotwin"),
        ("hanging_mug", 1.0, "robotwin"),
        ("lift_pot", 1.0, "robotwin"),
        ("move_can_pot", 1.0, "robotwin"),
        ("move_pillbottle_pad", 1.0, "robotwin"),
        ("move_playingcard_away", 1.0, "robotwin"),
        ("move_stapler_pad", 1.0, "robotwin"),
        ("open_laptop", 1.0, "robotwin"),
        ("open_microwave", 1.0, "robotwin"),
        ("pick_diverse_bottles", 1.0, "robotwin"),
        ("pick_dual_bottles", 1.0, "robotwin"),
        ("place_a2b_left", 1.0, "robotwin"),
        ("place_a2b_right", 1.0, "robotwin"),
        ("place_bread_basket", 1.0, "robotwin"),
        ("place_bread_skillet", 1.0, "robotwin"),
        ("place_burger_fries", 1.0, "robotwin"),
        ("place_can_basket", 1.0, "robotwin"),
        ("place_cans_plasticbox", 1.0, "robotwin"),
        ("place_container_plate", 1.0, "robotwin"),
        ("place_dual_shoes", 1.0, "robotwin"),
        ("place_empty_cup", 1.0, "robotwin"),
        ("place_fan", 1.0, "robotwin"),
        ("place_mouse_pad", 1.0, "robotwin"),
        ("place_object_basket", 1.0, "robotwin"),
        ("place_object_scale", 1.0, "robotwin"),
        ("place_object_stand", 1.0, "robotwin"),
        ("place_phone_stand", 1.0, "robotwin"),
        ("place_shoe", 1.0, "robotwin"),
        ("press_stapler", 1.0, "robotwin"),
        ("put_bottles_dustbin", 1.0, "robotwin"),
        ("put_object_cabinet", 1.0, "robotwin"),
        ("rotate_qrcode", 1.0, "robotwin"),
        ("scan_object", 1.0, "robotwin"),
        ("shake_bottle", 1.0, "robotwin"),
        ("shake_bottle_horizontally", 1.0, "robotwin"),
        ("stack_blocks_three", 1.0, "robotwin"),
        ("stack_blocks_two", 1.0, "robotwin"),
        ("stack_bowls_three", 1.0, "robotwin"),
        ("stack_bowls_two", 1.0, "robotwin"),
        ("stamp_seal", 1.0, "robotwin"),
        ("turn_switch", 1.0, "robotwin"),
    ],

    "robotwin_clean": _scan_subdatasets(
        "/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/StarVLA_YJH/Benchmark/RoboTwin-Clean",
        "lerobot_v21_robotwin",
    ),

    "robotwin_mix": (
        _scan_subdatasets(
            "/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/StarVLA_YJH/Benchmark/RoboTwin-Clean",
            "lerobot_v21_robotwin",
            prefix="RoboTwin-Clean",
        )
        + _scan_subdatasets(
            "/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/StarVLA_YJH/Benchmark/RoboTwin-Randomized",
            "lerobot_v21_robotwin",
            prefix="RoboTwin-Randomized",
        )
    ),

    # RDT-yhq ALOHA dataset (296 tasks)
    "rdt_yhq": [
        ("airpods_on_second_layer", 1.0, "lerobot_v21_aloha"),
        ("airpods_on_third_layer", 1.0, "lerobot_v21_aloha"),
        ("arrange_fruits_by_size", 1.0, "lerobot_v21_aloha"),
        ("arrange_word_2024", 1.0, "lerobot_v21_aloha"),
        ("arrange_word_23", 1.0, "lerobot_v21_aloha"),
        ("arrange_word_ABCD", 1.0, "lerobot_v21_aloha"),
        ("arrange_word_ALOHA", 1.0, "lerobot_v21_aloha"),
        ("arrange_word_DEEP", 1.0, "lerobot_v21_aloha"),
        ("arrange_word_LEARNING", 1.0, "lerobot_v21_aloha"),
        ("arrange_word_THU", 1.0, "lerobot_v21_aloha"),
        ("arrange_word_TURING", 1.0, "lerobot_v21_aloha"),
        ("choose_toy_by_color", 1.0, "lerobot_v21_aloha"),
        ("choose_toy_by_size", 1.0, "lerobot_v21_aloha"),
        ("clean_pour_water", 1.0, "lerobot_v21_aloha"),
        ("close_glasses_box", 1.0, "lerobot_v21_aloha"),
        ("close_laptop", 1.0, "lerobot_v21_aloha"),
        ("close_laptop_2", 1.0, "lerobot_v21_aloha"),
        ("close_the_book", 1.0, "lerobot_v21_aloha"),
        ("cocktail_sunset_coconut", 1.0, "lerobot_v21_aloha"),
        ("cocktail_sunset_ice", 1.0, "lerobot_v21_aloha"),
        ("cocktail_sunset_lemon", 1.0, "lerobot_v21_aloha"),
        ("cocktail_sunset_orange", 1.0, "lerobot_v21_aloha"),
        ("cocktail_sunset_pineapple", 1.0, "lerobot_v21_aloha"),
        ("cocktail_sunset_red", 1.0, "lerobot_v21_aloha"),
        ("coffee_cup_left_PC", 1.0, "lerobot_v21_aloha"),
        ("coffee_cup_left_to_left", 1.0, "lerobot_v21_aloha"),
        ("coffee_cup_middle_to_left", 1.0, "lerobot_v21_aloha"),
        ("coffee_cup_middle_to_right", 1.0, "lerobot_v21_aloha"),
        ("coffee_cup_on_PC", 1.0, "lerobot_v21_aloha"),
        ("coffee_cup_right_PC", 1.0, "lerobot_v21_aloha"),
        ("coffee_cup_right_to_right", 1.0, "lerobot_v21_aloha"),
        ("collect_earphone", 1.0, "lerobot_v21_aloha"),
        ("collect_litter", 1.0, "lerobot_v21_aloha"),
        ("collect_pen", 1.0, "lerobot_v21_aloha"),
        ("connect_charging_cable", 1.0, "lerobot_v21_aloha"),
        ("cover_laptop", 1.0, "lerobot_v21_aloha"),
        ("cover_mirror_jsh", 1.0, "lerobot_v21_aloha"),
        ("cover_spary_lid", 1.0, "lerobot_v21_aloha"),
        ("draw_char_A", 1.0, "lerobot_v21_aloha"),
        ("draw_char_Z", 1.0, "lerobot_v21_aloha"),
        ("draw_check_mark", 1.0, "lerobot_v21_aloha"),
        ("draw_cross", 1.0, "lerobot_v21_aloha"),
        ("draw_line", 1.0, "lerobot_v21_aloha"),
        ("draw_rectangle", 1.0, "lerobot_v21_aloha"),
        ("draw_triangle", 1.0, "lerobot_v21_aloha"),
        ("exchange_cup_position", 1.0, "lerobot_v21_aloha"),
        ("flip_bottle", 1.0, "lerobot_v21_aloha"),
        ("flip_calendar_page", 1.0, "lerobot_v21_aloha"),
        ("flowering_fake", 1.0, "lerobot_v21_aloha"),
        ("flowering_fake_2", 1.0, "lerobot_v21_aloha"),
        ("fold_towel", 1.0, "lerobot_v21_aloha"),
        ("gather_paper_ball", 1.0, "lerobot_v21_aloha"),
        ("get_cold_water", 1.0, "lerobot_v21_aloha"),
        ("get_hot_water", 1.0, "lerobot_v21_aloha"),
        ("grab_stick_into_bottle", 1.0, "lerobot_v21_aloha"),
        ("handover_pan", 1.0, "lerobot_v21_aloha"),
        ("hang_bag_on_hook", 1.0, "lerobot_v21_aloha"),
        ("headphone_left_phone", 1.0, "lerobot_v21_aloha"),
        ("hook_earring", 1.0, "lerobot_v21_aloha"),
        ("hook_keys", 1.0, "lerobot_v21_aloha"),
        ("insert_blue_shoe", 1.0, "lerobot_v21_aloha"),
        ("insert_book_shelf", 1.0, "lerobot_v21_aloha"),
        ("insert_cable_charger", 1.0, "lerobot_v21_aloha"),
        ("insert_cube_slot", 1.0, "lerobot_v21_aloha"),
        ("insert_pen_container", 1.0, "lerobot_v21_aloha"),
        ("insert_phone_2", 1.0, "lerobot_v21_aloha"),
        ("insert_phone_charger", 1.0, "lerobot_v21_aloha"),
        ("kiwi_toward_upside_down_mouse", 1.0, "lerobot_v21_aloha"),
        ("match_the_counter", 1.0, "lerobot_v21_aloha"),
        ("math_cube_with_fruit_jsh", 1.0, "lerobot_v21_aloha"),
        ("math_cube_with_rag_jsh", 1.0, "lerobot_v21_aloha"),
        ("messy_pour_water", 1.0, "lerobot_v21_aloha"),
        ("messy_stack_tomatoes", 1.0, "lerobot_v21_aloha"),
        ("move_foam_mahjong_from_right_to_left", 1.0, "lerobot_v21_aloha"),
        ("move_object_near_mirror", 1.0, "lerobot_v21_aloha"),
        ("open_laptop", 1.0, "lerobot_v21_aloha"),
        ("pack_durant_shirt", 1.0, "lerobot_v21_aloha"),
        ("pack_nvidia_shirt", 1.0, "lerobot_v21_aloha"),
        ("pack_pants", 1.0, "lerobot_v21_aloha"),
        ("pack_soccer_shirt", 1.0, "lerobot_v21_aloha"),
        ("paper_on_second_layer", 1.0, "lerobot_v21_aloha"),
        ("paper_on_third_layer", 1.0, "lerobot_v21_aloha"),
        ("pick_apple_into_bag", 1.0, "lerobot_v21_aloha"),
        ("pick_backpack_from_chair_to_desk", 1.0, "lerobot_v21_aloha"),
        ("pick_badminton", 1.0, "lerobot_v21_aloha"),
        ("pick_book_into_backpack", 1.0, "lerobot_v21_aloha"),
        ("pick_bottle_get_water", 1.0, "lerobot_v21_aloha"),
        ("pick_chips_to_box", 1.0, "lerobot_v21_aloha"),
        ("pick_different_doll", 1.0, "lerobot_v21_aloha"),
        ("pick_key_to_basket", 1.0, "lerobot_v21_aloha"),
        ("pick_kiwifruit_into_bag", 1.0, "lerobot_v21_aloha"),
        ("pick_larger_value", 1.0, "lerobot_v21_aloha"),
        ("pick_magiccube_into_bag", 1.0, "lerobot_v21_aloha"),
        ("pick_marker_pen_from_cup", 1.0, "lerobot_v21_aloha"),
        ("pick_mask", 1.0, "lerobot_v21_aloha"),
        ("pick_orange_into_bag", 1.0, "lerobot_v21_aloha"),
        ("pick_out_clip", 1.0, "lerobot_v21_aloha"),
        ("pick_pen", 1.0, "lerobot_v21_aloha"),
        ("pick_pen_on_notebook", 1.0, "lerobot_v21_aloha"),
        ("pick_place_pineapple", 1.0, "lerobot_v21_aloha"),
        ("pick_place_ver_water_bottle", 1.0, "lerobot_v21_aloha"),
        ("pick_place_water_bottle", 1.0, "lerobot_v21_aloha"),
        ("pick_potato_chip_packaging_near_apple", 1.0, "lerobot_v21_aloha"),
        ("pick_power_bank", 1.0, "lerobot_v21_aloha"),
        ("pick_same_color_clip", 1.0, "lerobot_v21_aloha"),
        ("pick_staples_into_stapler", 1.0, "lerobot_v21_aloha"),
        ("pick_tape_into_bag", 1.0, "lerobot_v21_aloha"),
        ("pick_the_different_dice", 1.0, "lerobot_v21_aloha"),
        ("pick_tomato_to_desk", 1.0, "lerobot_v21_aloha"),
        ("pick_up_letter_c", 1.0, "lerobot_v21_aloha"),
        ("pick_up_the_pen", 1.0, "lerobot_v21_aloha"),
        ("pin_into_second_layer", 1.0, "lerobot_v21_aloha"),
        ("place_cube_in_the_center", 1.0, "lerobot_v21_aloha"),
        ("place_cube_in_the_center_2", 1.0, "lerobot_v21_aloha"),
        ("place_marker", 1.0, "lerobot_v21_aloha"),
        ("place_object_on_mirror", 1.0, "lerobot_v21_aloha"),
        ("place_octopus_upright", 1.0, "lerobot_v21_aloha"),
        ("place_phone", 1.0, "lerobot_v21_aloha"),
        ("playmahjong", 1.0, "lerobot_v21_aloha"),
        ("plug_charger", 1.0, "lerobot_v21_aloha"),
        ("pour_clip_paper_box", 1.0, "lerobot_v21_aloha"),
        ("pour_water_4", 1.0, "lerobot_v21_aloha"),
        ("pour_water_5", 1.0, "lerobot_v21_aloha"),
        ("pour_water_6", 1.0, "lerobot_v21_aloha"),
        ("pour_water_between_cup", 1.0, "lerobot_v21_aloha"),
        ("pour_water_bottle2cup_clean", 1.0, "lerobot_v21_aloha"),
        ("pour_water_bottle2cup_mess", 1.0, "lerobot_v21_aloha"),
        ("pour_water_can2cup", 1.0, "lerobot_v21_aloha"),
        ("pour_water_cup2jigger2glass", 1.0, "lerobot_v21_aloha"),
        ("pour_water_cup_changing_light", 1.0, "lerobot_v21_aloha"),
        ("pour_water_cup_full", 1.0, "lerobot_v21_aloha"),
        ("pour_water_cup_half", 1.0, "lerobot_v21_aloha"),
        ("pour_water_cup_little", 1.0, "lerobot_v21_aloha"),
        ("pour_water_cup_scene1", 1.0, "lerobot_v21_aloha"),
        ("pour_water_cup_scene2", 1.0, "lerobot_v21_aloha"),
        ("pour_water_cup_yellow_light", 1.0, "lerobot_v21_aloha"),
        ("pour_water_dark_2", 1.0, "lerobot_v21_aloha"),
        ("pour_water_dark_meeting_room", 1.0, "lerobot_v21_aloha"),
        ("pour_water_from_bottle_to_cup", 1.0, "lerobot_v21_aloha"),
        ("pour_water_g2k", 1.0, "lerobot_v21_aloha"),
        ("pour_water_glass_changing_light", 1.0, "lerobot_v21_aloha"),
        ("pour_water_jigger2cup", 1.0, "lerobot_v21_aloha"),
        ("pour_water_left_hand", 1.0, "lerobot_v21_aloha"),
        ("pour_water_meeting_room", 1.0, "lerobot_v21_aloha"),
        ("press_alcohol_sanitizing", 1.0, "lerobot_v21_aloha"),
        ("press_socket_button", 1.0, "lerobot_v21_aloha"),
        ("press_stapler", 1.0, "lerobot_v21_aloha"),
        ("prop_bottle", 1.0, "lerobot_v21_aloha"),
        ("pull_chair", 1.0, "lerobot_v21_aloha"),
        ("pull_paper_on_id_card_jsh", 1.0, "lerobot_v21_aloha"),
        ("pull_trash_can", 1.0, "lerobot_v21_aloha"),
        ("pull_wet_wipe", 1.0, "lerobot_v21_aloha"),
        ("push_add_chip", 1.0, "lerobot_v21_aloha"),
        ("push_chair", 1.0, "lerobot_v21_aloha"),
        ("push_container_back_to_chips_packaging", 1.0, "lerobot_v21_aloha"),
        ("push_max_chip", 1.0, "lerobot_v21_aloha"),
        ("push_min_chip", 1.0, "lerobot_v21_aloha"),
        ("push_mirror", 1.0, "lerobot_v21_aloha"),
        ("push_mirror_surface", 1.0, "lerobot_v21_aloha"),
        ("put_badminton_in_order", 1.0, "lerobot_v21_aloha"),
        ("put_cherry_bowl", 1.0, "lerobot_v21_aloha"),
        ("put_clothes_into_backpack", 1.0, "lerobot_v21_aloha"),
        ("put_cup_behind_laptop", 1.0, "lerobot_v21_aloha"),
        ("put_cup_front_laptop", 1.0, "lerobot_v21_aloha"),
        ("put_cup_left_laptop", 1.0, "lerobot_v21_aloha"),
        ("put_cup_right_laptop", 1.0, "lerobot_v21_aloha"),
        ("put_doll_on_coin", 1.0, "lerobot_v21_aloha"),
        ("put_fluffy_octopus_into_bowl", 1.0, "lerobot_v21_aloha"),
        ("put_glasses_into_box", 1.0, "lerobot_v21_aloha"),
        ("put_glue_into_box_yellow_light", 1.0, "lerobot_v21_aloha"),
        ("put_ice_scoop_in_box_messy_table", 1.0, "lerobot_v21_aloha"),
        ("put_ice_scoop_in_box_tidy_table", 1.0, "lerobot_v21_aloha"),
        ("put_kiwi_on_doll", 1.0, "lerobot_v21_aloha"),
        ("put_kiwifruit_into_box_yellow_light", 1.0, "lerobot_v21_aloha"),
        ("put_marker_pen_in_cup", 1.0, "lerobot_v21_aloha"),
        ("put_mouse_on_pad", 1.0, "lerobot_v21_aloha"),
        ("put_object_into_cabinet", 1.0, "lerobot_v21_aloha"),
        ("put_object_into_drawer", 1.0, "lerobot_v21_aloha"),
        ("put_orange_into_box", 1.0, "lerobot_v21_aloha"),
        ("put_orange_into_microwave", 1.0, "lerobot_v21_aloha"),
        ("put_orange_paperbox", 1.0, "lerobot_v21_aloha"),
        ("put_paper_cup_jsh", 1.0, "lerobot_v21_aloha"),
        ("put_paper_upside_down", 1.0, "lerobot_v21_aloha"),
        ("put_rag_on_laptop_blue", 1.0, "lerobot_v21_aloha"),
        ("put_rag_on_laptop_orange", 1.0, "lerobot_v21_aloha"),
        ("put_rag_on_laptop_purple", 1.0, "lerobot_v21_aloha"),
        ("put_rag_on_laptop_yellow", 1.0, "lerobot_v21_aloha"),
        ("put_rubbish_in_carton", 1.0, "lerobot_v21_aloha"),
        ("put_snack_into_microwave", 1.0, "lerobot_v21_aloha"),
        ("put_spitballs_in_carton", 1.0, "lerobot_v21_aloha"),
        ("put_spitballs_in_carton_2", 1.0, "lerobot_v21_aloha"),
        ("put_sponge_in_box", 1.0, "lerobot_v21_aloha"),
        ("put_sponge_into_drawer", 1.0, "lerobot_v21_aloha"),
        ("putmajiangintobox", 1.0, "lerobot_v21_aloha"),
        ("redirect_magiccube", 1.0, "lerobot_v21_aloha"),
        ("robot_dog_backward", 1.0, "lerobot_v21_aloha"),
        ("robot_dog_forward", 1.0, "lerobot_v21_aloha"),
        ("robot_dog_forward_2", 1.0, "lerobot_v21_aloha"),
        ("roll_dice", 1.0, "lerobot_v21_aloha"),
        ("roll_dice_and_compare", 1.0, "lerobot_v21_aloha"),
        ("search_wipe_glass_water", 1.0, "lerobot_v21_aloha"),
        ("shake_glass", 1.0, "lerobot_v21_aloha"),
        ("shovel_ice_into_cup", 1.0, "lerobot_v21_aloha"),
        ("sort_fruits", 1.0, "lerobot_v21_aloha"),
        ("sort_toy_by_color", 1.0, "lerobot_v21_aloha"),
        ("sort_toy_by_size", 1.0, "lerobot_v21_aloha"),
        ("sortmahjong", 1.0, "lerobot_v21_aloha"),
        ("spell_big", 1.0, "lerobot_v21_aloha"),
        ("spell_car", 1.0, "lerobot_v21_aloha"),
        ("spell_cool", 1.0, "lerobot_v21_aloha"),
        ("spell_cup", 1.0, "lerobot_v21_aloha"),
        ("spell_dog", 1.0, "lerobot_v21_aloha"),
        ("spell_love", 1.0, "lerobot_v21_aloha"),
        ("spell_sky", 1.0, "lerobot_v21_aloha"),
        ("spell_sun", 1.0, "lerobot_v21_aloha"),
        ("spell_yes", 1.0, "lerobot_v21_aloha"),
        ("stack_can", 1.0, "lerobot_v21_aloha"),
        ("stack_cap_on_apple_yellow_light", 1.0, "lerobot_v21_aloha"),
        ("stack_cup", 1.0, "lerobot_v21_aloha"),
        ("stack_letter_block_easy", 1.0, "lerobot_v21_aloha"),
        ("stack_letter_brick", 1.0, "lerobot_v21_aloha"),
        ("stack_letter_cube_2", 1.0, "lerobot_v21_aloha"),
        ("stack_magiccube_on_glass_yellow_light", 1.0, "lerobot_v21_aloha"),
        ("stack_tomato_cans", 1.0, "lerobot_v21_aloha"),
        ("stack_tomatoes_2", 1.0, "lerobot_v21_aloha"),
        ("stack_tomatoes_3", 1.0, "lerobot_v21_aloha"),
        ("stack_tomatoes_meeting_room", 1.0, "lerobot_v21_aloha"),
        ("stir_liquid_inside_cup", 1.0, "lerobot_v21_aloha"),
        ("straw_cafe_cup_l", 1.0, "lerobot_v21_aloha"),
        ("straw_cafe_cup_m", 1.0, "lerobot_v21_aloha"),
        ("straw_cafe_cup_s", 1.0, "lerobot_v21_aloha"),
        ("stuckmahjong", 1.0, "lerobot_v21_aloha"),
        ("swipe_the_letter", 1.0, "lerobot_v21_aloha"),
        ("take_clothes_into_backpack", 1.0, "lerobot_v21_aloha"),
        ("take_off_lid_of_ice_box", 1.0, "lerobot_v21_aloha"),
        ("take_out_chips_from_packaging", 1.0, "lerobot_v21_aloha"),
        ("take_out_glassescase_from_bag", 1.0, "lerobot_v21_aloha"),
        ("take_out_pen_lid", 1.0, "lerobot_v21_aloha"),
        ("take_out_spary_lid", 1.0, "lerobot_v21_aloha"),
        ("take_out_tissue", 1.0, "lerobot_v21_aloha"),
        ("take_sponge_from_box", 1.0, "lerobot_v21_aloha"),
        ("takemahjong", 1.0, "lerobot_v21_aloha"),
        ("takeout_insert_usb", 1.0, "lerobot_v21_aloha"),
        ("tape_on_second_layer", 1.0, "lerobot_v21_aloha"),
        ("tape_on_third_layer", 1.0, "lerobot_v21_aloha"),
        ("tian_fold_cloth", 1.0, "lerobot_v21_aloha"),
        ("tian_open_box_pick_bottle", 1.0, "lerobot_v21_aloha"),
        ("tian_open_laptop_press_keyboard", 1.0, "lerobot_v21_aloha"),
        ("tian_put_red_book_top", 1.0, "lerobot_v21_aloha"),
        ("turn_off_light", 1.0, "lerobot_v21_aloha"),
        ("turn_on_light", 1.0, "lerobot_v21_aloha"),
        ("turn_over_foam_mahjong", 1.0, "lerobot_v21_aloha"),
        ("unplug_charger", 1.0, "lerobot_v21_aloha"),
        ("unplug_data_cable", 1.0, "lerobot_v21_aloha"),
        ("unscrew_cap", 1.0, "lerobot_v21_aloha"),
        ("unwind_charging_cable", 1.0, "lerobot_v21_aloha"),
        ("unzip_the_bag", 1.0, "lerobot_v21_aloha"),
        ("up_down_cube", 1.0, "lerobot_v21_aloha"),
        ("wash_cup_1", 1.0, "lerobot_v21_aloha"),
        ("wash_cup_2", 1.0, "lerobot_v21_aloha"),
        ("wash_cup_3", 1.0, "lerobot_v21_aloha"),
        ("wash_cup_5", 1.0, "lerobot_v21_aloha"),
        ("wash_cup_6", 1.0, "lerobot_v21_aloha"),
        ("wash_water", 1.0, "lerobot_v21_aloha"),
        ("water_plant_1", 1.0, "lerobot_v21_aloha"),
        ("water_plant_2", 1.0, "lerobot_v21_aloha"),
        ("water_plant_4", 1.0, "lerobot_v21_aloha"),
        ("wipe_cherry", 1.0, "lerobot_v21_aloha"),
        ("wipe_desk", 1.0, "lerobot_v21_aloha"),
        ("wipe_glass_water", 1.0, "lerobot_v21_aloha"),
        ("wipe_laptop_water", 1.0, "lerobot_v21_aloha"),
        ("wipe_table_b2f", 1.0, "lerobot_v21_aloha"),
        ("wipe_table_f2b", 1.0, "lerobot_v21_aloha"),
        ("wipe_table_l2r", 1.0, "lerobot_v21_aloha"),
        ("wipe_table_l2r_avoid", 1.0, "lerobot_v21_aloha"),
        ("wipe_table_r2l", 1.0, "lerobot_v21_aloha"),
        ("wipe_table_r2l_avoid", 1.0, "lerobot_v21_aloha"),
        ("wipe_whitboard_wdy", 1.0, "lerobot_v21_aloha"),
        ("wipe_whiteboard", 1.0, "lerobot_v21_aloha"),
        ("write_board_0", 1.0, "lerobot_v21_aloha"),
        ("write_board_1", 1.0, "lerobot_v21_aloha"),
        ("write_board_1+1", 1.0, "lerobot_v21_aloha"),
        ("write_board_10", 1.0, "lerobot_v21_aloha"),
        ("write_board_11", 1.0, "lerobot_v21_aloha"),
        ("write_board_12", 1.0, "lerobot_v21_aloha"),
        ("write_board_2", 1.0, "lerobot_v21_aloha"),
        ("write_board_20", 1.0, "lerobot_v21_aloha"),
        ("write_board_3", 1.0, "lerobot_v21_aloha"),
        ("write_board_4", 1.0, "lerobot_v21_aloha"),
        ("write_board_5", 1.0, "lerobot_v21_aloha"),
        ("write_board_6", 1.0, "lerobot_v21_aloha"),
        ("write_board_7", 1.0, "lerobot_v21_aloha"),
        ("write_board_8", 1.0, "lerobot_v21_aloha"),
        ("write_board_9", 1.0, "lerobot_v21_aloha"),
        ("write_hi_whiteboard", 1.0, "lerobot_v21_aloha"),
        ("zip_the_bag", 1.0, "lerobot_v21_aloha"),
    ],

    "multi_robot": [
        ("LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot", 0.15, "libero_franka"),
        ("LEROBOT_LIBERO_DATA/libero_object_no_noops_1.0.0_lerobot", 0.15, "libero_franka"),
        ("LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot", 0.15, "libero_franka"),
        ("LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot", 0.15, "libero_franka"),
        ("OXE_LEROBOT_DATASET/bridge_orig_1.0.0_lerobot", 2, "oxe_bridge"),
        ("OXE_LEROBOT_DATASET/fractal20220817_data_0.1.0_lerobot", 2, "oxe_rt1"),
    ],
}


# add robocasa to DATASET_NAMED_MIXTURES
PREFIX="PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
multi_robot = [(f"{PREFIX}/{name}", weight*0.2, robot_type) for name, weight, robot_type in DATASET_NAMED_MIXTURES["fourier_gr1_unified_1000"]]
DATASET_NAMED_MIXTURES["multi_robot"].extend(multi_robot)


# add robotwin to DATASET_NAMED_MIXTURES
PREFIX="RoboTwin-Clean"
multi_robot = [(f"{PREFIX}/{name}", weight*0.4, robot_type) for name, weight, robot_type in DATASET_NAMED_MIXTURES["robotwin"]]
DATASET_NAMED_MIXTURES["multi_robot"].extend(multi_robot)


# ===== RDT-yhq FineGrainedInstruction mixtures =====
# 5 tasks without FineGrainedInstruction annotations
_RDT_YHQ_NO_FG = {"pack_pants", "pin_into_second_layer", "robot_dog_forward_2", "roll_dice", "roll_dice_and_compare"}

# FG entries: same tasks but with _fg suffix on robot_type to trigger FG loading
_rdt_yhq_fg_entries = [
    (name, 1.0, "lerobot_v21_aloha_fg")
    for name, w, rt in DATASET_NAMED_MIXTURES["rdt_yhq"]
    if name not in _RDT_YHQ_NO_FG
]

# FGOnly: only FG-annotated data
DATASET_NAMED_MIXTURES["rdt_yhq_FGOnly"] = _rdt_yhq_fg_entries

# FG1_1: FG weight 1.0, normal weight 1.0 (1:1 ratio)
DATASET_NAMED_MIXTURES["rdt_yhq_FG1_1"] = (
    [(n, 1.0, rt) for n, w, rt in _rdt_yhq_fg_entries] +
    [(n, 1.0, rt) for n, w, rt in DATASET_NAMED_MIXTURES["rdt_yhq"]]
)

# FG2_1: FG weight 2.0, normal weight 1.0 (2:1 ratio)
DATASET_NAMED_MIXTURES["rdt_yhq_FG2_1"] = (
    [(n, 2.0, rt) for n, w, rt in _rdt_yhq_fg_entries] +
    [(n, 1.0, rt) for n, w, rt in DATASET_NAMED_MIXTURES["rdt_yhq"]]
)

# FG4_1: FG weight 4.0, normal weight 1.0 (4:1 ratio)
DATASET_NAMED_MIXTURES["rdt_yhq_FG4_1"] = (
    [(n, 4.0, rt) for n, w, rt in _rdt_yhq_fg_entries] +
    [(n, 1.0, rt) for n, w, rt in DATASET_NAMED_MIXTURES["rdt_yhq"]]
)


# ===== Aloha Multi-Dataset Mixture (7 groups, all 14-dim joint+gripper) =====
# data_root_dir should be set to the common parent:
#   /mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt
# data_name entries use relative paths from there.

_ALOHA_MULTI_LEROBOT = "VLA_Data/Lerobot_v21"

# Group 1: RDT-yhq (reuse existing rdt_yhq entries, prefix with subdir)
_aloha_multi_rdt = [
    (f"{_ALOHA_MULTI_LEROBOT}/RDT-yhq/{name}", w, rt)
    for name, w, rt in DATASET_NAMED_MIXTURES["rdt_yhq"]
]

# Group 2: RoboCOIN CM14
_aloha_multi_cm14 = [
    (f"{_ALOHA_MULTI_LEROBOT}/RoboCOIN/{t}", 1.0, "robocoin_cm14") for t in [
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
] + [
    (f"{_ALOHA_MULTI_LEROBOT}/RoboCOIN_add1201/{t}", 1.0, "robocoin_cm14") for t in [
        "Cobot_Magic_cap_the_pen_a", "Cobot_Magic_classification_of_fruits_and_vegetables_a",
        "Cobot_Magic_drawer_storage_mineral_water", "Cobot_Magic_open_the_shoebox",
        "Cobot_Magic_plate_storage_toy", "Cobot_Magic_vase_storage_flower",
    ]
]

# Group 3: RoboCOIN CM26
_aloha_multi_cm26 = [
    (f"{_ALOHA_MULTI_LEROBOT}/RoboCOIN/{t}", 1.0, "robocoin_cm26") for t in [
        "Cobot_Magic_clean_blackboard", "Cobot_Magic_cube_reset",
        "Cobot_Magic_mobile_cube", "Cobot_Magic_mobile_cube_blackboard",
        "Cobot_Magic_move_beverage", "Cobot_Magic_move_plate",
        "Cobot_Magic_move_the_ball", "Cobot_Magic_move_the_ball_interference",
        "Cobot_Magic_place_square_pyramid", "Cobot_Magic_pour_drink",
        "Cobot_Magic_pour_water_bottle", "Cobot_Magic_prepare_breakfast",
        "Cobot_Magic_pull_zipper", "Cobot_Magic_pushing_magnet",
        "Cobot_Magic_twist_bottle_cap", "Cobot_Magic_water_bottle_storage",
    ]
] + [
    (f"{_ALOHA_MULTI_LEROBOT}/RoboCOIN_add1201/{t}", 1.0, "robocoin_cm26") for t in [
        "Cobot_Magic_cut_banana", "Cobot_Magic_desktop_organization",
        "Cobot_Magic_food_packaging", "Cobot_Magic_make_fruit_salad",
        "Cobot_Magic_make_hamburger", "Cobot_Magic_move_the_ball_and_the_cube_block",
        "Cobot_Magic_place_the_cube_block", "Cobot_Magic_pour_water_a",
        "Cobot_Magic_take_down_the_cube_block",
    ]
]

# Group 4: RoboCOIN Split Aloha (26-dim, same config as CM26)
_aloha_multi_split = [
    (f"{_ALOHA_MULTI_LEROBOT}/RoboCOIN/{t}", 1.0, "robocoin_cm26") for t in [
        "Split_aloha_basket_storage_banana", "Split_aloha_basket_storage_bread",
        "Split_aloha_basket_storage_egg_yolk_pastry", "Split_aloha_basket_storage_long_bread",
        "Split_aloha_basket_storage_orange", "Split_aloha_basket_storage_peach",
        "Split_aloha_plate_storage", "Split_aloha_pour_rice",
        "Split_aloha_scoop_coffee_beans", "Split_aloha_stack_baskets",
        "Split_aloha_stir_coffee", "Split_aloha_wipe_table",
        "Split_aloha_wipe_the_table", "Split_aloha_zip_up_the_document_bag",
    ]
] + [
    (f"{_ALOHA_MULTI_LEROBOT}/RoboCOIN_add1201/{t}", 1.0, "robocoin_cm26") for t in [
        "Split_aloha_fold_the_pants", "Split_aloha_pour_tea",
    ]
]

# Group 5: RoboMIND V1.0 Agilex 3RGB (auto-scan from benchmark dirs)
_aloha_multi_rmv1 = (
    _scan_subdatasets(
        f"/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/{_ALOHA_MULTI_LEROBOT}/RoboMindV1.0/benchmark1_0_compressed/agilex_3rgb",
        "robomind_v1_agilex",
    ) + _scan_subdatasets(
        f"/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/{_ALOHA_MULTI_LEROBOT}/RoboMindV1.0/benchmark1_1_compressed/agilex_3rgb",
        "robomind_v1_agilex",
    )
)
# Prefix with relative path from common root
_aloha_multi_rmv1_b0 = [
    (f"{_ALOHA_MULTI_LEROBOT}/RoboMindV1.0/benchmark1_0_compressed/agilex_3rgb/{t}", w, rt)
    for t, w, rt in _scan_subdatasets(
        f"/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/{_ALOHA_MULTI_LEROBOT}/RoboMindV1.0/benchmark1_0_compressed/agilex_3rgb",
        "robomind_v1_agilex",
    )
]
_aloha_multi_rmv1_b1 = [
    (f"{_ALOHA_MULTI_LEROBOT}/RoboMindV1.0/benchmark1_1_compressed/agilex_3rgb/{t}", w, rt)
    for t, w, rt in _scan_subdatasets(
        f"/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/{_ALOHA_MULTI_LEROBOT}/RoboMindV1.0/benchmark1_1_compressed/agilex_3rgb",
        "robomind_v1_agilex",
    )
]
_aloha_multi_rmv1_full = _aloha_multi_rmv1_b0 + _aloha_multi_rmv1_b1

# Group 6: RoboMIND V2.0 Agilex
# Exclude tasks with only 1 camera (need 3 cameras for training)
_ROBOMIND_V2_EXCLUDE = {"flip_cup_and_place_on_plate_with_arms", "place_corn_on_plate_with_both_arms", "stack_green_and_blue_bowls_with_arms"}
_aloha_multi_rmv2 = [
    (f"RoboMindV2.0-Lerobot/agilex/{t}", w, rt)
    for t, w, rt in _scan_subdatasets(
        "/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/RoboMindV2.0-Lerobot/agilex",
        "robomind_v2_agilex",
    )
    if t not in _ROBOMIND_V2_EXCLUDE
]

# Group 7: RoboMIND V2.0 Agilex Mobile
_aloha_multi_rmv2_mobile = [
    (f"RoboMindV2.0-Lerobot/agilex_mobile/{t}", w, rt)
    for t, w, rt in _scan_subdatasets(
        "/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/RoboMindV2.0-Lerobot/agilex_mobile",
        "robomind_v2_mobile",
    )
]

DATASET_NAMED_MIXTURES["aloha_multi_mix"] = (
    _aloha_multi_rdt
    + _aloha_multi_cm14
    + _aloha_multi_cm26
    + _aloha_multi_split
    + _aloha_multi_rmv1_full
    + _aloha_multi_rmv2
    + _aloha_multi_rmv2_mobile
)

# Small test mixture: 3 tasks per group for quick validation
_aloha_test_base = (
    _aloha_multi_rdt[:3]
    + _aloha_multi_cm14[:3]
    + _aloha_multi_cm26[:3]
    + _aloha_multi_split[:3]
    + _aloha_multi_rmv1_full[:3]
    + _aloha_multi_rmv2[:3]
    + _aloha_multi_rmv2_mobile[:3]
)
DATASET_NAMED_MIXTURES["aloha_multi_test"] = _aloha_test_base


# ===== Aloha Multi-Dataset FineGrained Instruction mixtures =====
# 7 tasks without FineGrainedInstruction annotations
_ALOHA_MULTI_NO_FG = {
    "VLA_Data/Lerobot_v21/RDT-yhq/pack_pants",
    "VLA_Data/Lerobot_v21/RDT-yhq/pin_into_second_layer",
    "VLA_Data/Lerobot_v21/RDT-yhq/robot_dog_forward_2",
    "VLA_Data/Lerobot_v21/RDT-yhq/roll_dice",
    "VLA_Data/Lerobot_v21/RDT-yhq/roll_dice_and_compare",
    "RoboMindV2.0-Lerobot/agilex/place_mug_in_plate_and_rotate_handle_right",
    "RoboMindV2.0-Lerobot/agilex/rotate_and_support_tennis_can_with_arms",
}

# Small FG1_1 test mixture (3 tasks per group)
_aloha_test_fg = [
    (name, 1.0, rt + "_fg")
    for name, w, rt in _aloha_test_base
    if name not in _ALOHA_MULTI_NO_FG
]
DATASET_NAMED_MIXTURES["aloha_multi_FG1_1_test"] = (
    [(n, 1.0, rt) for n, w, rt in _aloha_test_fg]
    + [(n, 1.0, rt) for n, w, rt in _aloha_test_base]
)

# FG entries: same data_name but robot_type gets _fg suffix to trigger FG loading
_aloha_multi_fg_entries = [
    (name, 1.0, rt + "_fg")
    for name, w, rt in DATASET_NAMED_MIXTURES["aloha_multi_mix"]
    if name not in _ALOHA_MULTI_NO_FG
]

# FGOnly: only FG-annotated data (588 tasks)
DATASET_NAMED_MIXTURES["aloha_multi_FGOnly"] = _aloha_multi_fg_entries

# FG1_1: FG weight 1.0, normal weight 1.0 (1:1 ratio)
DATASET_NAMED_MIXTURES["aloha_multi_FG1_1"] = (
    [(n, 1.0, rt) for n, w, rt in _aloha_multi_fg_entries]
    + [(n, 1.0, rt) for n, w, rt in DATASET_NAMED_MIXTURES["aloha_multi_mix"]]
)

# FG2_1: FG weight 2.0, normal weight 1.0 (2:1 ratio)
DATASET_NAMED_MIXTURES["aloha_multi_FG2_1"] = (
    [(n, 2.0, rt) for n, w, rt in _aloha_multi_fg_entries]
    + [(n, 1.0, rt) for n, w, rt in DATASET_NAMED_MIXTURES["aloha_multi_mix"]]
)

# FG4_1: FG weight 4.0, normal weight 1.0 (4:1 ratio)
DATASET_NAMED_MIXTURES["aloha_multi_FG4_1"] = (
    [(n, 4.0, rt) for n, w, rt in _aloha_multi_fg_entries]
    + [(n, 1.0, rt) for n, w, rt in DATASET_NAMED_MIXTURES["aloha_multi_mix"]]
)