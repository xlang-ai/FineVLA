"""
Smoke test: load one sample from each dataset group in aloha_multi_mix
to verify modality.json, DataConfig, and 14-dim action extraction work correctly.
"""
import sys, types
sys.path.insert(0, "/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt/StarVLA_YJH")

# Stub out accelerate to avoid import errors in __init__.py
acc_stub = types.ModuleType("accelerate")
acc_stub.logging = types.ModuleType("accelerate.logging")
acc_stub.logging.get_logger = lambda *a, **kw: __import__("logging").getLogger("stub")
sys.modules["accelerate"] = acc_stub
sys.modules["accelerate.logging"] = acc_stub.logging

from pathlib import Path
from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG, EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset

DATA_ROOT = Path("/mnt/cpfs_m6_29eu38p1/data/shared/Group-m6/tongzai.hxt")

TEST_CASES = [
    {
        "name": "Group 1: RDT-yhq (14-dim)",
        "data_name": "VLA_Data/Lerobot_v21/RDT-yhq/airpods_on_second_layer",
        "robot_type": "lerobot_v21_aloha",
    },
    {
        "name": "Group 2: RoboCOIN CM14 (14-dim)",
        "data_name": "VLA_Data/Lerobot_v21/RoboCOIN/Cobot_Magic_box_storage_chopsticks",
        "robot_type": "robocoin_cm14",
    },
    {
        "name": "Group 3: RoboCOIN CM26 -> 14-dim",
        "data_name": "VLA_Data/Lerobot_v21/RoboCOIN/Cobot_Magic_clean_blackboard",
        "robot_type": "robocoin_cm26",
    },
    {
        "name": "Group 4: Split Aloha (26-dim -> 14-dim)",
        "data_name": "VLA_Data/Lerobot_v21/RoboCOIN/Split_aloha_basket_storage_banana",
        "robot_type": "robocoin_cm26",
    },
    {
        "name": "Group 5: RoboMIND V1 Agilex (56-dim -> 14-dim)",
        "data_name": "VLA_Data/Lerobot_v21/RoboMindV1.0/benchmark1_0_compressed/agilex_3rgb/10_packplate",
        "robot_type": "robomind_v1_agilex",
    },
    {
        "name": "Group 6: RoboMIND V2 Agilex (28-dim -> 14-dim)",
        "data_name": "RoboMindV2.0-Lerobot/agilex/arrange_blocks_and_place_orange_in_center_with_arms",
        "robot_type": "robomind_v2_agilex",
    },
    {
        "name": "Group 7: RoboMIND V2 Mobile (41-dim -> 14-dim)",
        "data_name": "RoboMindV2.0-Lerobot/agilex_mobile/align_and_place_blue_bins_from_left_to_right",
        "robot_type": "robomind_v2_mobile",
    },
]


class FakeDataCfg:
    """Minimal data_cfg stub for testing."""
    def get(self, key, default=None):
        if key == "video_backend":
            return "pyav"
        if key == "lerobot_version":
            return "v2.0"
        if key == "include_state":
            return False
        return default


def test_single(case):
    name = case["name"]
    data_name = case["data_name"]
    robot_type = case["robot_type"]
    dataset_path = DATA_ROOT / data_name

    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"  path: {dataset_path}")
    print(f"  robot_type: {robot_type}")

    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality_config = data_config.modality_config()
    transforms = data_config.transform()

    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    else:
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]

    data_cfg = FakeDataCfg()

    ds = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend="pyav",
        data_cfg=data_cfg,
    )

    print(f"  Dataset loaded: {ds.dataset_name}, {len(ds)} steps")

    sample = ds[0]
    print(f"  Sample keys: {list(sample.keys())}")
    print(f"  action shape: {sample['action'].shape}")
    print(f"  action dtype: {sample['action'].dtype}")
    print(f"  num images: {len(sample['image'])}")
    print(f"  image size: {sample['image'][0].size}")
    print(f"  language: {sample['lang'][:80]}...")

    assert sample["action"].shape[1] == 14, f"Expected 14 action dims, got {sample['action'].shape[1]}"
    print(f"  [PASS] action_dim = 14")
    return True


def main():
    print("Smoke test: Aloha Multi-Dataset Mix")
    print(f"Testing {len(TEST_CASES)} dataset groups\n")

    results = {}
    for case in TEST_CASES:
        try:
            ok = test_single(case)
            results[case["name"]] = "PASS"
        except Exception as e:
            print(f"  [FAIL] {e}")
            import traceback
            traceback.print_exc()
            results[case["name"]] = f"FAIL: {e}"

    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, status in results.items():
        print(f"  {status:6s} | {name}")


if __name__ == "__main__":
    main()
