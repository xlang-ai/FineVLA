# CanonicalizeAndClean

Converts heterogeneous robot state/action representations into a unified 80-dimensional vector format, then filters episodes by quality criteria.

## Pipeline

```
Raw dataset parquets + modality.json
        │
        ▼
  convert_unified.py              Step 1: Convert to 80-dim unified representation
  (UnifyJointAction)              Output: unified_output/*.parquet + unified_meta.json
        │
        ▼
  filter_by_state_action_frame.py Step 2: Quality filtering (frames + task + L2)
  (cal_distance.py)               Output: {dataset}_filter_report.json
```

## Unified 80-dim Vector Layout

All state/action data is mapped to a canonical 80-dimensional vector with a binary mask indicating active dimensions:

| Range | Field | Description |
|-------|-------|-------------|
| `[0:7]` | `left_joint` | Left arm joint positions (7-DoF) |
| `[7:16]` | `left_eef` | Left end-effector pose (xyz + rotation) |
| `[16:17]` | `left_gripper` | Left gripper |
| `[17:29]` | `left_hand` | Left hand joints (dexterous hands) |
| `[29:36]` | `right_joint` | Right arm joint positions (7-DoF) |
| `[36:45]` | `right_eef` | Right end-effector pose (xyz + rotation) |
| `[45:46]` | `right_gripper` | Right gripper |
| `[46:58]` | `right_hand` | Right hand joints (dexterous hands) |
| `[58:80]` | `reserved` | Reserved for future use |

Supported rotation types: `abs_quat` (xyzw), `abs_wxyz`, `abs_euler` (xyz), `abs_rotvec`.

## Quality Filters

Three checks are applied per episode:

1. **Frame count**: Episodes below a dataset-specific minimum frame threshold are flagged
2. **Task completeness**: Episodes with empty or missing task descriptions are flagged
3. **State-action L2 divergence**: Range-normalized L2 distance between state and action trajectories; episodes exceeding a per-dataset threshold are flagged as having misaligned state/action data

## File Structure

```
CanonicalizeAndClean/
├── config.py                       Dataset-specific thresholds and field mappings
├── convert_unified.py              Unified representation converter (multiprocessing)
├── filter_by_state_action_frame.py Combined quality filter (main entry point)
├── cal_distance.py                 L2 distance computation module
└── utils/
    └── UnifyJointAction.py         Core transform: heterogeneous -> 80-dim vector
```

## Usage

```bash
export VLA_DATA_ROOT="/path/to/your/Lerobot_v21"

# Step 1: Convert to unified representation
python convert_unified.py $VLA_DATA_ROOT/BC_Z --episodes 5

# Step 2: Quality filtering
python filter_by_state_action_frame.py $VLA_DATA_ROOT/BC_Z

# With options
python filter_by_state_action_frame.py $VLA_DATA_ROOT/BC_Z \
    --episodes 100 --threshold 1.5 --force-reconvert --plot
```

## Dependencies

```bash
pip install -r requirements.txt
```

```
numpy
pandas
pyarrow
matplotlib
tqdm
```
