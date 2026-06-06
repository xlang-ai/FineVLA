"""
Purpose of this code:
Convert action and state into a unified representation, focusing on two aspects of unification:
1. Rotation representation unification; 2. Absolute/relative position unification.

1. Rotation representation unification
- Only EEF requires this unification, since EEF rotation has 4 representations (rotvec, quat, quat(wxyz), euler) that need to be unified;
- Joint is always joint angles, no rotation representation unification needed.

2. Absolute/relative position unification
- State is always in absolute position
- Action may be in abs/delta/rel representations that need to be unified:
  - abs: absolute spatial position
  - rel: position relative to state[0]
  - delta: position relative to state[t-1]

3. Assumptions
- State is always abs
"""



from abc import ABC, abstractmethod
from typing import Any, List, Dict, Tuple

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
import numpy as np
import json
from pathlib import Path
from scipy.spatial.transform import Rotation as R


# such class ,input the original state and action keys, and output the unified state and action keys
class ModalityTransform(BaseModel, ABC):
    apply_to: list[str] = Field(..., description="The keys to apply the transform to.")
    training: bool = Field(default=True)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        ...

############# Unified State Action Transforms ###############

UNIFIED_STATE_ACTION_DIM = 80
UNIFIED_STATE_ACTION_INDICES = {
    "left_joint": [0, 7], # 7
    "left_eef": [7, 16], # 9 — considering rotation may be unified to 6D rot representation, reserving 3 xyz + 6 rot dims
    "left_gripper": [16, 17], # 1 — open/close
    "left_hand": [17, 29], # 12 — dexterous hand, considering up to 12 DoF for now
    "right_joint": [29, 36], # 7
    "right_eef": [36, 45], # 9
    "right_gripper": [45, 46], # 1 — open/close
    "right_hand": [46, 58], # 12
    "reserved": [58, 80], # 22 — reserved
}


class UnifiedStateActionTransform(ModalityTransform):
    """
    Transforms state and action to UNIFIED_STATE_ACTION_DIM vectors.
    1. Each part of the original state & action will be transformed to the specified target representations 
    (target_joint_state_type, target_eef_state_type, target_joint_action_type, target_eef_action_type)
    2. The converted representations will be filled to the target vector.
    """
    apply_to: list[str] = Field(..., description="All state and action keys to be transformed. Must include all state_keys + action_keys.")
    modality_path: Path = Field(..., description="Path to the modality json file.")

    target_joint_state_type: str = Field(..., description="Target type for joint state, e.g. 'abs_joint'")
    target_eef_state_type: str = Field(..., description="Target type for eef state, e.g. 'abs_rotvec'")
    target_joint_action_type: str = Field(..., description="Target type for joint action, e.g. 'abs_joint'")
    target_eef_action_type: str = Field(..., description="Target type for eef action, e.g. 'abs_rotvec'")
    '''
    Type explanation: currently these are combinations of abs/delta/rel and joint/rotvec/quat/wxyz/euler.
    1. For the first part (whether absolute representation):
        - All states are always abs
        - The original action in modality json may be abs or delta (a_t expressed as a transform on top of s_t)
        - The target action representation may be abs, delta (a_t as transform on s_t), or rel (a_t as transform on s_0)
    2. For the second part, left_eef and right_eef use rotvec/euler/quat/wxyz to represent the rotation type;
    rotation transforms are needed when converting between delta/rel and abs.
    All other parts are joint and can be directly added/subtracted numerically.
    '''

    _modality_config: dict | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Load modality JSON from modality_path at construction time."""
        with open(self.modality_path, encoding="utf-8") as f:
            self._modality_config = json.load(f)

    # ---------- EEF processing functions ----------
    # Parse format: relative/absolute_rotation_representation_type
    def _eef_parse_type(self, type_str: str) -> tuple[str, str]:
        """Return (abs|delta|rel, joint|rotvec|euler|quat|wxyz)."""
        parts = type_str.split("_", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid eef type: {type_str}")
        return parts[0], parts[1]

    # Convert eef array to pos and scipy rot; returns pos (xyz) and rot (scipy Rotation object)
    def _eef_numpy_to_pos_rot(self, arr: np.ndarray, type_str: str) -> Tuple[np.ndarray, R]:
        """(N, D) eef array -> pos (N,3), scipy Rotation. Assumes layout xyz then rotation."""
        assert arr.ndim == 2
        pos = arr[:, :3]
        rot_part = arr[:, 3:]
        _, rot_fmt = self._eef_parse_type(type_str)
        if rot_fmt == "rotvec":
            rot = R.from_rotvec(rot_part)
        elif rot_fmt == "quat":
            rot = R.from_quat(rot_part) # xyzw
        elif rot_fmt == "wxyz":
            rot = R.from_quat(rot_part, scalar_first=True) # wxyz
        elif rot_fmt == "euler":
            rot = R.from_euler("xyz", rot_part)
        else:
            raise ValueError(f"Unknown eef rotation format: {rot_fmt}")
        return pos, rot

    # Convert pos and scipy rot to eef array
    def _eef_pos_rot_to_numpy(self, pos: np.ndarray, rot: R, target_type: str) -> np.ndarray:
        """pos (N,3), Rotation -> (N, D). Rotation representation use target_type."""
        _, rot_fmt = self._eef_parse_type(target_type)
        assert pos.ndim == 2
        if rot_fmt == "rotvec":
            rv = rot.as_rotvec()
            assert rv.ndim == 2
            return np.concatenate([pos, rv], axis=-1)
        if rot_fmt == "quat":
            q = rot.as_quat()
            assert q.ndim == 2
            return np.concatenate([pos, q], axis=-1)
        if rot_fmt == "wxyz":
            q = rot.as_quat()
            assert q.ndim == 2
            return np.concatenate([pos, q[..., 3:4], q[..., :3]], axis=-1) # xyzw -> wxyz
        if rot_fmt == "euler":
            eu = rot.as_euler("xyz")
            assert eu.ndim == 2
            return np.concatenate([pos, eu], axis=-1)
        raise ValueError(f"Unknown target rotation format: {rot_fmt}")

    # Fill arr into the unified state/action template at slot [lo:hi]
    def _fill_slot(self, unified: np.ndarray, lo: int, hi: int, arr: np.ndarray, mask: np.ndarray = None) -> None:
        """Fill unified[:, lo:lo+d] with arr. Assert arr length <= slot size (hi - lo).
        If mask is provided, set mask[:, lo:lo+d] to True to indicate valid values."""
        assert arr.ndim == 2
        d = arr.shape[-1]
        assert d <= hi - lo, f"Array dimension {d} exceeds slot range [{lo}:{hi}]"
        unified[:, lo : lo + d] = arr.astype(unified.dtype)
        if mask is not None:
            assert mask.shape == unified.shape, f"Mask shape {mask.shape} must match unified shape {unified.shape}"
            mask[:, lo : lo + d] = True


    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Input: state and action data; modality describes state and action (including original key, start, end, type).
        start and end indicate the start and end positions of this field in unified_state_action_dim.

        Input data: keys like state.left_joint (16,7), action.left_joint (16,7), etc.
        modality_config: state/action each have keys like left_joint, right_eef, reserved.waist with original_key, start, end, type.
        Output: remove all state.* / action.* in apply_to, add:
            - state.unified (N, UNIFIED_DIM): unified state representation
            - action.unified (N, UNIFIED_DIM): unified action representation
            - mask.state (N, UNIFIED_DIM): bool array indicating valid state values
            - mask.action (N, UNIFIED_DIM): bool array indicating valid action values
        """
        cfg = self._modality_config
        state_cfg = cfg.get("state", {})
        action_cfg = cfg.get("action", {})

        ### 1. Classify all keys in apply_to as state/action, and separate left_eef, right_eef, and others
        def is_eef_key(suffix: str) -> bool:
            return suffix == "left_eef" or suffix == "right_eef" or suffix.startswith("left_eef.") or suffix.startswith("right_eef.")

        state_left_eef_parts: List[Tuple[str, np.ndarray, str]] = [] # [(original key, array, type)]
        state_right_eef_parts: List[Tuple[str, np.ndarray, str]] = []
        state_others: Dict[str, Tuple[np.ndarray, str]] = {} # {original key: (array, type)}
        action_left_eef_parts: List[Tuple[str, np.ndarray, str]] = []
        action_right_eef_parts: List[Tuple[str, np.ndarray, str]] = []
        action_others: Dict[str, Tuple[np.ndarray, str]] = {}
        state_chunk_size, action_chunk_size = None, None

        #1. For each field, classify as state/action and left_eef/right_eef/others
        for key in self.apply_to:
            if key not in data:
                raise ValueError(f"Key {key} not found in data")
            val = data[key]
            chunk = val.shape[0]
            
            if key.startswith("state."):
                suffix = key[6:]
                if state_chunk_size is None:
                    state_chunk_size = chunk
                assert state_chunk_size == chunk, f"state chunk size mismatch: {key}"
                type_str = state_cfg[suffix]["type"]
                if is_eef_key(suffix): # If it's eef data, process left_eef and right_eef separately
                    arr = np.asarray(val, dtype=np.float32)
                    if suffix.startswith("left_eef"):
                        state_left_eef_parts.append((suffix, arr, type_str))
                    else:
                        state_right_eef_parts.append((suffix, arr, type_str))
                else:
                    state_others[suffix] = (np.asarray(val, dtype=np.float32), type_str)
            elif key.startswith("action."):
                suffix = key[7:]
                if action_chunk_size is None:
                    action_chunk_size = chunk
                assert action_chunk_size == chunk, f"action chunk size mismatch: {key}"
                type_str = action_cfg[suffix]["type"]
                if is_eef_key(suffix):
                    arr = np.asarray(val, dtype=np.float32)
                    if suffix.startswith("left_eef"):
                        action_left_eef_parts.append((suffix, arr, type_str))
                    else:
                        action_right_eef_parts.append((suffix, arr, type_str))
                else:
                    action_others[suffix] = (np.asarray(val, dtype=np.float32), type_str)
            else:
                raise ValueError(f"Key {key} is not a state or action key")

        assert state_chunk_size is not None and action_chunk_size is not None, "state or action chunk size is not set"

        # Concatenate left_eef and right_eef parts in the order of apply_to
        def concat_eef_parts(parts: List[Tuple[str, np.ndarray, str]]) -> Tuple[np.ndarray, str]:
            if len(parts) == 0:
                return None, None
            arrays = [p[1] for p in parts] # numpy arrays
            type_str = parts[-1][2] # left/right eef representation type, using the last occurrence of left_eef/right_eef key
            return np.concatenate(arrays, axis=-1), type_str

        state_left_eef_arr, state_left_eef_type = concat_eef_parts(state_left_eef_parts)
        state_right_eef_arr, state_right_eef_type = concat_eef_parts(state_right_eef_parts)
        action_left_eef_arr, action_left_eef_type = concat_eef_parts(action_left_eef_parts)
        action_right_eef_arr, action_right_eef_type = concat_eef_parts(action_right_eef_parts)

        ### 2. Template
        unified_state = np.zeros((state_chunk_size, UNIFIED_STATE_ACTION_DIM), dtype=np.float32)
        unified_action = np.zeros((action_chunk_size, UNIFIED_STATE_ACTION_DIM), dtype=np.float32)
        mask_state = np.zeros((state_chunk_size, UNIFIED_STATE_ACTION_DIM), dtype=bool)
        mask_action = np.zeros((action_chunk_size, UNIFIED_STATE_ACTION_DIM), dtype=bool)

        ### 3. Convert left_eef and right_eef parts, fill into the template
        eef_slots = [
            ("left_eef", state_left_eef_arr, state_left_eef_type, action_left_eef_arr, action_left_eef_type),
            ("right_eef", state_right_eef_arr, state_right_eef_type, action_right_eef_arr, action_right_eef_type),
        ]
        for name, s_arr, s_type, a_arr, a_type in eef_slots:
            lo, hi = UNIFIED_STATE_ACTION_INDICES[name]
            dim = hi - lo
            if s_arr is not None:
                ## 3.1 Unify eef state rotation representation, fill into template
                assert s_type.startswith("abs"), "State eef type must be absolute"
                pos_s, rot_s = self._eef_numpy_to_pos_rot(s_arr, s_type)
                out_s = self._eef_pos_rot_to_numpy(pos_s, rot_s, self.target_eef_state_type)
                self._fill_slot(unified_state, lo, hi, out_s, mask_state)
            if a_arr is not None:
                # Simplified check: if eef action is used, eef state must also exist
                assert s_arr is not None, f"Get eef action {name}, but missing eef state {name}"
                # pos_s, rot_s already computed above
                pos_a, rot_a = self._eef_numpy_to_pos_rot(a_arr, a_type)
                
                ## 3.2 Based on the unified target type, convert eef action to the target coordinate frame
                # If the original eef action is delta (transform on state in world frame), first convert to abs using state
                if a_type.startswith("delta"):
                    pos_a = pos_s + pos_a
                    rot_a = rot_a * rot_s
                else:
                    assert a_type.startswith("abs"), "Unknown eef action type"
                # Given the absolute eef state and action, compute the target eef action representation
                if self.target_eef_action_type.startswith("rel"):
                    # Convert to transform relative to the first state's coordinate frame
                    delta_p = pos_a - pos_s[0]
                    pos_a = rot_s[0].inv().apply(delta_p) # relies on scipy broadcast
                    rot_a = rot_s[0].inv() * rot_a
                elif self.target_eef_action_type.startswith("delta"):
                    # Convert to transform relative to current state in world frame
                    assert state_chunk_size == action_chunk_size, "state and action chunk size must be the same for delta action conversion"
                    pos_a = pos_a - pos_s
                    rot_a = rot_a * rot_s.inv()
                else:
                    assert self.target_eef_action_type.startswith("abs"), "Unknown target eef action type"
                    # pos_a and rot_a are already abs, nothing to do
                
                ## 3.3 Unify eef action rotation representation, fill into template
                out_a = self._eef_pos_rot_to_numpy(pos_a, rot_a, self.target_eef_action_type)
                self._fill_slot(unified_action, lo, hi, out_a, mask_action)

        ### 4. Convert others parts (all treated as joint semantics), fill into template
        ## 4.1 For state, both original and target are always abs_joint, fill directly
        current_state_slots_lo = {k: UNIFIED_STATE_ACTION_INDICES[k][0] for k in UNIFIED_STATE_ACTION_INDICES.keys()}
        for key, (arr, type_str) in state_others.items():
            assert type_str.startswith("abs"), "State type must be absolute"
            assert self.target_joint_state_type.startswith("abs"), "Target state type must be absolute"
            name = key.split(".")[0]
            _, hi = UNIFIED_STATE_ACTION_INDICES[name]
            lo = current_state_slots_lo[name]
            self._fill_slot(unified_state, lo, hi, arr, mask_state)
            current_state_slots_lo[name] += arr.shape[-1]
        ## 4.2 For action, first convert to abs_joint, then convert to the target format
        current_action_slots_lo = {k: UNIFIED_STATE_ACTION_INDICES[k][0] for k in UNIFIED_STATE_ACTION_INDICES.keys()}
        for key, (arr, type_str) in action_others.items():
            # raw action type -> abs joint
            if type_str.startswith("delta"):
                assert state_chunk_size == action_chunk_size, "state and action chunk size must be the same for delta action conversion"
                assert key in state_others, f"Get delta action {key}, but missing state {key}"
                arr = arr + state_others[key][0]
            else:
                assert type_str.startswith("abs"), "Unknown action type"
            # abs joint -> target action type
            if self.target_joint_action_type.startswith("rel"):
                assert key in state_others, f"rel target requires state for {key}"
                arr = arr - state_others[key][0][0]
            elif self.target_joint_action_type.startswith("delta"):
                assert state_chunk_size == action_chunk_size, "state and action chunk size must be the same for delta action conversion"
                assert key in state_others, f"delta target requires state for {key}"
                arr = arr - state_others[key][0]
            else:
                assert self.target_joint_action_type.startswith("abs"), "Unknown target action type"
                # arr is already abs, nothing to do
            # Fill into template
            name = key.split(".")[0]
            _, hi = UNIFIED_STATE_ACTION_INDICES[name]
            lo = current_action_slots_lo[name]
            self._fill_slot(unified_action, lo, hi, arr, mask_action)
            current_action_slots_lo[name] += arr.shape[-1]

        ### 5. Remove original state and action, replace with unified ones
        for key in list(data.keys()):
            if key in self.apply_to:
                del data[key]
        for k in list(data.keys()):
            if k.startswith("state.") or k.startswith("action."):
                raise AssertionError(f"apply_to should have removed all state/action keys; remaining: {k}")
        data["state.unified"] = unified_state
        data["action.unified"] = unified_action
        data["mask.state"] = mask_state
        data["mask.action"] = mask_action
        return data
