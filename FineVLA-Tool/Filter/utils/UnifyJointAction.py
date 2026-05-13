"""
【代码的作用】
将action 和state 的变成统一的表示，重点在于两个方面的统一：1.旋转表示的统一；2.绝对位置和相对位置的统一

1.旋转表示的统一
- 只有EEF 会需要有这种统一，因为EEF的旋转表示有4种（rotvec, quat, quat(wxyz), euler），需要统一成一种;
- joint 只会是关节角，没有旋转表示的统一问题

2.绝对位置和相对位置的统一
- state 一定是绝对位置
- action 可能是 abs/delta/rel 三种表示，需要统一成一种
  - abs: 绝对空间位置
  - rel：相对 state[0] 的相对位置
  - delta：相对 state[t-1] 的相对位置

3.一些assumption
- state 一定是 abs

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
    "left_eef": [7, 16], # 9 考虑旋转可能统一成6d rot表示，预留3xyz+6rot维
    "left_gripper": [16, 17], # 1 开合
    "left_hand": [17, 29], # 12 灵巧手先考虑12dof以内的，
    "right_joint": [29, 36], # 7
    "right_eef": [36, 45], # 9
    "right_gripper": [45, 46], # 1 开合
    "right_hand": [46, 58], # 12
    "reserved": [58, 80], # 22 预留
}


class UnifiedStateActionTransform(ModalityTransform):
    """
    Transforms state and action to UNIFIED_STATE_ACTION_DIM vectors.
    1. Each part of the original state & action will be transformed to the specified target representations 
    (target_joint_state_type, target_eef_state_type, target_joint_action_type, target_eef_action_type)
    2. The converted representations will be filled to the target vector.
    """
    apply_to: list[str] = Field(..., description="All state and action keys to be transformed. 实际需要给全部的state_keys+action_keys")
    modality_path: Path = Field(..., description="Path to the modality json file.")

    target_joint_state_type: str = Field(..., description="Target type for joint state, e.g. 'abs_joint'")
    target_eef_state_type: str = Field(..., description="Target type for eef state, e.g. 'abs_rotvec'")
    target_joint_action_type: str = Field(..., description="Target type for joint action, e.g. 'abs_joint'")
    target_eef_action_type: str = Field(..., description="Target type for eef action, e.g. 'abs_rotvec'")
    '''
    type说明：目前有abs/delta/rel_joint/rotvec/quat/wxyz/euler这些组合
    1. 对于第一部分（是否绝对表示）：
        - 所有state一定是abs的
        - modality json中原始的action可能是abs或delta（a_t表示成在s_t基础上的变换）
        - target type中目标的action表示可能是abs或delta（a_t表示成在s_t基础上的变换）或rel（a_t表示成在s_0基础上的变换）
    2. 对于第二部分，left_eef和right_eef使用rotvec/euler/quat/wxyz来表示旋转变换表达的类型，在delta/rel和abs之间互转时需要用到旋转变换
    其他部分都是joint，可以直接在数值上加减
    '''

    _modality_config: dict | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Load modality JSON from modality_path at construction time."""
        with open(self.modality_path, encoding="utf-8") as f:
            self._modality_config = json.load(f)

    # ---------- EEF processing functions ----------
    # 解析格式：相对/绝对_旋转表示类型
    def _eef_parse_type(self, type_str: str) -> tuple[str, str]:
        """Return (abs|delta|rel, joint|rotvec|euler|quat|wxyz)."""
        parts = type_str.split("_", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid eef type: {type_str}")
        return parts[0], parts[1]

    # 将eef array转成pos和scipy rot 返回时一个 pos（xyz） 和 rot（scipy 的Rotation对象）
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

    # 将pos和scipy rot转成eef array
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

    # 将arr填入统一state/action模板unified中的字段[lo:hi]
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
        输入：state 和 action 的数据，modality 是对state 和action 的说明（包含原始的key，start，end，type），
        start 和 end 表示的是 这个字段 在unified_state_action_dim 中的起始和结束位置

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

        ### 1. 将apply_to中所有的key区分为state/action, 并且把left_eef, right_eef, others区分开
        def is_eef_key(suffix: str) -> bool:
            return suffix == "left_eef" or suffix == "right_eef" or suffix.startswith("left_eef.") or suffix.startswith("right_eef.")

        state_left_eef_parts: List[Tuple[str, np.ndarray, str]] = [] # [(original key, array, type)]
        state_right_eef_parts: List[Tuple[str, np.ndarray, str]] = []
        state_others: Dict[str, Tuple[np.ndarray, str]] = {} # {original key: (array, type)}
        action_left_eef_parts: List[Tuple[str, np.ndarray, str]] = []
        action_right_eef_parts: List[Tuple[str, np.ndarray, str]] = []
        action_others: Dict[str, Tuple[np.ndarray, str]] = {}
        state_chunk_size, action_chunk_size = None, None

        #1. 对于字段，分为 state/action 和 left_eef/right_eef/others 三类
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
                if is_eef_key(suffix): #如果是 eef 的信息，则需要分别处理 left_eef 和 right_eef
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

        # 按照apply_to的顺序把left_eef和right_eef的部分拼接起来
        def concat_eef_parts(parts: List[Tuple[str, np.ndarray, str]]) -> Tuple[np.ndarray, str]:
            if len(parts) == 0:
                return None, None
            arrays = [p[1] for p in parts] #numpy的 数值
            type_str = parts[-1][2] # 左/右手eef的表示类型 以最后一次出现的left_eef/right_eef key为准
            return np.concatenate(arrays, axis=-1), type_str

        state_left_eef_arr, state_left_eef_type = concat_eef_parts(state_left_eef_parts)
        state_right_eef_arr, state_right_eef_type = concat_eef_parts(state_right_eef_parts)
        action_left_eef_arr, action_left_eef_type = concat_eef_parts(action_left_eef_parts)
        action_right_eef_arr, action_right_eef_type = concat_eef_parts(action_right_eef_parts)

        ### 2. 模板
        unified_state = np.zeros((state_chunk_size, UNIFIED_STATE_ACTION_DIM), dtype=np.float32)
        unified_action = np.zeros((action_chunk_size, UNIFIED_STATE_ACTION_DIM), dtype=np.float32)
        mask_state = np.zeros((state_chunk_size, UNIFIED_STATE_ACTION_DIM), dtype=bool)
        mask_action = np.zeros((action_chunk_size, UNIFIED_STATE_ACTION_DIM), dtype=bool)

        ### 3. 转换left_eef和right_eef部分，填入模板
        eef_slots = [
            ("left_eef", state_left_eef_arr, state_left_eef_type, action_left_eef_arr, action_left_eef_type),
            ("right_eef", state_right_eef_arr, state_right_eef_type, action_right_eef_arr, action_right_eef_type),
        ]
        for name, s_arr, s_type, a_arr, a_type in eef_slots:
            lo, hi = UNIFIED_STATE_ACTION_INDICES[name]
            dim = hi - lo
            if s_arr is not None:
                ## 3.1 统一eef state的旋转表示，填进模板
                assert s_type.startswith("abs"), "State eef type must be absolute"
                pos_s, rot_s = self._eef_numpy_to_pos_rot(s_arr, s_type)
                out_s = self._eef_pos_rot_to_numpy(pos_s, rot_s, self.target_eef_state_type)
                self._fill_slot(unified_state, lo, hi, out_s, mask_state)
            if a_arr is not None:
                # 简化判定：只要用到eef action，就必须有eef state
                assert s_arr is not None, f"Get eef action {name}, but missing eef state {name}"
                # 上面已经计算 #pos_s, rot_s = self._eef_numpy_to_pos_rot(s_arr, s_type)
                pos_a, rot_a = self._eef_numpy_to_pos_rot(a_arr, a_type)
                
                ## 3.2 根据统一的target type，把eef action转到目标的坐标系下
                # 如果原始eef action是delta的（世界坐标系下对state的变换），先根据state转成abs
                if a_type.startswith("delta"):
                    pos_a = pos_s + pos_a
                    rot_a = rot_a * rot_s
                else:
                    assert a_type.startswith("abs"), "Unknown eef action type"
                # 根据得到的绝对eef state和action，计算目标eef action的表示
                if self.target_eef_action_type.startswith("rel"):
                    # 转成在第一个state坐标系下的变换
                    delta_p = pos_a - pos_s[0]
                    pos_a = rot_s[0].inv().apply(delta_p) # 靠scipy broadcast
                    rot_a = rot_s[0].inv() * rot_a
                elif self.target_eef_action_type.startswith("delta"):
                    # 转成世界坐标系下相对当前state的变换
                    assert state_chunk_size == action_chunk_size, "state and action chunk size must be the same for delta action conversion"
                    pos_a = pos_a - pos_s
                    rot_a = rot_a * rot_s.inv()
                else:
                    assert self.target_eef_action_type.startswith("abs"), "Unknown target eef action type"
                    # 当前pos_a, rot_a已经是abs了，啥都不做
                
                ## 3.3 统一eef action的旋转表示，填进模板
                out_a = self._eef_pos_rot_to_numpy(pos_a, rot_a, self.target_eef_action_type)
                self._fill_slot(unified_action, lo, hi, out_a, mask_action)

        ### 4. 转换others部分(全部视为joint意义)，填入模板
        ## 4.1 对于state，原始和目标一定都是abs_joint，直接填入
        current_state_slots_lo = {k: UNIFIED_STATE_ACTION_INDICES[k][0] for k in UNIFIED_STATE_ACTION_INDICES.keys()}
        for key, (arr, type_str) in state_others.items():
            assert type_str.startswith("abs"), "State type must be absolute"
            assert self.target_joint_state_type.startswith("abs"), "Target state type must be absolute"
            name = key.split(".")[0]
            _, hi = UNIFIED_STATE_ACTION_INDICES[name]
            lo = current_state_slots_lo[name]
            self._fill_slot(unified_state, lo, hi, arr, mask_state)
            current_state_slots_lo[name] += arr.shape[-1]
        ## 4.2 对于action，先转到abs_joint，再转到目标格式
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
                # 当前arr已经是abs了，啥都不做
            # 填入模板
            name = key.split(".")[0]
            _, hi = UNIFIED_STATE_ACTION_INDICES[name]
            lo = current_action_slots_lo[name]
            self._fill_slot(unified_action, lo, hi, arr, mask_action)
            current_action_slots_lo[name] += arr.shape[-1]

        ### 5. 删除原本的state和action，改成统一的
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
