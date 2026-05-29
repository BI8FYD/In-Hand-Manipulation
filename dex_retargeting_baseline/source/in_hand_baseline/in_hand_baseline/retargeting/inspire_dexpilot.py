"""DexPilot retargeter wrapper for right-hand-only Inspire hand control."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field

import numpy as np
import torch
import yaml
from dex_retargeting.retargeting_config import RetargetingConfig
from scipy.spatial.transform import Rotation as R

from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.openxr.common import HAND_JOINT_NAMES
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

from .joint_names import INSPIRE_HAND_ACTION_JOINT_NAMES


_HAND_JOINTS_INDEX = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25]

_OPERATOR2MANO_RIGHT = np.array(
    [
        [0, -1, 0],
        [-1, 0, 0],
        [0, 0, -1],
    ]
)

_RIGHT_HAND_URDF_PATH = f"{ISAACLAB_NUCLEUS_DIR}/Mimic/G1_inspire_assets/retarget_inspire_white_right_hand.urdf"
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "unitree_hand_right_dexpilot.yml")


class RightHandInspireRetargeter(RetargeterBase):
    """Retarget OpenXR-style right-hand tracking data to the 24-dim Inspire action space.

    The baseline environment still exposes both Inspire hands to keep the action contract
    compatible with the existing 24-dim hand-only task. This retargeter fills only the
    right-hand joints and leaves all left-hand commands at zero.
    """

    def __init__(self, cfg: RightHandInspireRetargeterCfg):
        super().__init__(cfg)
        self._sim_device = cfg.sim_device
        self._hand_joint_names = cfg.hand_joint_names
        self._hold_last_on_missing = cfg.hold_last_on_missing
        self._dex_right_hand = RetargetingConfig.load_from_file(self._make_runtime_config()).build()
        self._right_dof_names = self._dex_right_hand.optimizer.robot.dof_joint_names
        self._last_action = np.zeros(len(self._hand_joint_names), dtype=np.float32)

    def retarget(self, data: dict) -> torch.Tensor:
        right_hand_poses = data.get(DeviceBase.TrackingTarget.HAND_RIGHT)
        action = np.zeros(len(self._hand_joint_names), dtype=np.float32)

        if right_hand_poses is None:
            if self._hold_last_on_missing:
                action = self._last_action.copy()
            return torch.tensor(action, dtype=torch.float32, device=self._sim_device)

        right_hand_q = self._compute_right(right_hand_poses)
        for joint_name, joint_pos in zip(self._right_dof_names, right_hand_q):
            action[self._hand_joint_names.index(joint_name)] = joint_pos

        self._last_action = action.copy()
        return torch.tensor(action, dtype=torch.float32, device=self._sim_device)

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.HAND_TRACKING]

    def _make_runtime_config(self) -> str:
        local_urdf_path = retrieve_file_path(_RIGHT_HAND_URDF_PATH, force_download=False)
        with open(_CONFIG_PATH) as file:
            config = yaml.safe_load(file)
        config["retargeting"]["urdf_path"] = local_urdf_path

        tmp_file = tempfile.NamedTemporaryFile(
            mode="w", prefix="inspire_right_dexpilot_", suffix=".yml", delete=False
        )
        with tmp_file:
            yaml.safe_dump(config, tmp_file)
        return tmp_file.name

    def _compute_right(self, right_hand_poses: dict[str, np.ndarray]) -> np.ndarray:
        joint_pos = self._convert_hand_joints(right_hand_poses, _OPERATOR2MANO_RIGHT)
        ref_value = self._compute_ref_value(
            joint_pos,
            indices=self._dex_right_hand.optimizer.target_link_human_indices,
            retargeting_type=self._dex_right_hand.optimizer.retargeting_type,
        )
        with torch.enable_grad():
            with torch.inference_mode(False):
                return self._dex_right_hand.retarget(ref_value)

    @staticmethod
    def _convert_hand_joints(hand_poses: dict[str, np.ndarray], operator2mano: np.ndarray) -> np.ndarray:
        joint_position = np.zeros((21, 3))
        hand_joints = [hand_poses[name] for name in HAND_JOINT_NAMES]
        for i, hand_joint_index in enumerate(_HAND_JOINTS_INDEX):
            joint_position[i] = hand_joints[hand_joint_index][:3]

        joint_position = joint_position - joint_position[0:1, :]
        wrist_quat = hand_poses["wrist"][3:]
        wrist_rot = R.from_quat([wrist_quat[1], wrist_quat[2], wrist_quat[3], wrist_quat[0]]).as_matrix()
        return joint_position @ wrist_rot @ operator2mano

    @staticmethod
    def _compute_ref_value(joint_position: np.ndarray, indices: np.ndarray, retargeting_type: str) -> np.ndarray:
        if retargeting_type == "POSITION":
            return joint_position[indices, :]
        origin_indices = indices[0, :]
        task_indices = indices[1, :]
        return joint_position[task_indices, :] - joint_position[origin_indices, :]


@dataclass
class RightHandInspireRetargeterCfg(RetargeterCfg):
    """Configuration for right-hand Inspire DexPilot retargeting."""

    sim_device: str = "cpu"
    hand_joint_names: list[str] = field(default_factory=lambda: list(INSPIRE_HAND_ACTION_JOINT_NAMES))
    hold_last_on_missing: bool = True
    retargeter_type: type[RetargeterBase] = RightHandInspireRetargeter
