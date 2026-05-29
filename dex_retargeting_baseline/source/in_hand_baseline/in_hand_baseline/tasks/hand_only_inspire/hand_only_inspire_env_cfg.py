import torch

import isaaclab.envs.mdp as mdp
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_unitree_g1_inspire_hand_env_cfg import (
    PickPlaceG1InspireFTPEnvCfg,
)

from in_hand_baseline.retargeting import MediaPipeXRDeviceCfg, RightHandInspireRetargeterCfg
from in_hand_baseline.retargeting.joint_names import INSPIRE_HAND_ACTION_JOINT_NAMES


RIGHT_ARM_SUPPORT_POSE = {
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 1.2,
    "right_wrist_yaw_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
}


@configclass
class HandOnlyActionsCfg:
    hand_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=INSPIRE_HAND_ACTION_JOINT_NAMES,
        scale=1.0,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class HandOnlyInspireEnvCfg(PickPlaceG1InspireFTPEnvCfg):
    actions: HandOnlyActionsCfg = HandOnlyActionsCfg()

    idle_action = torch.zeros(24)

    def __post_init__(self):
        self.decimation = 6
        self.episode_length_s = 20.0

        self.sim.dt = 1 / 120
        self.sim.render_interval = 2

        self.scene.robot.init_state.joint_pos.update(RIGHT_ARM_SUPPORT_POSE)

        self.teleop_devices = DevicesCfg(
            devices={
                "mediapipe": MediaPipeXRDeviceCfg(
                    sim_device=self.sim.device,
                    retargeters=[
                        RightHandInspireRetargeterCfg(
                            sim_device=self.sim.device,
                            hand_joint_names=INSPIRE_HAND_ACTION_JOINT_NAMES,
                        )
                    ],
                )
            }
        )
