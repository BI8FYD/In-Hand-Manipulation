import torch

import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_unitree_g1_inspire_hand_env_cfg import (
    PickPlaceG1InspireFTPEnvCfg,
)


@configclass
class HandOnlyActionsCfg:
    hand_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["[LR]_.*_joint"],
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

        self.teleop_devices = {}
