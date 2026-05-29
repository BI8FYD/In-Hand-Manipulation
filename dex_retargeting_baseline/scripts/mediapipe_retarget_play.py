# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Run right-hand MediaPipe-to-Inspire retargeting in the hand-only IsaacLab environment."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Right-hand MediaPipe retargeting for the Inspire hand baseline.")
parser.add_argument("--task", type=str, default="Isaac-HandOnly-Inspire-Baseline-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--video_path", type=str, default=None, help="Path to a video file. Uses camera if omitted.")
parser.add_argument("--camera_index", type=int, default=0, help="Camera index used when --video_path is omitted.")
parser.add_argument("--right_hand_label", type=str, default="Right", help="MediaPipe handedness label to retarget.")
parser.add_argument("--no_loop_video", action="store_true", default=False, help="Do not loop video files.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument(
    "--disable_pinocchio_preload",
    action="store_true",
    default=False,
    help="Do not preload pinocchio before launching Isaac Sim. DexPilot normally needs the preload.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not args_cli.disable_pinocchio_preload:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

PROJECT_SOURCE_DIR = Path(__file__).resolve().parents[1] / "source" / "in_hand_baseline"
if str(PROJECT_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SOURCE_DIR))

import gymnasium as gym
import torch

from isaaclab.devices.teleop_device_factory import create_teleop_device

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import in_hand_baseline.tasks  # noqa: F401


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    device_cfg = env_cfg.teleop_devices.devices["mediapipe"]
    device_cfg.video_path = args_cli.video_path
    device_cfg.camera_index = args_cli.camera_index
    device_cfg.loop_video = not args_cli.no_loop_video
    device_cfg.right_hand_label = args_cli.right_hand_label

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    teleop_interface = create_teleop_device("mediapipe", env_cfg.teleop_devices.devices)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    print(f"[INFO]: Using teleop device: {teleop_interface}")

    env.reset()
    teleop_interface.reset()

    try:
        while simulation_app.is_running():
            action = teleop_interface.advance()
            actions = action.repeat(env.num_envs, 1)
            with torch.inference_mode():
                env.step(actions)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
