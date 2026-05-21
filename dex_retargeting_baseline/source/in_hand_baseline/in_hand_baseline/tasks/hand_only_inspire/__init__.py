import gymnasium as gym

gym.register(
    id="Isaac-HandOnly-Inspire-Baseline-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "in_hand_baseline.tasks.hand_only_inspire.hand_only_inspire_env_cfg:"
            "HandOnlyInspireEnvCfg"
        ),
    },
    disable_env_checker=True,
)
