import os

import numpy as np
from gymnasium.utils import EzPickle

from gymnasium_robotics.envs.shadow_dexterous_hand import MujocoManipulateEnv

ASSETS_DIR = os.path.abspath(os.path.join(os.path.curdir, "assets"))  # move to central place
MANIPULATE_SCISSORS_XML = os.path.join(ASSETS_DIR, "shadow_dexterous_hand", "manipulate_block_touch_sensors.xml")

DEFAULT_CAMERA_CONFIG = {
    "distance": 1,  # 0.5
    "azimuth": 0.0,  # 55.0
    "elevation": -15.0,  # -25.0
    # "lookat": np.array([1, 0.96, 0.14]),
    "lookat": np.array([0.5, 0.75, 0.5]),
}


class MujocoHandScissorsEnv(MujocoManipulateEnv, EzPickle):
    def __init__(
            self,
            target_position="random",
            target_rotation="xyz",
            reward_type="sparse",
            **kwargs,
    ):
        MujocoManipulateEnv.__init__(
            self,
            model_path=MANIPULATE_SCISSORS_XML,
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
        EzPickle.__init__(self, target_position, target_rotation, reward_type, **kwargs)

    def _set_action(self, action):
        super()._set_action(action)
        # ctrlrange = self.model.actuator_ctrlrange
        # actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.0
        #
        # if self.relative_control:
        #     actuation_center = np.zeros_like(action)
        #     for i in range(self.data.ctrl.shape[0]):
        #         actuation_center[i] = self.data.get_joint_qpos(
        #             self.model.actuator_names[i].replace(":A_", ":")
        #         )
        #     for joint_name in ["FF", "MF", "RF", "LF"]:
        #         act_idx = self.model.actuator_name2id(f"robot0:A_{joint_name}J1")
        #         actuation_center[act_idx] += self.data.get_joint_qpos(
        #             f"robot0:{joint_name}J0"
        #         )
        # else:
        #     actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.0
        # self.data.ctrl[:] = actuation_center + action * actuation_range
        # self.data.ctrl[:] = np.clip(self.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])

    def compute_reward(self, achieved_goal, goal, info):
        _reward = super().compute_reward(achieved_goal, goal, info)
        return 100

    def compute_terminated(self, achieved_goal, desired_goal, info):
        """All the available environments are currently continuing tasks and non-time dependent. The objective is to reach the goal for an indefinite period of time."""
        return False

    def compute_truncated(self, achievec_goal, desired_goal, info):
        """The environments will be truncated only if setting a time limit with max_steps which will automatically wrap the environment in a gymnasium TimeLimit wrapper."""
        return False
