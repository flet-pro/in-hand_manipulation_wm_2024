import numpy as np
from gymnasium import spaces, error
from gymnasium.utils import EzPickle

from gymnasium_robotics.envs.shadow_dexterous_hand.manipulate import quat_from_angle_and_axis
from gymnasium_robotics.utils import rotations

from envs.config import GRASP_OBJECT_ENV_XML, DEFAULT_CAMERA_CONFIG
from envs.generate_target_object import generate_target_object
from envs.robot_env import MujocoRobotEnv


# ASSETS_DIR = os.path.abspath(os.path.join(os.path.curdir, "assets"))  # move to central place
# MANIPULATE_SCISSORS_XML = os.path.join(ASSETS_DIR, "shadow_dexterous_hand", "manipulate_block_touch_sensors.xml")
#
# DEFAULT_CAMERA_CONFIG = {
#     "distance": 0.7,  # 0.5
#     "azimuth": 0.0,  # 55.0
#     "elevation": -35.0,  # -25.0
#     "lookat": np.array([1.3, 0.75, 0.45]),  # np.array([1, 0.96, 0.14])
# }

# DEFAULT_CAMERA_CONFIG = {
#     "distance": 0.5,
#     "azimuth": 55.0,
#     "elevation": -25.0,
#     "lookat": np.array([1, 0.96, 0.14]),
# }


def compute_pos_distance(goal_a, goal_b):
    assert goal_a.shape == (3,) and goal_b.shape == (3,)
    delta_pos = goal_a - goal_b
    d_pos = np.linalg.norm(delta_pos, axis=-1)
    return d_pos


class GraspObjectEnv(MujocoRobotEnv, EzPickle):
    """
        ## Description

        ## Additional Action Space
        The action space is a `Box(-1.0, 1.0, (20 + 6 = 26,), float32)`.
        * (Maybe not true) The control actions are absolute angular positions of the actuated joints (non-coupled).
        * The input of the control actions is set to a range between -1 and 1 by scaling the actual actuator angle ranges.
        * index 20 to 25 is the action target? for forearm sliders and hinges (6,)

        ## Observation Space
        The observation is a `goal-aware observation space`. It consists of a dictionary with information about the robot's joint and block states, as well as information about the goal. The dictionary consists of the following 3 keys:
        * `observation`: its value is an `ndarray` of shape `(30 + 30 + 9 = 69,)`.
        * It consists of kinematic information of the block (to be changed) object and finger joints.
        * index 0 to 5 is the positions info of the forearm sliders and hinges (6,)
        * index 6 to 29 is the positions info of the shadow hand itself (24,)
        * index 30 to 35 is the velocities info of the forearm sliders and hinges (6,)
        * index 36 to 59 is the velocities info of the shadow hand itself (24,)
        * index 60 to 68 is the positions info of three fingers (9,)

        ## Rewards (to be changed)

        ## Starting State (maybe to be changed)

        When the environment is reset the joints of the hand are initialized to their resting position with a 0 displacement. The blocks position and orientation are randomly selected. The initial position is set to `(x,y,z)=(1, 0.87, 0.2)` and an offset is added to each coordinate
        sampled from a normal distribution with 0 mean and 0.005 standard deviation.
        While the initial orientation is set to `(w,x,y,z)=(1,0,0,0)` and an axis is randomly selected depending on the environment variation to add an angle offset sampled from a uniform distribution with range `[-pi, pi]`.

        The target pose of the block is obtained by adding a random offset to the initial block pose. For the position the offset is sampled from a uniform distribution with range `[(x_min, x_max), (y_min,y_max), (z_min, z_max)] = [(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]`. The orientation
        offset is sampled from a uniform distribution with range `[-pi,pi]` and added to one of the Euler axis depending on the environment variation.


        ## Episode End (to be changed)

        The episode will be `truncated` when the duration reaches a total of `max_episode_steps` which by default is set to 50 timesteps.
        The episode is never `terminated` since the task is continuing with infinite horizon.

        ## Arguments

        To increase/decrease the maximum number of timesteps before the episode is `truncated` the `max_episode_steps` argument can be set at initialization. The default value is 50. For example, to increase the total number of timesteps to 100 make the environment as follows:

        ```python
        import gymnasium as gym
        import gymnasium_robotics

        gym.register_envs(gymnasium_robotics)

        env = gym.make('HandManipulateBlock-v1', max_episode_steps=100)
        ```

        The same applies for the other environment variations.

        ## Version History

        * v1: the environment depends on the newest [mujoco python bindings](https://mujoco.readthedocs.io/en/latest/python.html) maintained by the MuJoCo team in Deepmind.
        * v0: the environment depends on `mujoco_py` which is no longer maintained.
        """

    def __init__(
            self,
            random_init_pos=True,
            random_init_rot=True,
            target_position="random",
            target_rotation="z",
            reward_type="sparse",
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            initial_qpos=None,
            distance_threshold=0.01,
            rotation_threshold=0.1,
            n_substeps=20,
            relative_control=False,
            ignore_z_target_rotation=False,
            target_obj_name="random",
            **kwargs,
    ):
        self.relative_control = relative_control

        self.target_obj_name = generate_target_object(target_obj_name)

        self.target_position = target_position
        self.target_rotation = target_rotation
        self.target_position_range = target_position_range
        self.parallel_quats = [
            rotations.euler2quat(r) for r in rotations.get_parallel_rotations()
        ]
        self.randomize_initial_rotation = random_init_rot
        self.randomize_initial_position = random_init_pos
        self.distance_threshold = distance_threshold
        self.rotation_threshold = rotation_threshold
        self.reward_type = reward_type
        self.ignore_z_target_rotation = ignore_z_target_rotation

        assert self.target_position in ["ignore", "fixed", "random"]
        assert self.target_rotation in ["ignore", "fixed", "xyz", "z", "parallel"]
        initial_qpos = initial_qpos or {}

        super().__init__(
            model_path=GRASP_OBJECT_ENV_XML,
            n_substeps=n_substeps,
            initial_qpos=initial_qpos,
            # relative_control=relative_control,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            n_actions=20,
            **kwargs,
        )
        EzPickle.__init__(self, target_position, target_rotation, reward_type, **kwargs)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(26,), dtype="float32")

    def _set_action(self, action):
        # if action.shape[0] == 20:
        #     action = np.concatenate([action, np.zeros(6)])
        # hand_action, forearm_action = np.split(action, [20]) <- forearm_action should be at last

        ctrl_range = self.model.actuator_ctrlrange  # (26, 2)
        actuation_range = (ctrl_range[:, 1] - ctrl_range[:, 0]) / 2.0

        if self.relative_control:
            # fixme not implemented yet
            actuation_center = np.zeros_like(action)
            for i in range(self.data.ctrl.shape[0]):
                actuation_center[i] = self.data.get_joint_qpos(
                    self.model.actuator_names[i].replace(":A_", ":")
                )
            for joint_name in ["FF", "MF", "RF", "LF"]:
                act_idx = self.model.actuator_name2id(f"robot0:A_{joint_name}J1")
                actuation_center[act_idx] += self.data.get_joint_qpos(
                    f"robot0:{joint_name}J0"
                )
        else:
            actuation_center = (ctrl_range[:, 1] + ctrl_range[:, 0]) / 2.0
        self.data.ctrl[:] = actuation_center + action * actuation_range
        self.data.ctrl[:] = np.clip(self.data.ctrl, ctrl_range[:, 0], ctrl_range[:, 1])

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.data.set_joint_qpos(name, value)
        self._mujoco.mj_forward(self.model, self.data)

    # called at reset(), which is called before step()
    def _reset_sim(self):
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        self._mujoco.mj_forward(self.model, self.data)
        initial_qpos = self._utils.get_joint_qpos(
            self.model, self.data, "target:joint"
        ).copy()
        initial_pos, initial_quat = initial_qpos[:3], initial_qpos[3:]
        assert initial_qpos.shape == (7,)
        assert initial_pos.shape == (3,)
        assert initial_quat.shape == (4,)
        initial_qpos = None

        # Randomization initial rotation.
        if self.randomize_initial_rotation:
            if self.target_rotation == "z":
                angle = self.np_random.uniform(-np.pi/3, np.pi/3)  # todo this is changed to be easier
                self.scissors_angle = angle
                axis = np.array([0.0, 0.0, 1.0])
                offset_quat = quat_from_angle_and_axis(angle, axis)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)

            elif self.target_rotation == "parallel":
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = np.array([0.0, 0.0, 1.0])
                z_quat = quat_from_angle_and_axis(angle, axis)
                parallel_quat = self.parallel_quats[
                    self.np_random.integers(len(self.parallel_quats))
                ]
                offset_quat = rotations.quat_mul(z_quat, parallel_quat)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation in ["xyz", "ignore"]:
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = self.np_random.uniform(-1.0, 1.0, size=3)
                offset_quat = quat_from_angle_and_axis(angle, axis)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation == "fixed":
                pass
            else:
                raise error.Error(
                    f'Unknown target_rotation option "{self.target_rotation}".'
                )

        # Randomize initial position.
        if self.randomize_initial_position:
            if self.target_position != "fixed":
                initial_pos += self.np_random.normal(size=3, scale=0.005)

        initial_quat /= np.linalg.norm(initial_quat)
        initial_qpos = np.concatenate([initial_pos, initial_quat])

        self._utils.set_joint_qpos(self.model, self.data, "target:joint", initial_qpos)

        # fixme is_on_palm should be deprecated
        def is_on_palm():
            self._mujoco.mj_forward(self.model, self.data)
            cube_middle_idx = self._model_names._site_name2id["target:center"]
            cube_middle_pos = self.data.site_xpos[cube_middle_idx]
            is_on_palm = cube_middle_pos[2] > 0.04
            return is_on_palm

        # Run the simulation for a bunch of timesteps to let everything settle in.
        for _ in range(1):
            self._set_action(np.zeros(self.action_space.shape))
            try:
                self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
            except Exception:
                return False

        self._mujoco.mj_forward(self.model, self.data)

        return is_on_palm()

    def __get_site_pos(self, names):
        site_pos = []
        for name in names:
            self._mujoco.mj_forward(self.model, self.data)
            cube_middle_idx = self._model_names._site_name2id[name]
            cube_middle_pos = self.data.site_xpos[cube_middle_idx]
            site_pos.append(cube_middle_pos)
        return np.array(site_pos)

    def _sample_goal(self):
        # Select a goal for the object position.
        # target_pos = None
        # if self.target_position == "random":
        #     assert self.target_position_range.shape == (3, 2)
        #     offset = self.np_random.uniform(
        #         self.target_position_range[:, 0], self.target_position_range[:, 1]
        #     )
        #     assert offset.shape == (3,)
        #     target_pos = (
        #         self._utils.get_joint_qpos(self.model, self.data, "target:joint")[:3]
        #         + offset
        #     )
        # elif self.target_position in ["ignore", "fixed"]:
        #     target_pos = self._utils.get_joint_qpos(
        #         self.model, self.data, "target:joint"
        #     )[:3]
        # else:
        #     raise error.Error(
        #         f'Unknown target_position option "{self.target_position}".'
        #     )
        # assert target_pos is not None
        # assert target_pos.shape == (3,)
        #
        # # Select a goal for the object rotation.
        # target_quat = None
        # if self.target_rotation == "z":
        #     angle = self.np_random.uniform(-np.pi, np.pi)
        #     axis = np.array([0.0, 0.0, 1.0])
        #     target_quat = quat_from_angle_and_axis(angle, axis)

        # elif self.target_rotation == "parallel":
        #     angle = self.np_random.uniform(-np.pi, np.pi)
        #     axis = np.array([0.0, 0.0, 1.0])
        #     target_quat = quat_from_angle_and_axis(angle, axis)
        #     parallel_quat = self.parallel_quats[
        #         self.np_random.integers(len(self.parallel_quats))
        #     ]
        #     target_quat = rotations.quat_mul(target_quat, parallel_quat)
        # elif self.target_rotation == "xyz":
        #     angle = self.np_random.uniform(-np.pi, np.pi)
        #     axis = self.np_random.uniform(-1.0, 1.0, size=3)
        #     target_quat = quat_from_angle_and_axis(angle, axis)
        # elif self.target_rotation in ["ignore", "fixed"]:
        #     target_quat = self.data.get_joint_qpos("target:joint")
        # else:
        #     raise error.Error(
        #         f'Unknown target_rotation option "{self.target_rotation}".'
        #     )

        # assert target_quat is not None
        # assert target_quat.shape == (4,)

        # target_quat /= np.linalg.norm(target_quat)  # normalized quaternion
        # goal = np.concatenate([target_pos, target_quat])

        goal = self.__get_site_pos(["target:center", "target:hole0", "target:hole1"])
        return goal

    def _render_callback(self):
        # Assign current state to target object but offset a bit so that the actual object
        # is not obscured.
        # goal = self.goal.copy()
        # assert goal.shape == (7,)
        # if self.target_position == "ignore":
            # Move the object to the side since we do not care about it's position.
            # goal[0] += 0.15

        # to be deprecated
        # self._utils.set_joint_qpos(self.model, self.data, "target:joint", np.array([1.2999224 , 0.75466746 ,0.8,        0.92803925, 0.,         0. ,0.37248241]))
        # self._utils.set_joint_qvel(self.model, self.data, "target:joint", np.zeros(6))

        # to be deprecated
        # if "object_hidden" in self._model_names.geom_names:
        #     hidden_id = self._model_names.geom_name2id["object_hidden"]
        #     self.model.geom_rgba[hidden_id, 3] = 1.0

        self._mujoco.mj_forward(self.model, self.data)

    def _get_achieved_goal(self):
        # todo fix achieved goal to the finger's info
        # object_qpos = self._utils.get_joint_qpos(self.model, self.data, "target:joint")
        # assert object_qpos.shape == (7,)
        # return object_qpos

        # fingers_qpos, _ = self._utils.robot_get_obs(
        #     self.model, self.data, ["robot0:FFJ0", "robot0:MFJ0", "robot0:THJ0"]
        # )
        return self.__get_site_pos(["robot0:ff_pos_r", "robot0:mf_pos_r", "robot0:thumb_pos_r"])

    def _get_obs(self):
        robot_qpos, robot_qvel = self._utils.robot_get_obs(
            self.model, self.data, self._model_names.joint_names
        )
        # object_qvel = self._utils.get_joint_qvel(self.model, self.data, "target:joint")
        achieved_goal = (
            self._get_achieved_goal().ravel()
        )  # this contains the object position + rotation
        # print(robot_qpos.shape, robot_qvel.shape, object_qvel.shape, achieved_goal.shape)

        observation = np.concatenate(
            [robot_qpos, robot_qvel, achieved_goal]
        )

        return {
            "observation": observation.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.ravel().copy(),
        }

    def _is_success(self, achieved_goal, desired_goal):
        # d_pos, d_rot = self._goal_distance(achieved_goal, desired_goal)
        # achieved_pos = (d_pos < self.distance_threshold).astype(np.float32)
        # achieved_rot = (d_rot < self.rotation_threshold).astype(np.float32)
        # achieved_both = achieved_pos * achieved_rot
        desired_goal = desired_goal.ravel()
        ff_d = compute_pos_distance(achieved_goal[:3], desired_goal[3:6])
        mf_d = compute_pos_distance(achieved_goal[3:6], desired_goal[3:6])
        th_d = compute_pos_distance(achieved_goal[6:], desired_goal[6:])
        is_in_hold = ((ff_d < self.distance_threshold)
                      and (mf_d < self.distance_threshold)
                      and (th_d < self.distance_threshold))

        is_scissors_above = (self.__get_site_pos(["target:center"]).ravel()[2] - desired_goal[2]
                             > 0.05)
        return {"is_in_hold": is_in_hold, "is_scissors_above": is_scissors_above}

    def __is_out_of_bound(self):
        hand_pos = self.__get_site_pos(["robot0:is_out_of_bound"]).ravel()
        # print(hand_pos)
        return ((hand_pos[0] < 0.4 or hand_pos[0] > 1.65) or
                (hand_pos[1] < 0.05 or hand_pos[1] > 1.45))

    def __is_scissors_dropped(self):
        return self.goal[0, 2] - self.__get_site_pos(["target:center"]).ravel()[2] > 0.1

    def compute_reward(self, achieved_goal, goal, info):
        # _reward = super().compute_reward(achieved_goal, goal, info)
        # need to get obs
        # print(self._model_names.joint_names)

        goal = goal.ravel()
        ff_d = compute_pos_distance(achieved_goal[:3], goal[3:6])
        mf_d = compute_pos_distance(achieved_goal[3:6], goal[3:6])
        th_d = compute_pos_distance(achieved_goal[6:], goal[6:])

        if self.__is_out_of_bound():
            return -4
        if info["is_success"]["is_in_hold"]:
            return 0 + (self.__get_site_pos(["target:center"]).ravel()[2] - goal[2])
        return -(ff_d + mf_d + th_d)

    def compute_terminated(self, achieved_goal, desired_goal, info):
        """All the available environments are currently continuing tasks and non-time dependent. The objective is to reach the goal for an indefinite period of time."""
        return self.__is_out_of_bound() or self.__is_scissors_dropped()

    def compute_truncated(self, achievec_goal, desired_goal, info):
        """The environments will be truncated only if setting a time limit with max_steps which will automatically wrap the environment in a gymnasium TimeLimit wrapper."""
        return False
