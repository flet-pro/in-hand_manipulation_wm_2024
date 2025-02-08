import numpy as np
from gymnasium import spaces, error
from gymnasium.utils import EzPickle

from gymnasium_robotics.envs.shadow_dexterous_hand.manipulate import quat_from_angle_and_axis
from gymnasium_robotics.utils import rotations

from envs.config import GRASP_OBJECT_ENV_XML, DEFAULT_CAMERA_CONFIG, REWARD_CONFIG
from envs.generate_target_object import generate_target_object
from envs.robot_env import MujocoRobotEnv


def compute_pos_distance(goal_a, goal_b):
    assert goal_a.shape == (3,) and goal_b.shape == (3,), [goal_a.shape, goal_b.shape]
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
            n_substeps=20,

            relative_control=False,

            pre_train=True,
            target_obj_name="random",

            sim_pre_run=10,

            random_init_pos="random",
            random_init_rot="random_z",
            random_pos_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),

            initial_qpos=None,

            reward_cfg=REWARD_CONFIG,
            **kwargs,
    ):
        ## set action
        self.relative_control = relative_control

        ## init and set environment
        self.pre_train = pre_train
        self.target_obj_name = generate_target_object(target_obj_name, pre_train)

        self.random_init_pos = False
        self.random_init_rot = False
        if random_init_pos is not None:
            assert random_init_pos in ["random", "fixed"]
            self.random_init_pos = random_init_pos
        if random_init_rot is not None:
            assert random_init_rot in ["random_z", "fixed"]
            self.random_init_rot = random_init_rot

        self.random_pos_range = random_pos_range
        self.random_quats_right = [
            rotations.euler2quat(r) for r in rotations.get_parallel_rotations()
        ]

        initial_qpos = initial_qpos or {}

        self.sim_pre_run = sim_pre_run

        ## compute reward
        self.reward_cfg = reward_cfg

        super().__init__(
            model_path=GRASP_OBJECT_ENV_XML,
            n_substeps=n_substeps,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            n_actions=26,
            initial_qpos=initial_qpos,
            **kwargs,
        )
        EzPickle.__init__(self, n_substeps, relative_control, target_obj_name, random_init_pos, random_init_rot,
                          random_pos_range, reward_cfg, **kwargs)

    def _set_action(self, action):
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

        # Randomize initial position.
        if self.random_init_pos == "random":
            initial_pos += self.np_random.normal(size=3, scale=0.005)
        elif not self.random_init_pos:
            pass
        else:
            raise error.Error(
                f'Unknown target_rotation option "{self.random_init_pos}".'
            )

        # Randomization initial rotation.
        if self.random_init_rot == "random_z":
            angle = self.np_random.uniform(-np.pi / 3, np.pi / 3)  # note this is changed to be easier
            self.scissors_angle = angle
            axis = np.array([0.0, 0.0, 1.0])
            offset_quat = quat_from_angle_and_axis(angle, axis)
            initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            # elif self.target_rotation == "parallel":
            #     angle = self.np_random.uniform(-np.pi, np.pi)
            #     axis = np.array([0.0, 0.0, 1.0])
            #     z_quat = quat_from_angle_and_axis(angle, axis)
            #     parallel_quat = self.parallel_quats[
            #         self.np_random.integers(len(self.parallel_quats))
            #     ]
            #     offset_quat = rotations.quat_mul(z_quat, parallel_quat)
            #     initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            # elif self.target_rotation in ["xyz", "ignore"]:
            #     angle = self.np_random.uniform(-np.pi, np.pi)
            #     axis = self.np_random.uniform(-1.0, 1.0, size=3)
            #     offset_quat = quat_from_angle_and_axis(angle, axis)
            #     initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            # elif self.target_rotation == "fixed":
            #     pass
        elif not self.random_init_rot:
            pass
        else:
            raise error.Error(
                f'Unknown target_rotation option "{self.random_init_rot}".'
            )

        initial_quat /= np.linalg.norm(initial_quat)
        initial_qpos = np.concatenate([initial_pos, initial_quat])

        self._utils.set_joint_qpos(self.model, self.data, "target:joint", initial_qpos)

        # Run the simulation for a bunch of timesteps to let everything settle in.
        for _ in range(self.sim_pre_run):
            self._set_action(np.zeros(self.action_space.shape))
            try:
                self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
            except Exception:
                return False

        self._mujoco.mj_forward(self.model, self.data)

        # print(not self.__is_object_dropped())
        return not self.__is_object_dropped()

    ### obs, info, terminated, truncated, and reward functions ###
    def _sample_goal(self):
        if self.pre_train:
            goal = self.__get_site_pos(["target:center"])
        else:
            goal = self.__get_site_pos(["target:center", "target:hole0", "target:hole1"])
        return goal.ravel()

    def _get_achieved(self):
        if self.pre_train:
            achieved = self.__get_site_pos(["robot0:palm_pos_r"])
        else:
            achieved = self.__get_site_pos(["robot0:ff_pos_r", "robot0:mf_pos_r", "robot0:thumb_pos_r"])
        return achieved

    def _get_obs(self):
        robot_qpos, robot_qvel = self._utils.robot_get_obs(
            self.model, self.data, self._model_names.joint_names
        )

        __achieved = (
            self._get_achieved().ravel()
        )  # this contains the object position + rotation

        observation = np.concatenate(
            [robot_qpos, robot_qvel, __achieved]
        )

        return {
            "observation": observation.copy(),
            "__achieved": __achieved.copy(),
            "__goal": self.goal.copy(),
        }

    def _get_info(self, __achieved, __goal):
        # __goal = __goal.ravel()
        if self.pre_train:
            palm_d = compute_pos_distance(__achieved, __goal)
            is_in_hold = palm_d < self.reward_cfg["dis_threshold"]
        else:
            ff_d = compute_pos_distance(__achieved[:3], __goal[3:6])
            mf_d = compute_pos_distance(__achieved[3:6], __goal[3:6])
            th_d = compute_pos_distance(__achieved[6:], __goal[6:])
            is_in_hold = ((ff_d < self.reward_cfg["dis_threshold"])
                          and (mf_d < self.reward_cfg["dis_threshold"])
                          and (th_d < self.reward_cfg["dis_threshold"]))

        is_object_above = (self.__get_site_pos(["target:center"]).ravel()[2] - __goal[2]
                             > self.reward_cfg["above_threshold"])
        return {"is_success": is_object_above, "is_in_hold": is_in_hold}

    def compute_terminated(self, __achieved, __goal, info):
        """
        All the available environments are currently continuing tasks and non-time dependent. The objective is to reach the goal for an indefinite period of time.
        """
        return (self.__is_out_of_bound() or self.__is_object_dropped()
                or
                info["is_success"])

    def compute_truncated(self, __achieved, __goal, info):
        """
        The environments will be truncated only if setting a time limit with max_steps which will automatically wrap the environment in a gymnasium TimeLimit wrapper.
        """
        return False

    def compute_reward(self, __achieved, __goal, info):
        # _reward = super().compute_reward(achieved_goal, goal, info)
        # __goal = __goal.ravel()
        if self.__is_out_of_bound() or self.__is_object_dropped():
            return self.reward_cfg["r_termination"]
        if info["is_in_hold"]:
            return 0 + (self.__get_site_pos(["target:center"]).ravel()[2] - __goal[2])
        if self.pre_train:
            palm_d = compute_pos_distance(__achieved, __goal)
            return -palm_d
        ff_d = compute_pos_distance(__achieved[:3], __goal[3:6])
        mf_d = compute_pos_distance(__achieved[3:6], __goal[3:6])
        th_d = compute_pos_distance(__achieved[6:], __goal[6:])
        return -(ff_d + mf_d + th_d)

    def _render_callback(self):
        # Assign current state to target object but offset a bit so that the actual object
        # is not obscured.
        # goal = self.goal.copy()
        # assert goal.shape == (7,)
        # if self.target_position == "ignore":
        # Move the object to the side since we do not care about it's position.
        # goal[0] += 0.15

        self._mujoco.mj_forward(self.model, self.data)

    def __is_out_of_bound(self):
        hand_pos = self.__get_site_pos(["robot0:is_out_of_bound"]).ravel()
        # print(hand_pos)
        return ((hand_pos[0] < 0.4 or hand_pos[0] > 1.65) or
                (hand_pos[1] < 0.05 or hand_pos[1] > 1.45))

    def __is_object_dropped(self):
        # todo not work properly
        # return self.goal[0, 2] - self.__get_site_pos(["target:center"]).ravel()[2] > 0.1
        # print(self.__get_site_pos(["target:center"]).ravel()[2])
        return self.__get_site_pos(["target:center"]).ravel()[2] < 0.2 - 0.02

    ### util functions ###
    def __get_site_pos(self, names):
        site_pos = []
        for name in names:
            self._mujoco.mj_forward(self.model, self.data)
            cube_middle_idx = self._model_names._site_name2id[name]
            cube_middle_pos = self.data.site_xpos[cube_middle_idx]
            site_pos.append(cube_middle_pos)
        return np.array(site_pos)
