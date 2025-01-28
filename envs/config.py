import os

import numpy as np

ASSETS_DIR = os.path.abspath(os.path.join(os.path.curdir, "assets"))  # move to central place
GRASP_OBJECT_ENV_XML = os.path.join(ASSETS_DIR, "shadow_dexterous_hand", "manipulate_block_touch_sensors.xml")
TARGET_OBJECT_XML = os.path.join(ASSETS_DIR, "shadow_dexterous_hand", "target_object.xml")

DEFAULT_CAMERA_CONFIG = {
    "distance": 1.0,  # 0.5
    "azimuth": 0.0,  # 55.0
    "elevation": -50.0,  # -25.0
    "lookat": np.array([1.3, 0.75, 0.45]),  # np.array([1, 0.96, 0.14])
}

TARGET_OBJECT_STR = """
<mujoco>
    {}
</mujoco>
"""

TARGET_OBJECT_DICT = {
    "scissors":
        '<geom name="target" euler="0 0 0" type="mesh" mesh="object:scissors" material="block_mat" condim="4" mass="1" />',
    "block_box":
        '<geom name="target" euler="0 0 0" type="box" size="0.035 0.035 0.035" material="block_mat" condim="4" mass="1" />',
    "block_ball":
        '<geom name="target" euler="0 0 0" type="sphere" size="0.035" material="block_mat" condim="4" mass="1" />',
    "block_cylinder":
        '<geom name="target" euler="0 0 0" type="cylinder" size="0.035 0.035" material="block_mat" condim="4" mass="1" />',
    "block_capsule":
        '<geom name="target" euler="0 0 0" type="capsule" size="0.025 0.035" material="block_mat" condim="4" mass="1" />'
}
