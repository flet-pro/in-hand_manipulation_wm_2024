import os

import numpy as np

ASSETS_DIR = os.path.abspath(os.path.join(os.path.curdir, "assets"))  # move to central place
MANIPULATE_SCISSORS_XML = os.path.join(ASSETS_DIR, "shadow_dexterous_hand", "manipulate_block_touch_sensors.xml")
TARGET_OBJECT_XML = os.path.join(ASSETS_DIR, "shadow_dexterous_hand", "target_object.xml")

DEFAULT_CAMERA_CONFIG = {
    "distance": 0.7,  # 0.5
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
        '<geom name="target" type="mesh" mesh="object:scissors" material="block_mat" condim="3" mass="1"></geom>',
    "block":
        '<geom name="target" type="box" size="0.01 0.01 0.01" material="block_mat" condim="3" mass="1"></geom>'
}
