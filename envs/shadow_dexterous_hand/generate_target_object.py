import random

from envs.shadow_dexterous_hand.config import TARGET_OBJECT_XML, TARGET_OBJECT_DICT, TARGET_OBJECT_STR

from gymnasium import error


def generate_target_object(target_obj="random"):
    if target_obj == "random":
        obj_name, obj_str = random.choice(list(TARGET_OBJECT_DICT.items()))
    elif target_obj == "block":
        obj_name = "block"
        obj_str = TARGET_OBJECT_DICT["block"]
    elif target_obj == "scissors":
        obj_name = "scissors"
        obj_str = TARGET_OBJECT_DICT["scissors"]
    else:
        raise error.Error(
            f'Unknown target_obj: {target_obj}".'
        )

    obj_str = TARGET_OBJECT_STR.format(obj_str)
    with open(TARGET_OBJECT_XML, mode="w", encoding="utf-8") as f:
        f.write(obj_str)

    return obj_name
