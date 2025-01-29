import random

from envs.config import TARGET_OBJECT_XML, TARGET_OBJECT_DICT, TARGET_OBJECT_STR

from gymnasium import error


def generate_target_object(target_obj="random", pre_train=True):
    obj_type = "pre_train" if pre_train else "tools"
    if target_obj == "random":
        obj_name, obj_str = random.choice(list(TARGET_OBJECT_DICT[obj_type].items()))
    elif target_obj in TARGET_OBJECT_DICT[obj_type].keys():
        obj_name = target_obj
        obj_str = TARGET_OBJECT_DICT[obj_type][target_obj]
    else:
        raise error.Error(
            f'Unknown target_obj: {target_obj}".'
        )

    obj_str = TARGET_OBJECT_STR.format(obj_str)
    with open(TARGET_OBJECT_XML, mode="w", encoding="utf-8") as f:
        f.write(obj_str)

    return obj_name
