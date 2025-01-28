from gymnasium.envs.registration import register


register(
    id="ShadowHandGraspObject-v1",
    entry_point="envs.shadow_dexterous_hand.grasp_object:GraspObjectEnv",
)

register(
    id="AdroitGraspPreTrain-v1",
    entry_point="envs.adroit_hand.grasp_pre_train:GraspPreTrainEnv",
)
