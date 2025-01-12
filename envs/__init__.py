from gymnasium.envs.registration import register


register(
    id="HandManipulateScissorsGrasp-v1",
    entry_point="envs.shadow_dexterous_hand.manipulate_scissors:MujocoHandScissorsEnv",
)
