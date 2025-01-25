import gymnasium as gym
import gymnasium_robotics
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation

import envs

from models.wrapper import GymWrapper, RepeatAction


ENV_NAME = "AdroitGraspPreTrain-v1"


def make_env() -> RepeatAction:
    """
    作成たラッパーをまとめて適用して環境を作成する関数．

    Returns
    -------
    env : RepeatAction
        ラッパーを適用した環境．
    """
    gym.register_envs(gymnasium_robotics)
    gym.register_envs(envs)
    env = gym.make(ENV_NAME, render_mode="rgb_array", max_episode_steps=50)

    # Dreamerでは観測は64x64のRGB画像
    # env = GymWrapper(
    #     env, render_width=64, render_height=64
    # )
    # env = RepeatAction(env, skip=2)  # DreamerではActionRepeatは2
    return env


if __name__ == "__main__":
    env = make_env()

    obs, obs_hand = env.reset()

    imgs = []
    fig = plt.figure()

    for _ in range(30):
        action = env.action_space.sample()  # User-defined policy function
        action[:] = 0
        # action[22] = 0

        obs, reward, terminated, truncated, info = env.step(action)

        # print(obs.shape)
        # print(obs_hand.shape)
        # print(terminated)
        # print(reward)

        img = env.render()
        im = plt.imshow(img, animated=True)
        imgs.append([im])

    ani = ArtistAnimation(fig, imgs, interval=100, blit=True, repeat=False)
    ani.save('videos/anim.mp4', writer="ffmpeg")
    plt.show()
