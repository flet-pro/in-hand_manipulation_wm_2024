import gymnasium_robotics
import numpy as np

import envs
import matplotlib.pyplot as plt
import gymnasium as gym

from matplotlib.animation import ArtistAnimation

# actually not necessary
gym.register_envs(gymnasium_robotics)
gym.register_envs(envs)

# env = gym.make("HandManipulateBlockRotateZ-v1", render_mode="rgb_array")
env = gym.make("HandManipulateScissorsGrasp-v1", render_mode="rgb_array")
env.reset()

# action = env.action_space.sample()
# print(action.shape)
# action[20:26] = 0
# obs, reward, terminated, truncated, info = env.step(action)
# img = env.render()
#
# print(reward)
# print(terminated)
# print(truncated)
# print(obs['observation'].shape)
# print(obs['observation'])
# plt.imshow(img)
# plt.show()

imgs = []
fig = plt.figure()

for _ in range(30):
    action = env.action_space.sample()  # User-defined policy function
    # action[:] = 0
    obs, reward, terminated, truncated, info = env.step(action)
    # print(reward)

    img = env.render()
    im = plt.imshow(img, animated=True)
    imgs.append([im])

ani = ArtistAnimation(fig, imgs, interval=50, blit=True, repeat=False)
ani.save('videos/anim.mp4', writer="ffmpeg")
plt.show()
