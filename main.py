import gymnasium_robotics
import envs
import matplotlib.pyplot as plt
import gymnasium as gym

# actually not necessary
gym.register_envs(gymnasium_robotics)
gym.register_envs(envs)


env = gym.make("HandManipulateScissorsGrasp-v0", render_mode="rgb_array")
env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
img = env.render()

print(reward)
print(terminated)
print(truncated)
plt.imshow(img)
plt.show()
