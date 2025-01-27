import gc
import os
import time
import random
from typing import Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from einops import rearrange
import cv2
from tqdm.notebook import tqdm
import wandb

import gymnasium as gym
import gymnasium_robotics

from models.dreamerv2 import (
    Agent,
    Encoder,
    Decoder,
    Actor,
    RSSM,
    RewardModel,
    DiscountModel,
    preprocess_obs,
    calculate_lambda_target,
)
from models.wrapper import GymWrapper, RepeatAction





ENV_NAME = 'HandManipulateBoxRotate_BooleanTouchSensorsDense-v1'


def make_env(seed=None, max_steps=50) -> RepeatAction:
    """
    作成たラッパーをまとめて適用して環境を作成する関数．

    Returns
    -------
    env : RepeatAction
        ラッパーを適用した環境．
    """
    gym.register_envs(gymnasium_robotics)
    env = gym.make('HandManipulateBlockRotateZ_BooleanTouchSensorsDense-v1', render_mode="rgb_array", max_episode_steps=max_steps)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    # Dreamerでは観測は64x64のRGB画像
    env = GymWrapper(
        env, render_width=64, render_height=64
    )
    env = RepeatAction(env, skip=2)  # DreamerではActionRepeatは2
    return env

def set_seed(seed: int) -> None:
    """
    Pytorch, NumPyのシード値を固定します．これによりモデル学習の再現性を担保できます．

    Parameters
    ----------
    seed : int
        シード値．
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Config:
    def __init__(self, **kwargs):
        # コメントアウトされている値は，元実装のハイパーパラメータの値
        # data settings
        self.buffer_size = 100_000  # バッファにためるデータの上限
        self.batch_size = 50  # 50  # 学習時のバッチサイズ
        self.seq_length = 50  # 各バッチの系列長
        self.imagination_horizon = 15  # 15  # 想像上の軌道の系列長

        # model dimensions
        self.state_dim = 32  # 32  # 確率的な状態の次元数
        self.num_classes = 32  # 32  # 確率的な状態のクラス数（離散表現のため）
        self.rnn_hidden_dim = 600  # 600  # 決定論的な状態の次元数
        self.mlp_hidden_dim = 400  # 400  # MLPの隠れ層の次元数

        # learning params
        self.model_lr = 2e-4  # world model(transition / prior / posterior / discount / image predictor)の学習率
        self.actor_lr = 4e-5  # actorの学習率
        self.critic_lr = 1e-4  # criticの学習率
        self.epsilon = 1e-5  # optimizerのepsilonの値
        self.weight_decay = 1e-6  # weight decayの係数
        self.gradient_clipping = 100  # 勾配クリッピング
        self.kl_scale = 0.1  # kl lossのスケーリング係数
        self.kl_balance = 0.8  # kl balancingの係数(fix posterior)
        self.actor_entropy_scale = 1e-3  # entropy正則化のスケーリング係数
        self.slow_critic_update = 100  # target critic networkの更新頻度
        self.reward_loss_scale = 1.0  # reward lossのスケーリング係数
        self.discount_loss_scale = 1.0  # discount lossのスケーリング係数
        self.update_freq = 80  # 4

        # lambda return params
        self.discount = 0.995  # 割引率
        self.lambda_ = 0.95  # lambda returnのパラメータ

        # learning period settings
        self.seed_iter = 5_000  # 事前にランダム行動で探索する回数
        self.eval_freq = 5  # 評価頻度（エピソード）
        self.eval_episodes = 5  # 評価に用いるエピソード数



if __name__ == "__main__":
    cfg = Config()
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
