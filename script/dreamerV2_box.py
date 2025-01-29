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

import torch
import torch.distributions as td
from torch.distributions import Normal, Categorical, OneHotCategorical, OneHotCategoricalStraightThrough
from torch.distributions.kl import kl_divergence
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)

from models.dreamerv2 import (
    Agent,
    Encoder,
    Decoder,
    Actor,
    RSSM,
    RewardModel,
    DiscountModel,
    Critic,
    preprocess_obs,
    calculate_lambda_target,
)
from models.replay_buffer import ReplayBuffer
from models.wrapper import GymWrapper, RepeatAction

ENV_NAME = 'HandManipulateBoxRotate_BooleanTouchSensorsDense-v1'


# 環境設定
class CustomManipulateBoxEnv(gym.Env):
    def __init__(self, env):
        self.env = env

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        image = self.env.render()
        return image, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)

class RepeatAction(CustomManipulateBoxEnv):
    def __init__(self, env, skip=4, max_steps=100_000):
        gym.Wrapper.__init__(self, env)
        self.max_steps = max_steps if max_steps else float("inf")
        self.steps = 0
        self.height = env.observation_space.shape[0]
        self.width = env.observation_space.shape[1]
        self._skip = skip

    @property
    def observation_space():
        img = self.env.render()
        return gym.spaces.Box(img)
        
    def reset(self):
        obs = self.env.reset()
        return obs[0]

    def step(self, action):
        if self.steps >= self.max_steps:
            print("Reached max iterations.")
            return None

        total_reward = 0.0
        self.steps += 1
        for _ in range(self._skip):
            obs, reward, done, _, info = self.env.step(action)
            img = self.env.render()

            total_reward += reward
            if self.steps >= self.max_steps:
                done = True

            if done:
                break

        return img, total_reward, done, info

def make_env(seed=None, max_steps=50) -> RepeatAction:
    """
    作成たラッパーをまとめて適用して環境を作成する関数．

    Returns
    -------
    env : RepeatAction
        ラッパーを適用した環境．
    """
    env = gym.make('HandManipulateBlockRotateZ_BooleanTouchSensorsDense-v1', render_mode="rgb_array", max_episode_steps=max_steps)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    # Dreamerでは観測は64x64のRGB画像
    env = CustomManipulateBoxEnv(env)
    # env = GymWrapper(
    #     env, render_width=64, render_height=64
    # )
    env = RepeatAction(env, skip=4)  # DreamerではActionRepeatは2
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
    
# モデルパラメータをGoogleDriveに保存・後で読み込みするためのヘルパークラス
class TrainedModels:
    def __init__(self, *models) -> None:
        """
        コンストラクタ．

        Parameters
        ----------
        models : nn.Module
            保存するモデル．複数モデルを渡すことができます．

        使用例: trained_models = TraindModels(encoder, rssm, value_model, action_model)
        """
        assert np.all([nn.Module in model.__class__.__bases__ for model in models]), "Arguments for TrainedModels need to be nn models."

        self.models = models

    def save(self, dir: str) -> None:
        """
        initで渡したモデルのパラメータを保存します．
        パラメータのファイル名は01.pt, 02.pt, ... のように連番になっています．

        Parameters
        ----------
        dir : str
            パラメータの保存先．
        """
        for i, model in enumerate(self.models):
            torch.save(
                model.state_dict(),
                os.path.join(dir, f"{str(i + 1).zfill(2)}.pt")
            )

    def load(self, dir: str, device: str) -> None:
        """
        initで渡したモデルのパラメータを読み込みます．

        Parameters
        ----------
        dir : str
            パラメータの保存先．
        device : str
            モデルをどのデバイス(CPU or GPU)に載せるかの設定．
        """
        for i, model in enumerate(self.models):
            model.load_state_dict(
                torch.load(
                    os.path.join(dir, f"{str(i + 1).zfill(2)}.pt"),
                    map_location=device
                )
            )

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
    # モデル等の初期化
    seed = 0
    NUM_ITER = 100_000  # 環境とのインタラクション回数の制限 ※変更しないでください
    set_seed(seed)
    env = make_env(max_steps=NUM_ITER)
    eval_env = make_env(seed=1234, max_steps=None)  # omnicampus上の環境と同じシード値で評価環境を作成
    device = "cuda" if torch.cuda.is_available() else "cpu"

    action_dim = env.action_space.n
    # リプレイバッファ
    replay_buffer = ReplayBuffer(
        capacity=cfg.buffer_size,
        observation_shape=(64, 64, 1),
        action_dim=env.action_space.n
    )

    # モデル
    rssm = RSSM(cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes, action_dim).to(device)
    encoder = Encoder().to(device)
    decoder = Decoder(cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    reward_model =  RewardModel(cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    discount_model = DiscountModel(cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    actor = Actor(action_dim, cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    critic = Critic(cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    target_critic = Critic(cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    target_critic.load_state_dict(critic.state_dict())

    trained_models = TrainedModels(
        rssm,
        encoder,
        decoder,
        reward_model,
        discount_model,
        actor,
        critic
    )

    # optimizer
    wm_params = list(rssm.parameters())         + \
                list(encoder.parameters())      + \
                list(decoder.parameters())      + \
                list(reward_model.parameters()) + \
                list(discount_model.parameters())

    wm_optimizer = torch.optim.AdamW(wm_params, lr=cfg.model_lr, eps=cfg.epsilon, weight_decay=cfg.weight_decay)
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=cfg.actor_lr, eps=cfg.epsilon, weight_decay=cfg.weight_decay)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=cfg.critic_lr, eps=cfg.epsilon, weight_decay=cfg.weight_decay)
    
    # 学習を行う
    # 環境と相互作用 → 一定イテレーションでモデル更新を繰り返す
    policy = Agent(encoder, rssm, actor)

    # 環境，収益等の初期化
    obs = env.reset()
    done = False
    total_reward = 0
    total_episode = 1
    best_reward = -1

    for iteration in range(NUM_ITER - cfg.seed_iter):
        with torch.no_grad():
            # 環境と相互作用
            action = policy(obs)  # モデルで行動をサンプリング(one-hot)
            action_int = np.argmax(action)  # 環境に渡すときはint型
            next_obs, reward, done, _ = env.step(action_int)  # 環境を進める

            # 得たデータをリプレイバッファに追加して更新
            replay_buffer.push(preprocess_obs(obs), action, np.tanh(reward), done)  # x_t, a_t, r_t, gamma_t
            obs = next_obs
            total_reward += reward

        if (iteration + 1) % cfg.update_freq == 0:
            # モデルの学習
            # リプレイバッファからデータをサンプリングする
            # (batch size, seq_lenght, *data shape)
            observations, actions, rewards, done_flags =\
                replay_buffer.sample(cfg.batch_size, cfg.seq_length)
            done_flags = 1 - done_flags  # 終端でない場合に1をとる

            # torchで扱える形（seq lengthを最初の次元に，画像はchnnelを最初の次元にする）に変形，observationの前処理
            observations = torch.permute(torch.as_tensor(observations, device=device), (1, 0, 4, 2, 3))  # (T, B, C, H, W)
            actions = torch.as_tensor(actions, device=device).transpose(0, 1)  # (T, B, action dim)
            rewards = torch.as_tensor(rewards, device=device).transpose(0, 1)  # (T, B, 1)
            done_flags = torch.as_tensor(done_flags, device=device).transpose(0, 1).float()  # (T, B, 1)

            # =================
            # world modelの学習
            # =================
            # 観測をベクトルに埋めこみ
            emb_observations = encoder(observations.reshape(-1, 1, 64, 64)).view(cfg.seq_length, cfg.batch_size, -1)  # (T, B, 1536)

            # 状態表現z，行動aはゼロで初期化
            # バッファから取り出したデータをt={1, ..., seq length}とするなら，以下はz_1とみなせる
            state = torch.zeros(cfg.batch_size, cfg.state_dim*cfg.num_classes, device=device)
            rnn_hidden = torch.zeros(cfg.batch_size, cfg.rnn_hidden_dim, device=device)

            # 各観測に対して状態表現を計算
            # タイムステップごとに計算するため，先に格納するTensorを定義する(t={1, ..., seq length})
            states = torch.zeros(cfg.seq_length, cfg.batch_size, cfg.state_dim*cfg.num_classes, device=device)
            rnn_hiddens = torch.zeros(cfg.seq_length, cfg.batch_size, cfg.rnn_hidden_dim, device=device)

            # prior, posteriorを計算してKL lossを計算する
            kl_loss = 0
            for i in range(cfg.seq_length-1):
                # rnn hiddenを更新
                rnn_hidden = rssm.recurrent(state, actions[i], rnn_hidden)  # h_t+1

                # prior, posteriorを計算
                next_state_prior, next_detach_prior = rssm.get_prior(rnn_hidden, detach=True) # \hat{z}_t+1
                next_state_posterior, next_detach_posterior = rssm.get_posterior(rnn_hidden, emb_observations[i+1], detach=True)  # z_t+1

                # posteriorからzをサンプリング
                state = next_state_posterior.rsample().flatten(1)
                rnn_hiddens[i+1] = rnn_hidden  # h_t+1
                states[i+1] = state  # z_t+1

                # KL lossを計算
                kl_loss +=  cfg.kl_balance * torch.mean(kl_divergence(next_detach_posterior, next_state_prior)) + \
                            (1 - cfg.kl_balance) * torch.mean(kl_divergence(next_state_posterior, next_detach_prior))
            kl_loss /= (cfg.seq_length - 1)

            # 初期状態は使わない
            rnn_hiddens = rnn_hiddens[1:]  # (seq lenghth - 1, batch size rnn hidden)
            states = states[1:]  # (seq length - 1, batch size, state dim * num_classes)

            # 得られた状態を利用して再構成，報酬，終端フラグを予測
            # そのままでは時間方向，バッチ方向で次元が多いため平坦化
            flatten_rnn_hiddens = rnn_hiddens.view(-1, cfg.rnn_hidden_dim)  # ((T-1) * B, rnn hidden)
            flatten_states = states.view(-1, cfg.state_dim * cfg.num_classes)  # ((T-1) * B, state_dim * num_classes)

            # 上から再構成，報酬，終端フラグ予測
            obs_dist = decoder(flatten_states, flatten_rnn_hiddens)  # (T * B, 3, 64, 64)
            reward_dist = reward_model(flatten_states, flatten_rnn_hiddens)  # (T * B, 1)
            discount_dist = discount_model(flatten_states, flatten_rnn_hiddens)  # (T * B, 1)

            # 各予測に対する損失の計算（対数尤度）
            C, H, W = observations.shape[2:]
            obs_loss = -torch.mean(obs_dist.log_prob(observations[1:].reshape(-1, C, H, W)))
            reward_loss = -torch.mean(reward_dist.log_prob(rewards[:-1].reshape(-1, 1)))
            discount_loss = -torch.mean(discount_dist.log_prob(done_flags[:-1].float().reshape(-1, 1)))

            # 総和をとってモデルを更新
            wm_loss = obs_loss + cfg.reward_loss_scale * reward_loss + cfg.discount_loss_scale * discount_loss + cfg.kl_scale * kl_loss

            wm_optimizer.zero_grad()
            wm_loss.backward()
            clip_grad_norm_(wm_params, cfg.gradient_clipping)
            wm_optimizer.step()

            #====================
            # Actor, Criticの更新
            #===================
            # wmから得た状態の勾配を切っておく
            flatten_rnn_hiddens = flatten_rnn_hiddens.detach()
            flatten_states = flatten_states.detach()

            # priorを用いた状態予測
            # 格納する空のTensorを用意
            imagined_states = torch.zeros(cfg.imagination_horizon + 1,
                                        *flatten_states.shape,
                                        device=flatten_states.device)
            imagined_rnn_hiddens = torch.zeros(cfg.imagination_horizon + 1,
                                            *flatten_rnn_hiddens.shape,
                                            device=flatten_rnn_hiddens.device)
            imagined_action_log_probs = torch.zeros((cfg.imagination_horizon, cfg.batch_size * (cfg.seq_length-1)),
                                                    device=flatten_rnn_hiddens.device)
            imagined_action_entropys = torch.zeros((cfg.imagination_horizon, cfg.batch_size * (cfg.seq_length-1)),
                                                    device=flatten_rnn_hiddens.device)

            # 未来予測をして想像上の軌道を作る前に, 最初の状態としては先ほどモデルの更新で使っていた
            # リプレイバッファからサンプルされた観測データを取り込んだ上で推論した状態表現を使う
            imagined_states[0] = flatten_states
            imagined_rnn_hiddens[0] = flatten_rnn_hiddens

            # open-loopで予測
            for i in range(1, cfg.imagination_horizon + 1):
                actions, action_log_probs, action_entropys = actor(flatten_states.detach(), flatten_rnn_hiddens.detach())  # ((T-1) * B, action dim)

                # rnn hiddenを更新, priorで次の状態を予測
                with torch.no_grad():
                    flatten_rnn_hiddens = rssm.recurrent(flatten_states, actions, flatten_rnn_hiddens)  # h_t+1
                    flatten_states_prior = rssm.get_prior(flatten_rnn_hiddens)
                    flatten_states = flatten_states_prior.rsample().flatten(1)

                imagined_rnn_hiddens[i] = flatten_rnn_hiddens.detach()
                imagined_states[i] = flatten_states.detach()
                imagined_action_log_probs[i-1] = action_log_probs
                imagined_action_entropys[i-1] = action_entropys

            imagined_states = imagined_states[1:]
            imagined_rnn_hiddens = imagined_rnn_hiddens[1:]

            # 得られた状態から報酬を予測
            flatten_imagined_states = imagined_states.view(-1, cfg.state_dim * cfg.num_classes).detach()  # ((imagination horizon) * (T-1) * B, state dim * num classes)
            flatten_imagined_rnn_hiddens = imagined_rnn_hiddens.view(-1, cfg.rnn_hidden_dim).detach()  # ((imagination horizon) * (T-1) * B, rnn hidden)

            # reward, done_flagsは分布なので平均値をとる
            # ((imagination horizon + 1), (T-1) * B)
            with torch.no_grad():
                imagined_rewards = reward_model(flatten_imagined_states, flatten_imagined_rnn_hiddens).mean.view(cfg.imagination_horizon, -1)
                target_values = target_critic(flatten_imagined_states, flatten_imagined_rnn_hiddens).view(cfg.imagination_horizon, -1)
                imagined_done_flags = discount_model(flatten_imagined_states, flatten_imagined_rnn_hiddens).base_dist.probs.view(cfg.imagination_horizon, -1)
                discount_arr = cfg.discount * torch.round(imagined_done_flags)

            # lambda targetの計算
            lambda_target = calculate_lambda_target(imagined_rewards, discount_arr, target_values, cfg.lambda_)

            # actorの損失を計算
            objective = imagined_action_log_probs * ((lambda_target - target_values).detach())
            discount_arr = torch.cat([torch.ones_like(discount_arr[:1]), discount_arr[1:]])
            discount = torch.cumprod(discount_arr, 0)
            actor_loss = -torch.sum(torch.mean(discount * (objective + cfg.actor_entropy_scale * imagined_action_entropys), dim=1))

            actor_optimizer.zero_grad()
            actor_loss.backward()
            clip_grad_norm_(actor.parameters(), cfg.gradient_clipping)
            actor_optimizer.step()

            # criticの損失を計算
            value_mean = critic(flatten_imagined_states.detach(), flatten_imagined_rnn_hiddens.detach()).view(cfg.imagination_horizon, -1)
            value_dist = td.Independent(td.Normal(value_mean, 1),  1)
            critic_loss = -torch.mean(discount.detach() * value_dist.log_prob(lambda_target.detach()).unsqueeze(-1))

            critic_optimizer.zero_grad()
            critic_loss.backward()
            clip_grad_norm_(critic.parameters(), cfg.gradient_clipping)
            critic_optimizer.step()

        if (iteration + 1) % cfg.slow_critic_update == 0:
            target_critic.load_state_dict(critic.state_dict())

        # エピソードが終了した時に再初期化
        if done:
            print(f"episode: {total_episode} total_reward: {total_reward:.8f}")
            print(f"num iter: {iteration} kl loss: {kl_loss.item():.8f} obs loss: {obs_loss.item():.8f} "
                f"rewrd loss: {reward_loss.item():.8f} discount loss: {discount_loss.item():.8f} "
                f"critic loss: {critic_loss.item():.8f} actor loss: {actor_loss.item():.8f}"
            )
            obs = env.reset()
            done = False
            total_reward = 0
            total_episode += 1
            policy.reset()

            # 一定エピソードごとに評価
            if total_episode % cfg.eval_freq == 0:
                eval_reward = evaluation(eval_env, policy, iteration, cfg)
                trained_models.save("./")
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    os.makedirs("./best_models", exist_ok=True)
                    trained_models.save("./best_models")

                eval_env.reset()
                policy.reset()

    trained_models.save("./")
