import time
import os
import gc
import random
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import gymnasium as gym
import gymnasium_robotics
import torch
import torch.distributions as td
from torch.distributions import Normal, Categorical, OneHotCategorical, OneHotCategoricalStraightThrough
from torch.distributions.kl import kl_divergence
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

# 環境設定
class CustomManipulateBoxEnv(gym.Env):
    def __init__(self, env):
        self.env = env
        self.egg_drop_threshold = 0.05

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

def make_env(seed=None, img_size=64, max_steps=100_000):
    env = gym.make('HandManipulateBlockRotateZ_BooleanTouchSensorsDense-v1', 
                  render_mode="rgb_array", 
                  max_episode_steps=max_steps)
    env = CustomManipulateBoxEnv(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    env = RepeatAction(env=env, skip=4, max_steps=max_steps)
    return env

# モデル定義
class RSSM(nn.Module):
    def __init__(self, mlp_hidden_dim: int, rnn_hidden_dim: int, state_dim: int, num_classes: int, action_dim: int):
        super().__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.state_dim = state_dim
        self.num_classes = num_classes

        # Recurrent model
        self.transition_hidden = nn.Linear(state_dim * num_classes + action_dim, mlp_hidden_dim)
        self.transition = nn.GRUCell(mlp_hidden_dim, rnn_hidden_dim)

        # transition predictor
        self.prior_hidden = nn.Linear(rnn_hidden_dim, mlp_hidden_dim)
        self.prior_logits = nn.Linear(mlp_hidden_dim, state_dim * num_classes)

        # representation model
        self.posterior_hidden = nn.Linear(rnn_hidden_dim + 1536, mlp_hidden_dim)
        self.posterior_logits = nn.Linear(mlp_hidden_dim, state_dim * num_classes)

    def recurrent(self, state: torch.Tensor, action: torch.Tensor, rnn_hidden: torch.Tensor):
        hidden = F.elu(self.transition_hidden(torch.cat([state, action], dim=1)))
        rnn_hidden = self.transition(hidden, rnn_hidden)
        return rnn_hidden

    def get_prior(self, rnn_hidden: torch.Tensor, detach=False):
        hidden = F.elu(self.prior_hidden(rnn_hidden))
        logits = self.prior_logits(hidden)
        logits = logits.reshape(logits.shape[0], self.state_dim, self.num_classes)
        prior_dist = td.Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
        if detach:
            detach_prior = td.Independent(OneHotCategoricalStraightThrough(logits=logits.detach()), 1)
            return prior_dist, detach_prior
        return prior_dist

    def get_posterior(self, rnn_hidden: torch.Tensor, embedded_obs: torch.Tensor, detach=False):
        hidden = F.elu(self.posterior_hidden(torch.cat([rnn_hidden, embedded_obs], dim=1)))
        logits = self.posterior_logits(hidden)
        logits = logits.reshape(logits.shape[0], self.state_dim, self.num_classes)
        posterior_dist = td.Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
        if detach:
            detach_posterior = td.Independent(OneHotCategoricalStraightThrough(logits=logits.detach()), 1)
            return posterior_dist, detach_posterior
        return posterior_dist

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 48, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(48, 96, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(96, 192, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(192, 384, kernel_size=4, stride=2)

    def forward(self, obs: torch.Tensor):
        hidden = F.elu(self.conv1(obs))
        hidden = F.elu(self.conv2(hidden))
        hidden = F.elu(self.conv3(hidden))
        embedded_obs = self.conv4(hidden).reshape(hidden.size(0), -1)
        return embedded_obs

class Decoder(nn.Module):
    def __init__(self, rnn_hidden_dim: int, state_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(state_dim*num_classes + rnn_hidden_dim, 1536)
        self.dc1 = nn.ConvTranspose2d(1536, 192, kernel_size=5, stride=2)
        self.dc2 = nn.ConvTranspose2d(192, 96, kernel_size=5, stride=2)
        self.dc3 = nn.ConvTranspose2d(96, 48, kernel_size=6, stride=2)
        self.dc4 = nn.ConvTranspose2d(48, 1, kernel_size=6, stride=2)

    def forward(self, state: torch.Tensor, rnn_hidden: torch.Tensor):
        hidden = self.fc(torch.cat([state, rnn_hidden], dim=1))
        hidden = hidden.view(hidden.size(0), 1536, 1, 1)
        hidden = F.elu(self.dc1(hidden))
        hidden = F.elu(self.dc2(hidden))
        hidden = F.elu(self.dc3(hidden))
        mean = self.dc4(hidden)
        obs_dist = td.Independent(td.Normal(mean, 1), 3)
        return obs_dist

class RewardModel(nn.Module):
    def __init__(self, hidden_dim: int, rnn_hidden_dim: int, state_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim*num_classes + rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, rnn_hidden: torch.Tensor):
        hidden = F.elu(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = F.elu(self.fc2(hidden))
        hidden = F.elu(self.fc3(hidden))
        mean = self.fc4(hidden)
        reward_dist = td.Independent(td.Normal(mean, 1), 1)
        return reward_dist

class DiscountModel(nn.Module):
    def __init__(self, hidden_dim: int, rnn_hidden_dim: int, state_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim*num_classes + rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, rnn_hidden: torch.Tensor):
        hidden = F.elu(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = F.elu(self.fc2(hidden))
        hidden = F.elu(self.fc3(hidden))
        mean= self.fc4(hidden)
        discount_dist = td.Independent(td.Bernoulli(logits=mean), 1)
        return discount_dist

class Actor(nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int, rnn_hidden_dim: int, state_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim * num_classes + rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.tensor, rnn_hidden: torch.Tensor, eval: bool = False):
        hidden = F.elu(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = F.elu(self.fc2(hidden))
        hidden = F.elu(self.fc3(hidden))
        hidden = F.elu(self.fc4(hidden))
        logits = self.out(hidden)

        if eval:
            action = torch.argmax(logits, dim=1)
            action = F.one_hot(action, logits.shape[1])
            return action, None, None

        action_dist = OneHotCategorical(logits=logits)
        action = action_dist.sample()
        action = action + action_dist.probs - action_dist.probs.detach()
        action_log_prob = action_dist.log_prob(torch.round(action.detach()))
        action_entropy = action_dist.entropy()
        return action, action_log_prob, action_entropy

class Critic(nn.Module):
    def __init__(self, hidden_dim: int, rnn_hidden_dim: int, state_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim * num_classes + rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.tensor, rnn_hidden: torch.Tensor):
        hidden = F.elu(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = F.elu(self.fc2(hidden))
        hidden = F.elu(self.fc3(hidden))
        hidden = F.elu(self.fc4(hidden))
        mean = self.out(hidden)
        return mean

# 学習設定
class Config:
    def __init__(self, **kwargs):
        self.buffer_size = 100_000
        self.batch_size = 50
        self.seq_length = 50
        self.imagination_horizon = 15
        self.state_dim = 32
        self.num_classes = 32
        self.rnn_hidden_dim = 600
        self.mlp_hidden_dim = 400
        self.model_lr = 2e-4
        self.actor_lr = 4e-5
        self.critic_lr = 1e-4
        self.epsilon = 1e-5
        self.weight_decay = 1e-6
        self.gradient_clipping = 100
        self.kl_scale = 0.1
        self.kl_balance = 0.8
        self.actor_entropy_scale = 1e-3
        self.slow_critic_update = 100
        self.reward_loss_scale = 1.0
        self.discount_loss_scale = 1.0
        self.update_freq = 80
        self.discount = 0.995
        self.lambda_ = 0.95
        self.seed_iter = 5_000
        self.eval_freq = 5
        self.eval_episodes = 5

# 学習ループ
def train():
    cfg = Config()
    seed = 0
    NUM_ITER = 100_000
    
    env = make_env(max_steps=NUM_ITER)
    eval_env = make_env(seed=1234, max_steps=None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # モデル初期化
    rssm = RSSM(cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes, env.action_space.n).to(device)
    encoder = Encoder().to(device)
    decoder = Decoder(cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    reward_model = RewardModel(cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    discount_model = DiscountModel(cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    actor = Actor(env.action_space.n, cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    critic = Critic(cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    
    # バッファ初期化
    buffer = []
    rnn_hidden = torch.zeros(1, cfg.rnn_hidden_dim).to(device)
    state = torch.zeros(1, cfg.state_dim * cfg.num_classes).to(device)
    
    # 初期シード
    obs, _ = env.reset()
    obs = torch.FloatTensor(obs).unsqueeze(0).to(device)
    
    # 学習ループ
    for step in range(NUM_ITER):
        # 行動選択
        with torch.no_grad():
            embedded_obs = encoder(obs)
            rnn_hidden = rssm.recurrent(state, torch.zeros(1, env.action_space.n).to(device), rnn_hidden)
            prior_dist = rssm.get_prior(rnn_hidden)
            state = prior_dist.sample()
            action, _, _ = actor(state, rnn_hidden)
        
        # 環境ステップ
        next_obs, reward, done, info = env.step(action.cpu().numpy()[0])
        next_obs = torch.FloatTensor(next_obs).unsqueeze(0).to(device)
        reward = torch.FloatTensor([reward]).unsqueeze(0).to(device)
        done = torch.FloatTensor([done]).unsqueeze(0).to(device)
        
        # バッファに保存
        buffer.append((obs, action, reward, done, next_obs))
        if len(buffer) > cfg.buffer_size:
            buffer.pop(0)
        
        # モデル更新
        if step > cfg.seed_iter and step % cfg.update_freq == 0:
            # バッチサンプリング
            idxs = np.random.randint(0, len(buffer) - cfg.seq_length, cfg.batch_size)
            sequences = [buffer[idx:idx+cfg.seq_length] for idx in idxs]
            
            # データ整形
            obs_batch = torch.stack([torch.stack([seq[0] for seq in s]) for s in sequences])
            action_batch = torch.stack([torch.stack([seq[1] for seq in s]) for s in sequences])
            reward_batch = torch.stack([torch.stack([seq[2] for seq in s]) for s in sequences])
            done_batch = torch.stack([torch.stack([seq[3] for seq in s]) for s in sequences])
            next_obs_batch = torch.stack([torch.stack([seq[4] for seq in s]) for s in sequences])
            
            # モデル更新
            # オプティマイザ初期化
            model_optimizer = torch.optim.Adam([
                {'params': rssm.parameters()},
                {'params': encoder.parameters()},
                {'params': decoder.parameters()},
                {'params': reward_model.parameters()},
                {'params': discount_model.parameters()}
            ], lr=cfg.model_lr, weight_decay=cfg.weight_decay, eps=cfg.epsilon)
            
            actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr, weight_decay=cfg.weight_decay, eps=cfg.epsilon)
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr, weight_decay=cfg.weight_decay, eps=cfg.epsilon)
            
            # 勾配リセット
            model_optimizer.zero_grad()
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            
            # モデル更新処理
            rnn_hidden_batch = torch.zeros(cfg.batch_size, cfg.rnn_hidden_dim).to(device)
            state_batch = torch.zeros(cfg.batch_size, cfg.state_dim * cfg.num_classes).to(device)
            
            model_loss = 0
            actor_loss = 0
            critic_loss = 0
            
            for t in range(cfg.seq_length):
                # 観測のエンコード
                embedded_obs = encoder(obs_batch[:, t])
                
                # RSSM更新
                rnn_hidden_batch = rssm.recurrent(state_batch, action_batch[:, t-1] if t > 0 else torch.zeros_like(action_batch[:, t]), rnn_hidden_batch)
                prior_dist = rssm.get_prior(rnn_hidden_batch)
                posterior_dist = rssm.get_posterior(rnn_hidden_batch, embedded_obs)
                state_batch = posterior_dist.sample()
                
                # モデル損失計算
                kl_loss = cfg.kl_scale * (cfg.kl_balance * kl_divergence(posterior_dist, prior_dist).mean() +
                                        (1 - cfg.kl_balance) * kl_divergence(prior_dist, posterior_dist).mean())
                model_loss += kl_loss
                
                # デコードと損失計算
                obs_dist = decoder(state_batch, rnn_hidden_batch)
                obs_loss = -obs_dist.log_prob(obs_batch[:, t]).mean()
                model_loss += obs_loss
                
                reward_dist = reward_model(state_batch, rnn_hidden_batch)
                reward_loss = -reward_dist.log_prob(reward_batch[:, t]).mean()
                model_loss += cfg.reward_loss_scale * reward_loss
                
                discount_dist = discount_model(state_batch, rnn_hidden_batch)
                discount_loss = -discount_dist.log_prob(done_batch[:, t]).mean()
                model_loss += cfg.discount_loss_scale * discount_loss
                
                # Actor-Critic更新
                with torch.no_grad():
                    value = critic(state_batch, rnn_hidden_batch)
                    next_value = critic(state_batch, rnn_hidden_batch)
                    target = reward_batch[:, t] + (1 - done_batch[:, t]) * cfg.discount * next_value
                    advantage = target - value
                
                # Actor損失
                action, action_log_prob, action_entropy = actor(state_batch, rnn_hidden_batch)
                actor_loss += -action_log_prob.mean() * advantage.mean() - cfg.actor_entropy_scale * action_entropy.mean()
                
                # Critic損失
                critic_loss += F.mse_loss(value, target)
            
            # 勾配計算と更新
            model_loss.backward()
            clip_grad_norm_(model_optimizer.param_groups[0]['params'], cfg.gradient_clipping)
            model_optimizer.step()
            
            actor_loss.backward()
            clip_grad_norm_(actor.parameters(), cfg.gradient_clipping)
            actor_optimizer.step()
            
            critic_loss.backward()
            clip_grad_norm_(critic.parameters(), cfg.gradient_clipping)
            critic_optimizer.step()
        
        # 状態更新
        obs = next_obs
        if done:
            obs, _ = env.reset()
            obs = torch.FloatTensor(obs).unsqueeze(0).to(device)
            rnn_hidden = torch.zeros(1, cfg.rnn_hidden_dim).to(device)
            state = torch.zeros(1, cfg.state_dim * cfg.num_classes).to(device)
        
        # 評価
        if step % cfg.eval_freq == 0:
            eval_rewards = []
            for _ in range(cfg.eval_episodes):
                eval_obs, _ = eval_env.reset()
                eval_obs = torch.FloatTensor(eval_obs).unsqueeze(0).to(device)
                eval_rnn_hidden = torch.zeros(1, cfg.rnn_hidden_dim).to(device)
                eval_state = torch.zeros(1, cfg.state_dim * cfg.num_classes).to(device)
                eval_done = False
                total_reward = 0
                
                while not eval_done:
                    with torch.no_grad():
                        embedded_eval_obs = encoder(eval_obs)
                        eval_rnn_hidden = rssm.recurrent(eval_state, torch.zeros(1, eval_env.action_space.n).to(device), eval_rnn_hidden)
                        eval_prior_dist = rssm.get_prior(eval_rnn_hidden)
                        eval_state = eval_prior_dist.sample()
                        eval_action, _, _ = actor(eval_state, eval_rnn_hidden, eval=True)
                    
                    eval_obs, eval_reward, eval_done, _ = eval_env.step(eval_action.cpu().numpy()[0])
                    eval_obs = torch.FloatTensor(eval_obs).unsqueeze(0).to(device)
                    total_reward += eval_reward
                
                eval_rewards.append(total_reward)
            
            print(f"Step: {step}, Eval Reward: {np.mean(eval_rewards):.2f}")

if __name__ == "__main__":
    train()
