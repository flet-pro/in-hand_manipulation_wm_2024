from typing import Tuple

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F


class RSSM:
    """
    TransitionModel, ObservationModel, RewardModelの3つをまとめたRSSMクラス．
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        rnn_hidden_dim: int,
        obs_hand_dim: int,
        device: torch.device
    ) -> None:
        """
        コンストラクタ．

        Parameters
        ----------
        state_dim : int
            確率的状態sの次元数．
        action_dim : int
            行動空間の次元数．
        rnn_hidden_dim : int
            決定的状態hの次元数．
        """
        self.transition = TransitionModel(state_dim, action_dim, rnn_hidden_dim, obs_hand_dim).to(
            device
        )
        self.observation = ObservationModel(
            state_dim,
            rnn_hidden_dim,
        ).to(device)
        self.reward = RewardModel(
            state_dim,
            rnn_hidden_dim,
        ).to(device)


class TransitionModel(nn.Module):
    """
    状態遷移を担うクラス．このクラスは複数の要素を含んでいます．
    決定的状態遷移 （RNN) : h_t+1 = f(h_t, s_t, a_t)
    確率的状態遷移による1ステップ予測として定義される "prior" : p(s_t+1 | h_t+1)
    観測の情報を取り込んで定義される "posterior": q(s_t+1 | h_t+1, e_t+1)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        rnn_hidden_dim: int,
        obs_hand_dim: int,
        hidden_dim: int = 200,
        min_stddev: float = 0.1,
        act: "function" = F.elu,
    ) -> None:
        """
        コンストラクタ．

        Parameters
        ----------
        state_dim : int
            確率的状態sの次元数．
        action_dim : int
            行動空間の次元数．
        rnn_hidden_dim : int
            決定的状態遷移を計算するRNNの隠れ層の次元数．
        hidden_dim : int
            決定的状態hの次元数．
        min_stddev : float
            確率状態遷移の標準偏差の最小値．
        act : function
            活性化関数．
        """
        super(TransitionModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.fc_state_action = nn.Linear(state_dim + action_dim, hidden_dim)

        self.fc_rnn_hidden = nn.Linear(rnn_hidden_dim, hidden_dim)
        self.fc_state_mean_prior = nn.Linear(hidden_dim, state_dim)
        self.fc_state_stddev_prior = nn.Linear(hidden_dim, state_dim)
        self.fc_rnn_hidden_embedded_obs = nn.Linear(rnn_hidden_dim + 1024 + obs_hand_dim, hidden_dim) # 1024 -> 153に変更
        self.fc_state_mean_posterior = nn.Linear(hidden_dim, state_dim)
        self.fc_state_stddev_posterior = nn.Linear(hidden_dim, state_dim)

        # next hidden stateを計算
        self.rnn = nn.GRUCell(hidden_dim, rnn_hidden_dim)
        self._min_stddev = min_stddev
        self.act = act

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        rnn_hidden: torch.Tensor,
        embedded_next_obs: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        """
        prior p(s_t+1 | h_t+1) と posterior q(s_t+1 | h_t+1, e_t+1) を返すメソッド．
        この2つが近づくように学習する．

        Parameters
        ----------
        state : torch.Tensor (batch size, state dim)
            時刻tの状態(s_t)．
        action : torch.Tensor (batch size, action dim)
            時刻tの行動(a_t)．
        rnn_hidden : torch.Tensor (batch size, rnn hidden dim)
            RNNが保持している決定的状態(h_t)．
        embedded_next_obs : torch.Tensor (batch size, 1024)
            時刻t+1の観測をエンコードしたもの(e_t+1)．

        Returns
        -------
        next_state_prior : torch.Tensor (batch size, state dim)
            prior(p(s_t+1 | h_t+1))による次の時刻の状態の予測．
        next_state_posterior : torch.Tensor (batch size, state dim)
            posterior(q(s_t+1 | h_t+1, e_t+1))による次の時刻の状態の予測．
        rnn_hidden : torch.Tensor (batch size, rnn hidden dim)
            RNNが保持する次の決定的状態(h_t+1)．
        """
        next_state_prior, rnn_hidden = self.prior(
            self.recurrent(state, action, rnn_hidden)
        )
        next_state_posterior = self.posterior(rnn_hidden, embedded_next_obs)
        return next_state_prior, next_state_posterior, rnn_hidden

    def recurrent(
        self, state: torch.Tensor, action: torch.Tensor, rnn_hidden: torch.Tensor
    ) -> torch.Tensor:
        """
        決定的状態 h_t+1 = f(h_t, s_t, a_t)を計算するメソッド．

        Parameters
        ----------
        state : torch.Tensor (batch size, state dim)
            時刻tの状態(s_t)．
        action : torch.Tensor (batch size, action dim)
            時刻tの行動(a_t)．
        rnn_hidden : torch.Tensor (batch size, rnn hidden dim)
            RNNが保持している決定的状態(h_t)．

        Returns
        -------
        rnn_hidden : torch.Tensor (batch size, rnn hidden dim)
            RNNが保持する次の決定的状態(h_t+1)．
        """
        hidden = self.act(self.fc_state_action(torch.cat([state, action], dim=1)))
        # h_t+1を求める
        rnn_hidden = self.rnn(hidden, rnn_hidden)
        return rnn_hidden

    def prior(self, rnn_hidden: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        prior p(s_t+1 | h_t+1) を計算するメソッド．

        Parameters
        ----------
        rnn_hidden : torch.Tensor (batch size, rnn hidden dim)
            RNNが保持している決定的状態(h_t+1)．

        Returns
        -------
        state : torch.Tensor (batch size, state dim)
            決定的状態を用いてサンプリングされた確率的な状態(s_t+1)．
            ここでは決定的状態h_t+1からガウス分布の平均，標準偏差を推定してサンプリングしています．
        rnn_hidden : torch.Tensor (batch size, rnn hidden dim)
            RNNが保持する決定的状態(h_t+1)．
            入力からのものをそのまま返しています．
        """
        #h_t+1を求める（ヒント: self.act, self.fc_rnn_hiddenを使用）
        hidden = self.act(self.fc_rnn_hidden(rnn_hidden)) # WRITE ME

        mean = self.fc_state_mean_prior(hidden)
        stddev = F.softplus(self.fc_state_stddev_prior(hidden)) + self._min_stddev
        return Normal(mean, stddev), rnn_hidden

    def posterior(
        self, rnn_hidden: torch.Tensor, embedded_obs: torch.Tensor
    ) -> torch.Tensor:
        """
        posterior q(s_t+1 | h_t+1, e_t+1)  を計算するメソッド．

        Parameters
        ----------
        rnn_hidden : torch.Tensor (batch size, rnn hidden dim)
            RNNが保持している決定的状態(h_t+1)．
        embedded_obs : torch.Tensor (batch size, 1024)
            時刻t+1の観測をエンコードしたもの．

        Returns
        -------
        state : torch.Tensor (batch size, state dim)
            決定的状態とエンコードした観測を用いてサンプリングされた確率的な状態(s_t+1)．
            ここでは決定的状態h_t+1とエンコードした観測e_t+1からガウス分布の平均，標準偏差を推定してサンプリングしています．
        """
        # h_t+1，o_t+1を結合し，q(s_t+1 | h_t+1, e_t+1) を計算する
        # print("rnn_hidden shape: ", rnn_hidden.shape)
        # print("rnn_hidden dtype: ", rnn_hidden.dtype)
        # print("embedded_obs shape: ", embedded_obs.shape)
        # print("embedded_obs dtype: ", embedded_obs.dtype)
        hidden = self.act(self.fc_rnn_hidden_embedded_obs(torch.cat([rnn_hidden, embedded_obs], dim=1))) # WRITE ME
        mean = self.fc_state_mean_posterior(hidden)
        stddev = F.softplus(self.fc_state_stddev_posterior(hidden)) + self._min_stddev
        return Normal(mean, stddev)
    

class ObservationModel(nn.Module):
    """
    p(o_t | s_t, h_t)
    低次元の状態表現から画像を再構成するデコーダ (3, 64, 64)
    """

    def __init__(self, state_dim: int, rnn_hidden_dim: int) -> None:
        """
        コンストラクタ．

        Parameters
        ----------
        state_dim : int
            確率的状態sの次元数．
        rnn_hidden_dim : int
            決定的状態hの次元数．
        """
        super(ObservationModel, self).__init__()
        self.fc = nn.Linear(state_dim + rnn_hidden_dim, 1024)
        self.dc1 = nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2)
        self.dc2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.dc3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2)
        self.dc4 = nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2)

    def forward(self, state: torch.Tensor, rnn_hidden: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行うメソッド．確率的状態sと決定的状態hから観測を再構成する．

        Parameters
        ----------
        state : torch.Tensor (batch size, state dim)
            確率的状態s．
        rnn_hidden : torch.Tensor (batch size, rnn_hidden_dim)
            決定的状態h．

        Returns
        -------
        obs : torch.Tensor (batch size, 3, 64, 64)
            再構成された観測o．
        """
        hidden = self.fc(torch.cat([state, rnn_hidden], dim=1))
        hidden = hidden.view(hidden.size(0), 1024, 1, 1)
        hidden = F.relu(self.dc1(hidden))
        hidden = F.relu(self.dc2(hidden))
        hidden = F.relu(self.dc3(hidden))
        obs = self.dc4(hidden)
        return obs


class RewardModel(nn.Module):
    """
    p(r_t | s_t, h_t)
    低次元の状態表現から報酬を予測する．
    """

    def __init__(
        self,
        state_dim: int,
        rnn_hidden_dim: int,
        hidden_dim: int = 400,
        act: "function" = F.elu,
    ) -> None:
        """
        コンストラクタ．

        Parameters
        ----------
        state_dim : int
            確率的状態sの次元数．
        rnn_hidden_dim : int
            決定的状態hの次元数．
        hidden_dim : int
            報酬モデルの隠れ層の次元数． (default=400)
        act : function
            報酬モデルに利用される活性化関数． (default=torch.nn.functional.elu)
        """
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.act = act

    def forward(self, state: torch.Tensor, rnn_hidden: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行うメソッド．確率的状態sと決定的状態hから報酬rを推定する．

        Parameters
        ----------
        state : torch.Tensor (batch size, state dim)
            確率的状態s．
        rnn_hidden : torch.Tensor (batch size, rnn_hidden_dim)
            決定的状態h．

        Returns
        -------
        reward : torch.Tensor (batch size, 1)
            確率的状態s，決定的状態hに対する報酬r．
        """
        hidden = self.act(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = self.act(self.fc2(hidden))
        hidden = self.act(self.fc3(hidden))
        reward = self.fc4(hidden)
        return reward
