import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from models.dreamer.encoder import Encoder
from models.dreamer.rssm import RSSM
from models.dreamer.utils import preprocess_obs


class ValueModel(nn.Module):
    """
    低次元の状態表現(state_dim + rnn_hidden_dim)から状態価値を出力するクラス．
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
            モデルの隠れ層の次元数． (default=400)
        act : function
            モデルの活性化関数． (default=torch.nn.functional.elu)
        """
        super(ValueModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.act = act

    def forward(self, state: torch.Tensor, rnn_hidden: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行うメソッド．低次元の状態表現から状態価値を推定する．

        Parameters
        ----------
        state : torch.Tensor (batch size, state dim)
            確率的状態s．
        rnn_hidden : torch.Tensor (batch size, rnn_hidden_dim)
            決定的状態h．

        Returns
        -------
        state_value : torch.Tensor (batch size, 1)
            入力された状態に対する状態価値の推定値．
        """
        hidden = self.act(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = self.act(self.fc2(hidden))
        hidden = self.act(self.fc3(hidden))
        state_value = self.fc4(hidden)
        return state_value


class ActionModel(nn.Module):
    """
    低次元の状態表現(state_dim + rnn_hidden_dim)から行動を計算するクラス．
    """

    def __init__(
            self,
            state_dim: int,
            rnn_hidden_dim: int,
            action_dim: int,
            hidden_dim: int = 400,
            act: "function" = F.elu,
            min_stddev: float = 1e-4,
            init_stddev: float = 5.0,
    ) -> None:
        """
        コンストラクタ．

        Parameters
        ----------
        state_dim : int
            確率的状態sの次元数．
        rnn_hidden_dim : int
            決定的状態hの次元数．
        action_dim : int
            行動空間の次元数．
        hidden_dim : int
            モデルの隠れ層の次元数． (default=400)
        act : function
            モデルの活性化関数． (default=torch.nn.functional.elu)
        min_stddev : float
            行動をサンプリングする分布の標準偏差の最小値． (default=1e-4)
        init_stddev : float
            行動をサンプリングする分布の標準偏差の初期値． (default=5.0)
        """
        super(ActionModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_stddev = nn.Linear(hidden_dim, action_dim)
        self.act = act
        self.min_stddev = min_stddev
        self.init_stddev = np.log(np.exp(init_stddev) - 1)

    def forward(
            self, state: torch.Tensor, rnn_hidden: torch.Tensor, training: bool = True
    ) -> None:
        """
        順伝播を行うメソッド．入力された状態に対する行動を出力する．
        training=Trueなら，NNのパラメータに関して微分可能な形の行動のサンプル（Reparametrizationによる）を返す．
        training=Falseなら，行動の確率分布の平均値を返す．

        Parameters
        ----------
        staet : torch.Tensor (batch size, state dim)
            確率的状態s．
        rnn_hidden : torch.Tensor (batch size, rnn_hidden_dim)
            決定的状態h．
        training : bool
            訓練か推論かを示すフラグ． (default=True)

        Returns
        -------
        action : torch.Tensor (batch size, action dim)
            入力された状態に対する行動．
            training=Trueでは微分可能な形の行動をサンプリングした値，
            training=Falseでは行動の確率分布の平均値を返す．
        """
        hidden = self.act(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = self.act(self.fc2(hidden))
        hidden = self.act(self.fc3(hidden))
        hidden = self.act(self.fc4(hidden))

        # Dreamerの実装に合わせて少し平均と分散に対する簡単な変換が入っています
        mean = self.fc_mean(hidden)
        mean = 5.0 * torch.tanh(mean / 5.0)
        stddev = self.fc_stddev(hidden)
        stddev = F.softplus(stddev + self.init_stddev) + self.min_stddev

        if training:
            action = torch.tanh(Normal(mean, stddev).rsample())  # 微分可能にするためrsample()
        else:
            action = torch.tanh(mean)
        return action


class Agent:
    """
    ActionModelに基づき行動を決定する．そのためにRSSMを用いて状態表現をリアルタイムで推論して維持するクラス．
    """

    def __init__(self, encoder: Encoder, rssm: RSSM, action_model: ActionModel) -> None:
        """
        コンストラクタ．

        Parameters
        ----------
        encoder : Encoder
            上で定義したEncoderクラスのインスタンス．
            観測画像を1024次元のベクトルに埋め込む ．
        rssm : RSSM
            上で定義したRSSMクラスのインスタンス．
            遷移モデル，1024次元のベクトルを観測画像にするデコーダ，報酬を予測するモデルを持つ．
        action_model : ActionModel
            上で定義したActionModelのインスタンス．
            低次元の状態表現から行動を予測する．
        """
        self.encoder = encoder
        self.rssm = rssm
        self.action_model = action_model

        self.device = next(self.action_model.parameters()).device
        self.rnn_hidden = torch.zeros(1, rssm.rnn_hidden_dim, device=self.device)

    def __call__(self, obs: np.ndarray, obs_hand: np.ndarray, training=True) -> np.ndarray:
        """
        特殊メソッド．
        インスタンスに直接引数を渡すことで実行される．
        （例）agent = Agent(*args)
             action = agent(obs)  # このときに__call__メソッドが呼び出される．

        Parameters
        ----------
        obs : np.ndarray (batch size, 3, 64, 64)
            環境から得られた観測画像．
        training : bool
            訓練か推論かを示すフラグ． (default=True)

        Returns
        -------
        action : np.ndarray (batch size, action dim)
            入力された観測に対する行動の予測．
        """
        # preprocessを適用，PyTorchのためにChannel-Firstに変換
        obs = preprocess_obs(obs)
        obs = torch.as_tensor(obs, device=self.device)
        # print("obs shape: ", obs.shape)
        obs = obs.transpose(1, 2).transpose(0, 1).unsqueeze(0)

        obs_hand = torch.as_tensor(obs_hand, device=self.device).unsqueeze(0)

        with torch.no_grad():
            # 観測を低次元の表現に変換し，posteriorからのサンプルをActionModelに入力して行動を決定する
            embedded_obs = self.encoder(obs, obs_hand).to(torch.float32)
            state_posterior = self.rssm.posterior(self.rnn_hidden, embedded_obs)
            state = state_posterior.sample()
            action = self.action_model(state, self.rnn_hidden, training=training)

            # 次のステップのためにRNNの隠れ状態を更新しておく
            _, self.rnn_hidden = self.rssm.prior(
                self.rssm.recurrent(state, action, self.rnn_hidden)
            )

        return action.squeeze().cpu().numpy()

    def reset(self) -> None:
        """
        RNNの隠れ状態（=決定的状態）をリセットする．
        """
        self.rnn_hidden = torch.zeros(1, self.rssm.rnn_hidden_dim, device=self.device)
