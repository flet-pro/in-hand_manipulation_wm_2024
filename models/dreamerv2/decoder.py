import torch
import torch.distributions as td
from torch.distributions import Normal, Categorical, OneHotCategorical, OneHotCategoricalStraightThrough
from torch.distributions.kl import kl_divergence
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

class Decoder(nn.Module):
    def __init__(self, rnn_hidden_dim: int, state_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(state_dim*num_classes + rnn_hidden_dim, 1536)
        self.dc1 = nn.ConvTranspose2d(1536, 192, kernel_size=5, stride=2)
        self.dc2 = nn.ConvTranspose2d(192, 96, kernel_size=5, stride=2)
        self.dc3 = nn.ConvTranspose2d(96, 48, kernel_size=6, stride=2)
        self.dc4 = nn.ConvTranspose2d(48, 1, kernel_size=6, stride=2)


    def forward(self, state: torch.Tensor, rnn_hidden: torch.Tensor):
        """
        決定論的状態と，確率的状態を入力として，観測画像を復元するDecoder．
        出力は多次元正規分布の平均値をとる．

        Paremters
        ---------
        state : torch.Tensor (B, state_dim * num_classes)
            確率的状態．
        rnn_hidden : torch.Tensor (B, rnn_hidden_dim)
            決定論的状態．

        Returns
        -------
        obs_dist : torch.distribution.Independent
            観測画像を再構成するための多次元正規分布．
        """
        hidden = self.fc(torch.cat([state, rnn_hidden], dim=1))
        hidden = hidden.view(hidden.size(0), 1536, 1, 1)
        hidden = F.elu(self.dc1(hidden))
        hidden = F.elu(self.dc2(hidden))
        hidden = F.elu(self.dc3(hidden))
        mean = self.dc4(hidden)

        obs_dist = td.Independent(td.Normal(mean, 1), 3)
        return obs_dist  # p(\hat{x}_t | h_t, z_t)