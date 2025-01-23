import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    """
    (3, 64, 64)の画像を(1024,)のベクトルに変換し，ロボットハンドの観測環境（153,）と連結を行うエンコーダクラス．
    """

    def __init__(self) -> None:
        """
        コンストラクタ．
        層の定義のみを行う．
        """
        super(Encoder, self).__init__()
        self.cv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2)
        self.cv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.cv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.cv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2)

    def forward(self, obs: torch.Tensor, obs_hand: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行うメソッド．観測画像をベクトルに埋め込む．

        Parameters
        ----------
        obs : torch.Tensor (batch size, 3, 64, 64)
            環境から得られた観測画像．

        Returns
        -------
        embedded_obs : torch.Tensor (batch size, 1024+153)
            観測を1024+153次元のベクトルに埋め込んだもの．
        """
        hidden = F.relu(self.cv1(obs))
        hidden = F.relu(self.cv2(hidden))
        hidden = F.relu(self.cv3(hidden))
        embedded_obs = F.relu(self.cv4(hidden)).reshape(hidden.size(0), -1)
        # print("embedded_obs shape: ", embedded_obs.shape)
        # print("obs_hand shape: ", obs_hand.shape)
        embedded_obs = torch.cat([embedded_obs, obs_hand], dim=-1)
        return embedded_obs