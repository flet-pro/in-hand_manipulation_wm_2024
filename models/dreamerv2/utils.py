import numpy as np
import torch


def preprocess_obs(obs: np.ndarray) -> np.ndarray:
    """
    画像を正規化する．[0, 255] -> [-0.5, 0.5]．

    Parameters
    ----------
    obs : np.ndarray (64, 64, 3) or (chank length, batch size, 64, 64, 3)
        環境から得られた観測．画素値は[0, 255]．

    Returns
    -------
    normalized_obs : np.ndarray (64, 64, 3) or (chank length, batch size, 64, 64, 3)
        画素値を[-0.5, 0.5]で正規化した観測．
    """
    obs = obs.astype(np.float32)
    normalized_obs = obs / 255.0 - 0.5
    return normalized_obs


def calculate_lambda_target(rewards: torch.Tensor, discounts: torch.Tensor, values: torch.Tensor, lambda_: float):
    """
    lambda targetを計算する関数．

    Parameters
    ---------
    rewards : torch.Tensor (imagination_horizon, D)
        報酬．1次元目が時刻tを表しており，2次元目は自由な次元数にでき，想像の軌道を作成するときに入力されるサンプルindexと考える．
    discounts : torch.Tensor (imagination_horizon, D)
        割引率．gammaそのままを利用するのではなく，DiscountModelの出力をかけて利用する．
    values : torch.Tensor (imagination_horizon, D)
        状態価値関数．criticで予測された値を利用するが，Dreamer v2ではtarget networkで計算する．
    lambda_ : float
        lambda targetのハイパラ．

    Returns
    -------
    V_lambda : torch.Tensor (imagination_horizon, D)
        lambda targetの値．
    """
    V_lambda = torch.zeros_like(rewards)

    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            V_lambda[t] = rewards[t] + discounts[t] * values[t]  # t=Hの場合（式4の下の条件）
        else:
            V_lambda[t] = rewards[t] + discounts[t] * ((1-lambda_) * values[t+1] + lambda_ * V_lambda[t+1])

    return V_lambda
