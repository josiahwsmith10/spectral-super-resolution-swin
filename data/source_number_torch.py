import torch
import numpy as np

from data.fr import make_hankel_torch

EPS = 1e-10


def SORTE_torch(s, param=20):
    """
    Courtesy of: https://github.com/sreyas-mohan/DeepFreq
    Based on https://www.ese.wustl.edu/~nehorai/paper/Han_Jackknifing_TSP_2013.pdf
    """
    Y = make_hankel_torch(s, param)
    sig = Y @ torch.conj(Y.T)
    eig_values = torch.linalg.eigh(sig)[0].flip(dims=(0,))  # \lamda_1, ..., \lambda_N
    delta_lambda = -torch.diff(eig_values)
    var = var_delta_torch(delta_lambda)
    sorte = torch.divide(var[1:], var[:-1])
    return torch.argmin(sorte) + 1


def var_delta_torch(delta_lambda):
    """
    Courtesy of: https://github.com/sreyas-mohan/DeepFreq
    """
    device = delta_lambda.device
    cummean = torch.cumsum(delta_lambda.flip(dims=(0,)), dim=0) / torch.arange(
        1, len(delta_lambda) + 1, device=device
    )  # mean( \delta_K, ..., \delta_{N-1} )
    delta_lambda_norm = (delta_lambda[None] - cummean[:, None]) ** 2
    var = torch.sum(torch.triu(delta_lambda_norm), axis=1) / torch.arange(
        1, len(delta_lambda) + 1, device=device
    )
    return var


def AIC_torch(s, param=20):
    """
    Courtesy of: https://github.com/sreyas-mohan/DeepFreq
    Based on http://www.dsp-book.narod.ru/DSPMW/67.PDF
    """
    device = s.device
    Y = make_hankel_torch(s, param)
    sig = Y @ torch.conj(Y.T)
    eig_values = torch.linalg.eigh(sig)[0].flip(dims=(0,))  # \lamda_1, ..., \lambda_N
    eig_values = torch.clip(eig_values, EPS, torch.inf)

    cumprod = torch.cumprod(eig_values.flip(dims=(0,)), dim=0).flip(
        dims=(0,)
    )  # , 1/torch.arange(1, len(eig_values)+1)).flip(dims=(0,))
    cummean = torch.cumsum(eig_values.flip(dims=(0,)), dim=0) / torch.arange(
        1, len(eig_values) + 1, device=device
    )
    cummean = torch.pow(
        cummean, torch.arange(1, len(eig_values) + 1, device=device)
    ).flip(dims=(0,))
    log_div = torch.log(cumprod / cummean)
    n_s = torch.arange(1, len(eig_values) + 1, device=device)
    m = len(s)
    n = sig.shape[0]
    k = torch.argmin(-n * log_div + n_s * (2 * m - m))
    return k


def MDL_torch(s, param=20):
    """
    Courtesy of: https://github.com/sreyas-mohan/DeepFreq
    http://www.dsp-book.narod.ru/DSPMW/67.PDF
    """
    device = s.device
    Y = make_hankel_torch(s, param)
    sigma = Y @ torch.conj(Y.T)
    eig_values = torch.linalg.eigh(sigma)[0].flip(dims=(0,))  # \lamda_1, ..., \lambda_N
    eig_values = torch.clip(eig_values, EPS, torch.inf)

    cumprod = torch.cumprod(eig_values.flip(dims=(0,)), dim=0).flip(dims=(0,))
    cummean = torch.cumsum(eig_values.flip(dims=(0,)), dim=0) / torch.arange(
        1, len(eig_values) + 1, device=device
    )
    cummean = torch.pow(
        cummean, torch.arange(1, len(eig_values) + 1, device=device)
    ).flip(dims=(0,))
    log_div = torch.log(cumprod / cummean)
    n_s = torch.arange(1, len(eig_values) + 1, device=device)
    m = len(s)
    n = sigma.shape[0]

    k = torch.argmin(-n * log_div + 0.5 * n_s * (2 * m - n_s) * np.log(n))
    return k


def sorte_arr_torch(signals, param=20):
    """
    Courtesy of: https://github.com/sreyas-mohan/DeepFreq
    """
    result = []
    for s in signals:
        result.append(SORTE_torch(s, param))
    return torch.tensor(result)


def mdl_arr_torch(signals, param=20):
    """
    Courtesy of: https://github.com/sreyas-mohan/DeepFreq
    """
    result = []
    for s in signals:
        result.append(MDL_torch(s, param))
    return torch.tensor(result)


def aic_arr_torch(signals, param=20):
    """
    Courtesy of: https://github.com/sreyas-mohan/DeepFreq
    """
    result = []
    for s in signals:
        result.append(AIC_torch(s, param))
    return torch.tensor(result)
