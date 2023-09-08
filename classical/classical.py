import torch
import torch.nn as nn

from data.fr import periodogram_torch, music_torch, omp_torch
from data.source_number_torch import sorte_arr_torch, mdl_arr_torch, aic_arr_torch


class Periodogram(nn.Module):
    def __init__(self, xgrid):
        super().__init__()
        self.xgrid = torch.tensor(xgrid, dtype=torch.float)

    def cuda(self):
        super().cuda()

        self.xgrid = self.xgrid.cuda()

        return self

    def forward(self, x: torch.Tensor):
        x = x[:, 0] + 1j * x[:, 1]

        return periodogram_torch(x, self.xgrid)


class MUSIC(nn.Module):
    def __init__(self, xgrid, m=20, source_number_method="SORTE", param=20):
        super().__init__()
        self.xgrid = torch.tensor(xgrid, dtype=torch.float)
        self.m = m
        self.source_number_method = source_number_method
        self.param = param

        if source_number_method.lower() == "sorte":
            self.nfreq_estimator = sorte_arr_torch
        elif source_number_method.lower() == "mdl":
            self.nfreq_estimator = mdl_arr_torch
        elif source_number_method.lower() == "aic":
            self.nfreq_estimator = aic_arr_torch

    def cuda(self):
        super().cuda()

        self.xgrid = self.xgrid.cuda()

        return self

    def forward(self, x: torch.Tensor):
        x = x[:, 0] + 1j * x[:, 1]

        nfreqs = self.nfreq_estimator(x, param=self.param)

        return music_torch(x, self.xgrid, nfreqs, m=self.m)


class OMP(nn.Module):
    def __init__(self, signal_dim, fr_size, m, source_number_method="SORTE", param=20):
        super().__init__()

        self.signal_dim = signal_dim
        self.fr_size = fr_size
        self.m = m
        self.source_number_method = source_number_method
        self.param = param

        if source_number_method.lower() == "sorte":
            self.nfreq_estimator = sorte_arr_torch
        elif source_number_method.lower() == "mdl":
            self.nfreq_estimator = mdl_arr_torch
        elif source_number_method.lower() == "aic":
            self.nfreq_estimator = aic_arr_torch

        # create dictionary
        dict_freq = torch.arange(-0.5, 0.5, 1 / fr_size).unsqueeze(0)
        t = torch.arange(signal_dim).unsqueeze(1)
        self.D = torch.exp(1j * 2 * torch.pi * dict_freq * t)  # (signal_dim, fr_size)

    def cuda(self):
        super().cuda()

        self.D = self.D.cuda()

        return self

    def forward(self, x):
        x = x[:, 0] + 1j * x[:, 1]

        nfreqs = self.nfreq_estimator(x, param=self.param)

        y = omp_torch(self.D, x.T, nfreqs).T

        return y.abs() / y.abs().max()
