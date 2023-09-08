"""
Summary: reverted to original cResFreq faster implementation
"""

import torch.nn as nn
import numpy as np

import complextorch.nn as cvnn
from complextorch import CVTensor


class TConvRB(nn.Module):
    """Transposed Convolution Reconstruction Block."""

    def __init__(self, in_channels, out_channels, kernel_out, upsampling, out_padding):
        super().__init__()
        self.upsampling = upsampling
        self.kernel_out = kernel_out
        self.out_padding = out_padding

        self.tconv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_out,
            stride=upsampling,
            padding=(kernel_out - upsampling + 1) // 2,
            output_padding=out_padding,
            bias=False,
        )

    def forward(self, x):
        batch_size = x.shape[0]

        return self.tconv(x).view(batch_size, -1)


class MFB(nn.Module):
    """Matched Filter Block."""

    def __init__(self, signal_dim=64, n_filters=32, inner_dim=256, kernel_size=3):
        super().__init__()
        self.signal_dim = signal_dim
        self.n_filters = n_filters
        self.inner_dim = inner_dim
        self.kernrel_size = kernel_size

        assert np.log2(n_filters) == np.floor(
            np.log2(n_filters)
        ), "n_filters must be power of 2"

        # Ensures output feature dimension is inner_dim
        self.large_factor = int(2 ** np.ceil(np.log2(n_filters) / 2))
        self.small_factor = int(2 ** np.floor(np.log2(n_filters) / 2))

        # Need to use bias=False for cvnn
        self.linear = cvnn.CVLinear(
            in_features=signal_dim,
            out_features=inner_dim * n_filters // self.large_factor,
            bias=False,
        )

        self.conv = cvnn.CVConv2d(
            in_channels=1,
            out_channels=n_filters // self.small_factor,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2),
            bias=False,
        )

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.linear(x).view(batch_size, 1, self.n_filters // self.large_factor, -1)

        return self.conv(x).view(batch_size, self.n_filters, -1)


class ResBlock(nn.Module):
    """cResFreq residual block."""

    def __init__(
        self,
        channels,
        kernel_size,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
            padding_mode="circular",
        )

        self.conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
            padding_mode="circular",
        )

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.relu(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))
        return self.relu(x + res)


class cResFreqFast(nn.Module):
    """
    Modified from: https://github.com/panpp-git/cResFreq
    """

    def __init__(
        self,
        signal_dim=64,
        n_filters=32,
        n_layers=24,
        inner_dim=256,
        kernel_size=3,
        upsampling=16,
        reduction_factor=4,
        kernel_out=18,
        out_padding=0,
    ):
        super().__init__()
        self.signal_dim = signal_dim
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.inner_dim = inner_dim
        self.kernrel_size = kernel_size
        self.upsampling = upsampling
        self.reduction_factor = reduction_factor
        self.kernel_out = kernel_out
        self.out_padding = out_padding

        self.fr_size = inner_dim * upsampling

        self.mf = MFB(
            signal_dim=signal_dim,
            n_filters=n_filters,
            inner_dim=inner_dim,
            kernel_size=kernel_size,
        )

        res_layers = [
            ResBlock(
                channels=n_filters,
                reduction_factor=reduction_factor,
                kernel_size=kernel_size,
            )
            for _ in range(self.n_layers)
        ]
        self.res_layers = nn.Sequential(*res_layers)

        self.recon = TConvRB(
            in_channels=n_filters,
            out_channels=1,
            kernel_out=kernel_out,
            upsampling=upsampling,
            out_padding=out_padding,
        )

    def forward(self, x):
        x = CVTensor(x[:, 0], x[:, 1])  # batch_size x N

        # Input normalization
        x = x / x.abs().max(dim=1, keepdim=True)[0]

        x = self.mf(x)

        x = x.abs()

        x = self.res_layers(x)

        return self.recon(x)
