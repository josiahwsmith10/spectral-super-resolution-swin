import torch.nn as nn
import torch
from .complexLayers import ComplexLinear, ComplexConv2d


class FrequencyRepresentationModule_skiplayer32(nn.Module):
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
        kernel_out=18,
        out_padding=0,
    ):
        super().__init__()
        self.fr_size = inner_dim * upsampling

        self.n_filters = n_filters

        self.inner = inner_dim
        self.n_layers = n_layers

        self.in_layer = ComplexLinear(signal_dim, inner_dim * int(n_filters / 8))

        self.in_layer2 = ComplexConv2d(
            1,
            int(n_filters / 4),
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2),
            bias=False,
        )

        mod = []
        for i in range(self.n_layers):
            tmp = []
            tmp += [
                nn.Conv1d(
                    n_filters,
                    n_filters,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                    padding_mode="circular",
                ),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
                nn.Conv1d(
                    n_filters,
                    n_filters,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                    padding_mode="circular",
                ),
                nn.BatchNorm1d(n_filters),
            ]
            mod += [nn.Sequential(*tmp)]
        self.mod = nn.Sequential(*mod)
        activate_layer = []

        for i in range(self.n_layers):
            activate_layer += [nn.ReLU()]
        self.activate_layer = nn.Sequential(*activate_layer)

        self.out_layer = nn.ConvTranspose1d(
            n_filters,
            1,
            kernel_out,
            stride=upsampling,
            padding=(kernel_out - upsampling + 1) // 2,
            output_padding=out_padding,
            bias=False,
        )

    def forward(self, x):
        bsz = x.size(0)
        inp = x[:, 0, :].type(torch.complex64) + 1j * x[:, 1, :].type(torch.complex64)

        # Input normalization
        mv = inp.abs().max(dim=1, keepdim=True)[0]
        inp = inp / mv

        x = self.in_layer(inp).view(bsz, 1, int(self.n_filters / 8), -1)

        x = self.in_layer2(x).view(bsz, self.n_filters, -1)

        x = x.abs()

        for i in range(self.n_layers):
            res_x = self.mod[i](x)
            x = res_x + x
            x = self.activate_layer[i](x)

        x = self.out_layer(x).view(bsz, -1)
        return x
