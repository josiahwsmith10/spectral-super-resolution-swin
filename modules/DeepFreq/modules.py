import torch
import torch.nn as nn


class PSnet(nn.Module):
    """
    Courtesy of: https://github.com/sreyas-mohan/DeepFreq
    """

    def __init__(
        self,
        signal_dim=50,
        fr_size=1000,
        n_filters=8,
        inner_dim=100,
        n_layers=3,
        kernel_size=3,
    ):
        super().__init__()
        self.fr_size = fr_size
        self.num_filters = n_filters
        self.in_layer = nn.Linear(2 * signal_dim, inner_dim, bias=False)
        mod = []

        if torch.__version__ >= "1.7.0":
            conv_padding = "same"
        elif torch.__version__ >= "1.5.0":
            conv_padding = kernel_size // 2
        else:
            conv_padding = kernel_size - 1

        for n in range(n_layers):
            in_filters = n_filters if n > 0 else 1
            mod += [
                nn.Conv1d(
                    in_channels=in_filters,
                    out_channels=n_filters,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=conv_padding,
                    bias=False,
                ),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
            ]
        self.mod = nn.Sequential(*mod)
        self.out_layer = nn.Linear(inner_dim * n_filters, fr_size, bias=True)

    def forward(self, inp):
        bsz = inp.size(0)
        inp = inp.view(bsz, -1)
        x = self.in_layer(inp).view(bsz, 1, -1)
        x = self.mod(x).view(bsz, -1)
        output = self.out_layer(x)
        return output


class FrequencyRepresentationModule(nn.Module):
    """
    Courtesy of: https://github.com/sreyas-mohan/DeepFreq
    """

    def __init__(
        self,
        signal_dim=50,
        n_filters=8,
        n_layers=3,
        inner_dim=125,
        kernel_size=3,
        upsampling=8,
        kernel_out=25,
        out_padding=1,
    ):
        super().__init__()
        self.fr_size = inner_dim * upsampling
        self.n_filters = n_filters
        self.in_layer = nn.Linear(2 * signal_dim, inner_dim * n_filters, bias=False)
        mod = []

        if torch.__version__ >= "1.7.0":
            conv_padding = "same"
        elif torch.__version__ >= "1.5.0":
            conv_padding = kernel_size // 2
        else:
            conv_padding = kernel_size - 1

        for n in range(n_layers):
            mod += [
                nn.Conv1d(
                    n_filters,
                    n_filters,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    bias=False,
                    padding_mode="circular",
                ),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
            ]
        self.mod = nn.Sequential(*mod)
        self.out_layer = nn.ConvTranspose1d(
            n_filters,
            1,
            kernel_out,
            stride=upsampling,
            padding=(kernel_out - upsampling + 1) // 2,
            output_padding=out_padding,
            bias=False,
        )

    def forward(self, inp):
        bsz = inp.size(0)
        inp = inp.view(bsz, -1)
        x = self.in_layer(inp).view(bsz, self.n_filters, -1)
        x = self.mod(x)
        x = self.out_layer(x).view(bsz, -1)
        return x
