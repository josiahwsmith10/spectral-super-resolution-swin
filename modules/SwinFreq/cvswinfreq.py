"""
Summary: simply added channel attention to cResFreq residual module
"""

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

import complextorch.nn as cvnn
from complextorch import CVTensor

from modules.SwinFreq.swin1d import PatchEmbed1d, PatchUnEmbed1d
from modules.SwinFreq.cvswin1d import CVSSTB
from modules.cResFreq.cresfreqfast import TConvRB, MFB


class CVSwinBodyBlock(nn.Module):
    """Swin Body Block."""

    def __init__(
        self,
        N=256,
        patch_size=1,
        embed_dim=32,
        depths=[6, 6, 6, 6],
        num_heads=[4, 4, 4, 4],
        window_size=8,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=cvnn.CVLayerNorm,
        patch_norm=True,
        resi_connection="1conv",
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed1d(
            img_size=N,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed1d(
            img_size=N,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        self.pos_drop = cvnn.CVDropout(p=drop_rate)

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = CVSSTB(
                dim=embed_dim,
                input_resolution=patches_resolution,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                img_size=N,
                patch_size=patch_size,
                resi_connection=resi_connection,
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == "1conv":
            self.conv_after_body = cvnn.CVConv1d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "3conv":
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                cvnn.CVConv1d(embed_dim, embed_dim // 4, 3, 1, 1),
                cvnn.CVCardiod(),
                cvnn.CVConv1d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                cvnn.CVCardiod(),
                cvnn.CVConv1d(embed_dim // 4, embed_dim, 3, 1, 1),
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, cvnn.CVLinear):
            trunc_normal_(m.weight.real, std=0.02)
            trunc_normal_(m.weight.imag, std=0.02)
            if isinstance(m, cvnn.CVLinear) and m.bias is not None:
                nn.init.constant_(m.bias.real, 0)
                nn.init.constant_(m.bias.imag, 0)
        elif isinstance(m, cvnn.CVLayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # (B, N, C)
        x = self.patch_unembed(x)

        return x

    def forward(self, x):
        x = self.conv_after_body(self.forward_features(x)) + x

        return x.abs()


class CVSwinFreq(nn.Module):
    """
    Modified from: https://github.com/panpp-git/cResFreq
    """

    def __init__(
        self,
        signal_dim=64,
        n_filters=32,
        inner_dim=256,
        kernel_size=3,
        upsampling=16,
        kernel_out=18,
        out_padding=0,
        depths=[8, 8, 8],
        num_heads=[8, 8, 8],
        window_size=16,
        mlp_ratio=2.0,
        normalization=True,
        dropout=0.0,
    ):
        super().__init__()
        self.signal_dim = signal_dim
        self.n_filters = n_filters
        self.inner_dim = inner_dim
        self.kernrel_size = kernel_size
        self.upsampling = upsampling
        self.kernel_out = kernel_out
        self.out_padding = out_padding
        self.normalization = normalization
        self.dropout = dropout

        self.fr_size = inner_dim * upsampling

        self.mf = MFB(
            signal_dim=signal_dim,
            n_filters=n_filters,
            inner_dim=inner_dim,
            kernel_size=kernel_size,
        )

        self.res_layers = CVSwinBodyBlock(
            N=inner_dim,
            embed_dim=n_filters,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_rate=dropout,
            attn_drop_rate=dropout,
            resi_connection="1conv",
        )

        self.recon = TConvRB(
            in_channels=n_filters,
            out_channels=1,
            kernel_out=kernel_out,
            upsampling=upsampling,
            out_padding=out_padding,
        )

    def min_max_freq(self, x: CVTensor):
        # Computes min-max norm values in frequency domain
        # x is (batch_size x N)

        y = x.fft(dim=1)

        y_min = y.abs().min(dim=1, keepdim=True)[0]
        y_max = y.abs().max(dim=1, keepdim=True)[0]

        return 1 / (y_max - y_min), -y_min / (y_max - y_min)

    def norm(self, x, kind="min-max"):
        c = x[:, 0][:, None]

        x = x / c

        if kind == "min-max":
            a, b = self.min_max_freq(x)
        elif kind == "abs":
            a = 1 / x.abs().max(dim=1, keepdim=True)[0]
            b = 0

        return a * x + b

    def forward(self, x):
        x = CVTensor(x[:, 0], x[:, 1])  # batch_size x N

        if self.normalization:
            x = self.norm(x)

        x = self.mf(x)

        x = self.res_layers(x)

        return self.recon(x)
