# -----------------------------------------------------------------------------------
# Modified for complex-valued 1-D tensor data.
# SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Originally Written by Ze Liu, Modified by Jingyun Liang.
# -----------------------------------------------------------------------------------

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

import complextorch as cvtorch
import complextorch.nn as cvnn

from modules.SwinFreq.swin1d import (
    window_partition1d,
    window_reverse1d,
    PatchEmbed1d,
    PatchUnEmbed1d,
)

ACT = cvnn.CPReLU


class CVMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=ACT,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = cvnn.CVLinear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = cvnn.CVLinear(hidden_features, out_features)
        self.drop = cvnn.CVDropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CVWindowAttention1d(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window. Modified for 1-D tensor data.

    Args:
        dim (int): Number of input channels.
        window_size (int): The size of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1), num_heads, dtype=torch.cfloat)
        )  # 2*window_size - 1, nH
        trunc_normal_(self.relative_position_bias_table.real, std=0.02)
        trunc_normal_(self.relative_position_bias_table.imag, std=0.02)

        # get pair-wise relative position index for each token inside the window
        coords = torch.arange(self.window_size)
        relative_position_index = (
            coords[:, None] - coords[None, :]
        )  # window_size, window_size
        relative_position_index += self.window_size - 1  # shift to start from 0
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = cvnn.CVLinear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = cvnn.CVDropout(attn_drop)
        self.proj = cvnn.CVLinear(dim, dim)

        self.proj_drop = cvnn.CVDropout(proj_drop)
        self.softmax = cvnn.PhaseSoftMax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (B*num_windows, window_size, C)
            mask: (0/-inf) mask with shape of (num_windows, window_size, window_size) or None
        """
        B_, window_size, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, window_size, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # B_, num_heads, window_size, C//num_heads

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # B_, num_heads, window_size, window_size

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            window_size, window_size, self.num_heads
        )  # window_size, window_size, num_heads
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        )  # num_heads, window_size, window_size
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(
                B_ // num_windows, num_windows, self.num_heads, window_size, window_size
            ) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, window_size, window_size)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, window_size, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class CVSSTL(nn.Module):
    r"""Complex-Valued Signal Swin Transformer Layer (CVSSTL).

    Args:
        dim (int): Number of input channels.
        input_resolution (int): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        act_layer=ACT,
        norm_layer=cvnn.CVLayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if self.input_resolution <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = self.input_resolution
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = CVWindowAttention1d(
            dim=dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CVMLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, N):
        # calculate attention mask for SW-MSA
        img_mask = torch.zeros((1, N, 1))  # 1 N 1
        slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for s in slices:
            img_mask[:, s, :] = cnt
            cnt += 1

        mask_windows = window_partition1d(img_mask, self.window_size).view(
            -1, self.window_size
        )  # nW, window_size
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )  # num_windows, window_size, window_size

        return attn_mask

    def forward(self, x):
        B, N, C = x.shape

        shortcut = x
        x = self.norm1(x)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = cvtorch.roll(x, shifts=-self.shift_size, dims=1)  # B, N', C
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition1d(
            shifted_x, self.window_size
        )  # B*num_windows, window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == N:
            attn_windows = self.attn(
                x_windows, mask=self.attn_mask
            )  # B*num_windows, window_size, C
        else:
            attn_windows = self.attn(
                x_windows, mask=self.calculate_mask(N).to(x.device)
            )

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, C)
        shifted_x = window_reverse1d(attn_windows, self.window_size, N)

        # reverse cyclic shift
        if self.shift_size > 0:
            shifted_x = cvtorch.roll(x, shifts=self.shift_size, dims=1)  # B, N, C
        else:
            x = shifted_x

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )

    def flops(self):
        flops = 0
        N = self.input_resolution
        # norm1
        flops += self.dim * N
        # W-MSA/SW-MSA
        num_windows = N / self.window_size
        flops += num_windows * self.attn.flops(self.window_size)
        # mlp
        flops += 2 * N * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * N
        return flops


class CVBasicLayer1d(nn.Module):
    """A basic Swin Transformer layer for one stage. Modified for 1-D tensor data.

    Args:
        dim (int): Number of input channels.
        input_resolution (int): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        norm_layer=cvnn.CVLayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList(
            [
                CVSSTL(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops


class CVSSTB(nn.Module):
    """Complex-Valued Signal Swin Transformer Block (CVSSTB). 
    
    Args:
        dim (int): Number of input channels.
        input_resolution (int): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        norm_layer=cvnn.CVLayerNorm,
        img_size=224,
        patch_size=4,
        resi_connection="1conv",
    ):
        super(CVSSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = CVBasicLayer1d(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            norm_layer=norm_layer,
        )

        if resi_connection == "1conv":
            self.conv = cvnn.CVConv1d(dim, dim, 3, 1, 1)
        elif resi_connection == "3conv":
            # to save parameters and memory
            self.conv = nn.Sequential(
                cvnn.CVConv1d(dim, dim // 4, 3, 1, 1),
                ACT(),
                cvnn.CVConv1d(dim // 4, dim // 4, 1, 1, 0),
                ACT(),
                cvnn.CVConv1d(dim // 4, dim, 3, 1, 1),
            )

        self.patch_embed = PatchEmbed1d(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None,
        )

        self.patch_unembed = PatchUnEmbed1d(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None,
        )

    def forward(self, x):
        res = self.residual_group(x)
        res = self.patch_unembed(res)
        res = self.conv(res)
        res = self.patch_embed(res)
        return res + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        N = self.input_resolution
        flops += N * self.dim * self.dim * 9  # TODO: why 9?
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops
