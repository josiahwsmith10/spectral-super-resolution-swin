#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:30:02 2019
@author: Sebastien M. Popoff
Based on https://openreview.net/forum?id=H1T2hmZAb
"""

import torch
import torch.nn as nn


def apply_complex(fr, fi, x_in):
    """
    Courtesy of: https://github.com/panpp-git/cResFreq
    """
    return (fr(x_in.real) - fi(x_in.imag)).type(torch.complex64) + 1j * (
        fr(x_in.imag) + fi(x_in.real)
    ).type(torch.complex64)


class ComplexConv2d(nn.Module):
    """
    Courtesy of: https://github.com/panpp-git/cResFreq
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(ComplexConv2d, self).__init__()
        self.conv_r = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.conv_i = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

    def forward(self, x_in):
        return apply_complex(self.conv_r, self.conv_i, x_in)


class ComplexLinear(nn.Module):
    """
    Courtesy of: https://github.com/panpp-git/cResFreq
    """

    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = nn.Linear(in_features, out_features, bias=False)
        self.fc_i = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x_in):
        return apply_complex(self.fc_r, self.fc_i, x_in)
