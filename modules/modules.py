import torch

from .DeepFreq import FrequencyRepresentationModule, PSnet
from .cResFreq import FrequencyRepresentationModule_skiplayer32, cResFreqFast
from .SwinFreq import SwinFreq, CVSwinFreq


def select_model(args):
    """
    Modified from: https://github.com/sreyas-mohan/DeepFreq
    Create a frequency-representation module
    """
    net = None
    if args.fr_module_type.lower() == "psnet":
        assert (
            args.fr_size == args.fr_inner_dim * args.fr_upsampling
        ), "The desired size of the frequency representation (fr_size) must be equal to inner_dim*upsampling"
        net = PSnet(
            signal_dim=args.signal_dim,
            fr_size=args.fr_size,
            n_filters=args.fr_n_filters,
            inner_dim=args.fr_inner_dim,
            n_layers=args.fr_n_layers,
            kernel_size=args.fr_kernel_size,
        )
    elif args.fr_module_type.lower() == "deepfreq":
        assert (
            args.fr_size == args.fr_inner_dim * args.fr_upsampling
        ), "The desired size of the frequency representation (fr_size) must be equal to inner_dim*upsampling"
        net = FrequencyRepresentationModule(
            signal_dim=args.signal_dim,
            n_filters=args.fr_n_filters,
            inner_dim=args.fr_inner_dim,
            n_layers=args.fr_n_layers,
            upsampling=args.fr_upsampling,
            kernel_size=args.fr_kernel_size,
            kernel_out=args.fr_kernel_out,
            out_padding=args.fr_out_padding,
        )
    elif args.fr_module_type.lower() == "cresfreq":
        assert (
            args.fr_size == args.fr_inner_dim * args.fr_upsampling
        ), "The desired size of the frequency representation (fr_size) must be equal to inner_dim*upsampling"
        net = FrequencyRepresentationModule_skiplayer32(
            signal_dim=args.signal_dim,
            n_filters=args.fr_n_filters,
            inner_dim=args.fr_inner_dim,
            n_layers=args.fr_n_layers,
            upsampling=args.fr_upsampling,
            kernel_size=args.fr_kernel_size,
            kernel_out=args.fr_kernel_out,
            out_padding=args.fr_out_padding,
        )
    elif args.fr_module_type.lower() == "cresfreqfast":
        assert (
            args.fr_size == args.fr_inner_dim * args.fr_upsampling
        ), "The desired size of the frequency representation (fr_size) must be equal to inner_dim*upsampling"
        net = cResFreqFast(
            signal_dim=args.signal_dim,
            n_filters=args.fr_n_filters,
            inner_dim=args.fr_inner_dim,
            n_layers=args.fr_n_layers,
            upsampling=args.fr_upsampling,
            kernel_size=args.fr_kernel_size,
            kernel_out=args.fr_kernel_out,
            out_padding=args.fr_out_padding,
        )
    elif args.fr_module_type.lower() == "swinfreq":
        assert (
            args.fr_size == args.fr_inner_dim * args.fr_upsampling
        ), "The desired size of the frequency representation (fr_size) must be equal to inner_dim*upsampling"
        net = SwinFreq(
            signal_dim=args.signal_dim,
            n_filters=args.fr_n_filters,
            inner_dim=args.fr_inner_dim,
            kernel_size=args.fr_kernel_size,
            upsampling=args.fr_upsampling,
            kernel_out=args.fr_kernel_out,
            out_padding=args.fr_out_padding,
            depths=args.fr_depths,
            num_heads=args.fr_num_heads,
            window_size=args.fr_window_size,
            mlp_ratio=args.fr_mlp_ratio,
            normalization=args.normalization,
            dropout=args.fr_dropout,
            optional_relu=args.fr_optional_relu,
        )
    elif args.fr_module_type.lower() == "cvswinfreq":
        assert (
            args.fr_size == args.fr_inner_dim * args.fr_upsampling
        ), "The desired size of the frequency representation (fr_size) must be equal to inner_dim*upsampling"
        net = CVSwinFreq(
            signal_dim=args.signal_dim,
            n_filters=args.fr_n_filters,
            inner_dim=args.fr_inner_dim,
            kernel_size=args.fr_kernel_size,
            upsampling=args.fr_upsampling,
            kernel_out=args.fr_kernel_out,
            out_padding=args.fr_out_padding,
            depths=args.fr_depths,
            num_heads=args.fr_num_heads,
            window_size=args.fr_window_size,
            mlp_ratio=args.fr_mlp_ratio,
            normalization=args.normalization,
            dropout=args.fr_dropout,
        )
    else:
        raise NotImplementedError(
            "Frequency representation module type not implemented"
        )

    return net.to(args.device)
