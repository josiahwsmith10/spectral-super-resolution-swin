"""
Experiments for CV/SwinFreq paper. 
- Experiment 0: number of parameters
- Experiment 1: PSNR / SSIM across SNR
- Experiment 2: resolution capability
- Experiment 3: comparison of sidelobes
- Experiment 4: spinning point scatterers
- Experiment 5: real plane data
"""

import numpy as np
import torch
import argparse

import util

from paper_results import (
    create_methods,
    experiment0,
    experiment1,
    experiment2,
    experiment3,
    experiment4,
    experiment5,
)


def main():
    args = setup()

    create_methods(args)

    experiment0(args)
    experiment1(args)
    experiment2(args)
    experiment3(args)
    experiment4(args)
    experiment5(args)


def setup():
    parser = argparse.ArgumentParser()

    # methods to compare performance for experiments
    parser.add_argument(
        "--method_list",
        nargs="+",
        type=str,
        default=[
            "Periodogram",
            "MUSIC",
            "FISTA",
            #"saved/models/cresfreq.pth",
            #"saved/models/cvswinfreq222.pth",
            #"saved/models/swinfreq444.pth",
        ],
        help="methods to use (use path to checkpoint for ML models)",
    )

    # experiment 1 settings
    parser.add_argument(
        "--snr_list",
        nargs="+",
        type=int,
        default=[-10, 0, 10, 20, 30, 40],
        help="SNR values in dB for comparing numerical performance",
    )
    parser.add_argument(
        "--exp1_metrics_list",
        nargs="+",
        type=int,
        default=["PSNR", "SSIM"],
        help="list of metrics to use for experiment 1 (PSNR, SSIM)",
    )
    parser.add_argument(
        "--exp1_num_samples",
        type=int,
        default=10,
        help="number of samples used in experiment 1 per SNR value",
    )
    parser.add_argument(
        "--music_m",
        type=int,
        default=20,
        help="m parameter for MUSIC calculation",
    )
    parser.add_argument(
        "--source_number_method",
        type=str,
        default="SORTE",
        help="method to estimate number of frequencies in MUSIC & OMP",
    )
    parser.add_argument(
        "--source_number_param",
        type=int,
        default=20,
        help="m=param parameter for source number calculation",
    )

    # experiment 2 settings
    parser.add_argument(
        "--res_list",
        nargs="+",
        type=int,
        default=np.linspace(0.3, 1.0, 10),
        help="resolution spacing for computing resolution capability (x/N)",
    )
    parser.add_argument(
        "--res_snr",
        type=int,
        default=10,
        help="signal-to-noise ratio for resolution experiment",
    )
    parser.add_argument(
        "--res_tol",
        type=int,
        default=5,
        help="number of bins on either side to allow to find maximum of peak",
    )
    parser.add_argument(
        "--exp2_num_samples",
        type=int,
        default=100,
        help="number of samples used in experiment 2 per resolution spacing",
    )

    # experiment 3
    parser.add_argument(
        "--sidelobes_snr",
        nargs="+",
        type=int,
        default=[0, 20],
        help="SNR levels for sidelobe comparison",
    )
    parser.add_argument(
        "--sidelobes_max_n_freq",
        type=int,
        default=2,
        help="for each signal the number of frequencies is uniformly drawn between 1 and max_n_freq",
    )
    parser.add_argument(
        "--exp3_num_samples",
        type=int,
        default=2,
        help="number of samples used in experiment 3 per SNR value",
    )
    parser.add_argument(
        "--sidelobe_sep_list",
        nargs="+",
        type=int,
        default=[0.6, 1.5],
        help="resolution spacing for sidelobes (x/N)",
    )

    # experiment 4
    parser.add_argument(
        "--exp4_data_path",
        type=str,
        default="paper_results/paper_data/spinningtarget.mat",
        help="path for file to read for experiment 4",
    )

    # experiment 5
    parser.add_argument(
        "--exp5_data_path",
        type=str,
        default="paper_results/paper_data/plane.mat",
        help="path for file to read for experiment 5",
    )

    # cuda parameters
    parser.add_argument(
        "--no_cuda", action="store_true", help="avoid using CUDA when available"
    )

    # dataset parameters
    parser.add_argument(
        "--batch_size", type=int, default=256, help="batch size used during testing"
    )
    parser.add_argument(
        "--signal_dim", type=int, default=64, help="dimensionof the input signal"
    )
    parser.add_argument(
        "--fr_size", type=int, default=4096, help="size of the frequency representation"
    )
    parser.add_argument(
        "--max_n_freq",
        type=int,
        default=10,
        help="for each signal the number of frequencies is uniformly drawn between 1 and max_n_freq",
    )
    parser.add_argument(
        "--min_sep",
        type=float,
        default=0.5,
        help="minimum separation between spikes, normalized by signal_dim",
    )
    parser.add_argument(
        "--distance",
        type=str,
        default="normal",
        help="distance distribution between spikes",
    )
    parser.add_argument(
        "--amplitude",
        type=str,
        default="uniform",
        help="amplitude distribution",
    )
    parser.add_argument(
        "--floor_amplitude", type=float, default=0.1, help="minimum amplitude of spikes"
    )
    parser.add_argument(
        "--noise",
        type=str,
        default="gaussian_blind_contsant",
        help="kind of noise to use",
    )

    # kernel parameters used to generate the ideal frequency representation
    parser.add_argument(
        "--kernel_type",
        type=str,
        default="gaussian",
        help="type of kernel used to create the ideal frequency representation [gaussian, triangle or closest]",
    )
    parser.add_argument(
        "--triangle_slope",
        type=float,
        default=4000,
        help="slope of the triangle kernel normalized by signal_dim",
    )
    parser.add_argument(
        "--gaussian_std",
        type=float,
        default=0.12,
        help="std of the gaussian kernel normalized by signal_dim",
    )

    # seeding parameters
    parser.add_argument("--numpy_seed", type=int, default=76)
    parser.add_argument("--torch_seed", type=int, default=76)

    args = parser.parse_args()

    # Determine if cuda should be used
    args.device = util.set_device(args)

    # Seed numpy and pytorch
    np.random.seed(args.numpy_seed)
    torch.manual_seed(args.torch_seed)

    return args


if __name__ == "__main__":
    main()
