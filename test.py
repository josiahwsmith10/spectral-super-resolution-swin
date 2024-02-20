import argparse

import torch
import numpy as np

from data import dataset
import util


def main():
    args = setup()

    # Temporarily store
    n_testing = args.n_testing
    min_snr_db = args.min_snr_db
    max_snr_db = args.max_snr_db
    step_snr_db = args.step_snr_db
    batch_size = args.batch_size
    n_testing = args.n_testing

    np.random.seed(args.numpy_seed)
    torch.manual_seed(args.torch_seed)

    # Load the model from the checkpoint
    fr_module, _, _, args, _ = util.load(
        checkpoint_path=args.checkpoint_path,
        # device=torch.device("cpu") # Use if using CPU-based torch (or on Mac with Apple Silicon)
    )

    # Set parameters
    args.n_testing = n_testing
    args.min_snr_db = min_snr_db
    args.max_snr_db = max_snr_db
    args.step_snr_db = step_snr_db
    args.batch_size = batch_size
    args.n_testing = n_testing

    # Print args into console
    message = ""
    for k, v in sorted(vars(args).items()):
        message += "\n{:>30}: {:<30}".format(str(k), str(v))
    print(message)

    # Create the testing dataset
    test_loader = dataset.make_test_data(args)

    # Choose loss function
    if args.loss_fn == "l2":
        fr_criterion = torch.nn.MSELoss(reduction="sum")
    elif args.loss_fn == "l1":
        fr_criterion = torch.nn.L1Loss(reduction="sum")
    elif args.loss_fn == "l1_smooth":
        fr_criterion = torch.nn.SmoothL1Loss(reduction="sum")

    loss_test_fr = util.test_freq_SR(
        args=args,
        fr_module=fr_module,
        fr_criterion=fr_criterion,
        test_loader=test_loader,
    )

    print("Testing loss: ", loss_test_fr)


def setup():
    """
    Initializes argument parsing. Returns the parsed arguments.
    """
    parser = argparse.ArgumentParser()

    # checkpoint
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoint/skipfreq_snr_big8/fr/epoch_60.pth",
        help="path to checkpoint to load",
    )

    # basic parameters
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
        default="normal_floor",
        help="spike amplitude distribution",
    )
    parser.add_argument(
        "--floor_amplitude", type=float, default=0.1, help="minimum amplitude of spikes"
    )
    parser.add_argument(
        "--noise", type=str, default="gaussian_blind", help="kind of noise to use"
    )
    parser.add_argument("--min_snr_db", type=int, default=-10, help="minimum SNR in dB")
    parser.add_argument("--max_snr_db", type=int, default=40, help="maximum SNR in dB")
    parser.add_argument(
        "--step_snr_db", type=int, default=10, help="step size of SNR in dB"
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

    # testing parameters
    parser.add_argument(
        "--n_testing", type=int, default=5000, help="# of training data"
    )
    parser.add_argument(
        "--n_validation", type=int, default=1000, help="# of validation data"
    )

    # seed RNGs
    parser.add_argument("--numpy_seed", type=int, default=100)  # original=100
    parser.add_argument("--torch_seed", type=int, default=76)  # original=76

    args = parser.parse_args()

    # Determine if cuda should be used
    args.device = util.set_device(args)


if __name__ == "__main__":
    main()
