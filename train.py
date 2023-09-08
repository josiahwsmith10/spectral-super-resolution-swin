import os
import sys
import argparse
import logging
from datetime import datetime

import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from data import dataset
import modules
import util

logger = logging.getLogger(__name__)


def main(experiment=None, checkpoint_path=None):
    """
    Modified from: https://github.com/sreyas-mohan/DeepFreq
    """
    args, tb_writer = setup(experiment)

    np.random.seed(args.numpy_seed)
    torch.manual_seed(args.torch_seed)

    # Create the training and validation datasets
    train_loader = dataset.make_train_data(args)
    val_loader = dataset.make_eval_data_v2(args)

    # Initialize the network
    if checkpoint_path is None:
        fr_module = modules.select_model(args)
        fr_optimizer, fr_scheduler = util.set_optim(args, fr_module)
        start_epoch = 1
    else:
        fr_module, fr_optimizer, fr_scheduler, args, _ = util.load(
            checkpoint_path=checkpoint_path
        )

    logger.info(
        "[Network] Number of parameters in the frequency-representation module : %.3f M"
        % (util.model_parameters(fr_module) / 1e6)
    )

    # Choose loss function
    if args.loss_fn == "l2":
        fr_criterion = torch.nn.MSELoss(reduction="sum")
    elif args.loss_fn == "l1":
        fr_criterion = torch.nn.L1Loss(reduction="sum")
    elif args.loss_fn == "l1_smooth":
        fr_criterion = torch.nn.SmoothL1Loss(reduction="sum")

    for epoch in range(start_epoch, args.n_epochs_fr + 1):
        util.train_freq_SR(
            args=args,
            fr_module=fr_module,
            fr_optimizer=fr_optimizer,
            fr_criterion=fr_criterion,
            fr_scheduler=fr_scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            epoch=epoch,
            tb_writer=tb_writer,
        )

        if epoch % args.save_epoch_freq == 0 or epoch == args.n_epochs_fr:
            util.save(
                fr_module, fr_optimizer, fr_scheduler, args, epoch, args.fr_module_type
            )


def setup(experiment=None):
    """
    Initializes logging, tensorboard, and argument parsing. Returns the parsed arguments and the tensorboard writer.
    """
    parser = argparse.ArgumentParser()

    # basic parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="output directory",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="avoid using CUDA when available"
    )

    # dataset parameters
    parser.add_argument(
        "--batch_size", type=int, default=256, help="batch size used during training"
    )
    parser.add_argument(
        "--signal_dim", type=int, default=50, help="dimensionof the input signal"
    )
    parser.add_argument(
        "--fr_size", type=int, default=1000, help="size of the frequency representation"
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
        default=1.0,
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
    parser.add_argument(
        "--snr", type=float, default=-10, help="snr parameter used by cResFreq"
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
        default=0.3,
        help="std of the gaussian kernel normalized by signal_dim",
    )

    # frequency-representation (fr) module parameters
    parser.add_argument(
        "--fr_module_type",
        type=str,
        default="fr",
        help="type of the fr module: [deepfreq | psnet | cresfreq | custom]",
    )
    parser.add_argument(
        "--fr_n_layers",
        type=int,
        default=20,
        help="number of convolutional layers in the fr module",
    )
    parser.add_argument(
        "--fr_n_filters",
        type=int,
        default=64,
        help="number of filters per layer in the fr module",
    )
    parser.add_argument(
        "--fr_kernel_size",
        type=int,
        default=3,
        help="filter size in the convolutional blocks of the fr module",
    )
    parser.add_argument(
        "--fr_kernel_out",
        type=int,
        default=25,
        help="size of the conv transpose kernel",
    )
    parser.add_argument(
        "--fr_out_padding",
        type=int,
        default=0,
        help="output padding of the conv transpose layer",
    )
    parser.add_argument(
        "--fr_reduction_factor",
        type=int,
        default=4,
        help="reduction factor between conv layers to reduce channels",
    )
    parser.add_argument(
        "--fr_depths",
        nargs="+",
        type=int,
        default=[8, 8, 8],
        help="number of STL blocks per RSTB for Swin-based model",
    )
    parser.add_argument(
        "--fr_num_heads",
        nargs="+",
        type=int,
        default=[8, 8, 8],
        help="number heads for Swin-based model self-attention",
    )
    parser.add_argument(
        "--fr_window_size",
        type=int,
        default=16,
        help="window size for swin transformer",
    )
    parser.add_argument(
        "--fr_mlp_ratio",
        type=float,
        default=2.0,
        help="ratio for channel upsampling in MLP block of swin transformer",
    )
    parser.add_argument(
        "--fr_dropout",
        type=float,
        default=0.0,
        help="dropout in swin transformer",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="min-max",
        help="normalization method for input layer of swin transformer",
    )
    parser.add_argument(
        "--fr_optional_relu",
        type=int,
        default=1,
        help="boolean whether or not to use optional relu in RSTB block for real-valued swinfreq",
    )
    parser.add_argument(
        "--fr_inner_dim",
        type=int,
        default=125,
        help="dimension after first linear transformation",
    )
    parser.add_argument(
        "--fr_upsampling",
        type=int,
        default=8,
        help="stride of the transposed convolution, upsampling * inner_dim = fr_size",
    )

    # training parameters
    parser.add_argument(
        "--n_training", type=int, default=50000, help="# of training data"
    )
    parser.add_argument(
        "--n_validation", type=int, default=1000, help="# of validation data"
    )
    parser.add_argument(
        "--lr_fr",
        type=float,
        default=0.003,
        help="initial learning rate for adam optimizer used for the frequency-representation module",
    )
    parser.add_argument(
        "--loss_fn",
        type=str,
        default="l2",
        help="loss function used to train the fr module [l2 | l1 | l1_smooth]",
    )
    parser.add_argument(
        "--optim_type",
        type=str,
        default="adam",
        help="optimizer type [adam | adamw | rmsprop]",
    )
    parser.add_argument(
        "--n_epochs_fr",
        type=int,
        default=200,
        help="number of epochs used to train the fr module",
    )
    parser.add_argument(
        "--save_epoch_freq",
        type=int,
        default=10,
        help="frequency of saving checkpoints at the end of epochs",
    )

    # seeding parameters
    parser.add_argument("--numpy_seed", type=int, default=100)  # original=100
    parser.add_argument("--torch_seed", type=int, default=76)  # original=76

    # misc notes
    parser.add_argument(
        "--zzz_note",
        type=str,
        default="",
        help="Note about model/experiment",
    )

    args = parser.parse_args()

    if experiment is not None:
        # Model Type
        args.fr_module_type = experiment["fr_module_type"]

        # Signal Characteristics
        args.signal_dim = experiment["signal_dim"]
        args.fr_size = experiment["fr_size"]
        args.min_sep = experiment["min_sep"]
        args.amplitude = experiment["amplitude"]
        args.noise = experiment["noise"]
        args.snr = experiment["snr"]
        args.min_snr_db = experiment["min_snr_db"]
        args.max_snr_db = experiment["max_snr_db"]
        args.step_snr_db = experiment["step_snr_db"]
        args.gaussian_std = experiment["gaussian_std"]

        # cResFreq
        args.fr_n_layers = experiment["fr_n_layers"]
        args.fr_n_filters = experiment["fr_n_filters"]
        args.fr_inner_dim = experiment["fr_inner_dim"]
        args.fr_upsampling = experiment["fr_upsampling"]
        args.fr_kernel_out = experiment["fr_kernel_out"]
        args.fr_out_padding = experiment["fr_out_padding"]

        # Channel Attention
        args.fr_reduction_factor = experiment["fr_reduction_factor"]

        # Swin
        args.fr_depths = experiment["fr_depths"]
        args.fr_num_heads = experiment["fr_num_heads"]
        args.fr_window_size = experiment["fr_window_size"]
        args.fr_mlp_ratio = experiment["fr_mlp_ratio"]
        args.fr_dropout = experiment["fr_dropout"]
        args.normalization = experiment["normalization"]
        args.fr_optional_relu = experiment["fr_optional_relu"]

        # Training
        args.lr_fr = experiment["lr_fr"]
        args.batch_size = experiment["batch_size"]
        args.n_epochs_fr = experiment["n_epochs_fr"]
        args.optim_type = experiment["optim_type"]

        # Data
        args.n_training = experiment["n_training"]
        args.n_validation = experiment["n_validation"]

        # Note
        args.zzz_note = experiment["zzz_note"]

    if args.zzz_note:
        model_name = args.zzz_note.replace(" ", "_")
    else:
        model_name = args.fr_module_type

    if args.output_dir is None:
        args.output_dir = (
            f"./checkpoint/{model_name}_{datetime.now().strftime('%m-%d-%H%M')}"
        )

    if torch.cuda.is_available() and not args.no_cuda:
        args.use_cuda = True
    else:
        args.use_cuda = False

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    file_handler = logging.FileHandler(
        filename=os.path.join(args.output_dir, "run.log")
    )
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )

    tb_writer = SummaryWriter(args.output_dir)
    util.print_args(logger, args)

    return args, tb_writer


if __name__ == "__main__":
    main()
