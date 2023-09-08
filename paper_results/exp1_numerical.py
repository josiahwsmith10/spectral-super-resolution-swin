"""
Experiment 1: RMSE / SSIM across SNR
"""

import torch
import numpy as np
from scipy.io import savemat
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

import util
from util.metrics import PSNR1d, SSIM1d, RMSE1d
from data.dataset import load_dataset_fixed_noise
from .util import is_ml_model


def create_data_tuple(snr, args):
    xgrid = np.linspace(-0.5, 0.5, args.fr_size, endpoint=False)
    if args.kernel_type == "triangle":
        kernel_param = args.triangle_slope / args.signal_dim
    else:
        kernel_param = args.gaussian_std / args.signal_dim
    noisy_signals, _, fr, _ = load_dataset_fixed_noise(
        num_samples=args.exp1_num_samples,
        signal_dim=args.signal_dim,
        max_n_freq=args.max_n_freq,
        min_sep=args.min_sep,
        distance=args.distance,
        amplitude=args.amplitude,
        floor_amplitude=args.floor_amplitude,
        kernel_type=args.kernel_type,
        kernel_param=kernel_param,
        xgrid=xgrid,
        snr=snr,
        noise="gaussian_blind_constant",
    )[:]

    return noisy_signals, fr


def compute_metrics(model, data_tuple, args):
    x, y = data_tuple
    if is_ml_model(model):
        y_pred = util.test_basic_SR(model, x, args=args)

    results = {}
    for metric in args.exp1_metrics_list:
        if metric.lower() == "psnr":
            m = PSNR1d()
        elif metric.lower() == "ssim":
            m = SSIM1d()
        elif metric.lower() == "rmse":
            m = RMSE1d()
        if is_ml_model(model):
            results[metric] = (
                torch.tensor([m(b_pred, b) for b_pred, b in zip(y_pred, y)])
                .mean()
                .round(decimals=3)
            )
        else:
            results[metric] = torch.tensor(torch.nan)
    return results


def create_data(args):
    data = {}
    print("Creating data...")
    for snr in args.snr_list:
        data[snr] = create_data_tuple(snr, args)
    return data


def get_results(args, data):
    results = {}
    for method, model in args.models.items():
        results[method] = {}

        print(f"Computing metrics for method={method}")
        for snr in tqdm(args.snr_list):
            results[method][snr] = compute_metrics(model, data[snr], args)
    return results


def print_results(args, results):
    save_dict = {"snr": np.array(args.snr_list)}
    for metric in args.exp1_metrics_list:
        print(f"Metric: {metric}")
        snrs = np.array(args.snr_list).reshape(-1, 1)
        results_mat = np.zeros((snrs.size, len(args.methods)))

        for i, method in enumerate(args.methods):
            save_dict[f"{metric}_{method}"] = np.zeros(snrs.shape[0])
            for j, snr in enumerate(args.snr_list):
                results_mat[j, i] = results[method][snr][metric].detach().cpu().numpy()
                save_dict[f"{metric}_{method}"][j] = results_mat[j, i]

        table = np.hstack((snrs, results_mat))
        headers = ["SNR"] + args.methods

        print(tabulate(table, headers=headers))

        df = pd.DataFrame(table, columns=headers)
        df.to_csv(f"./paper_results/exp1_{metric}.csv")

    savemat("./paper_results/exp1.mat", save_dict)


def experiment1(args):
    print("Starting experiment 1...")

    # Create data for all methods
    data = create_data(args)

    # Get results
    results = get_results(args, data)

    # Print results
    print_results(args, results)
