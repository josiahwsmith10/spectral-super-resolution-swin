"""
Experiment 2: resolution capability 
"""

import torch
import numpy as np
from scipy.io import savemat
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

import util
from data import gen_signal_res
from data.noise import noise_torch


def create_data_tuple(const_sep, args):
    clean_signals, f = gen_signal_res(
        args.exp2_num_samples,
        args.signal_dim,
        const_sep,
        args.amplitude,
        args.floor_amplitude,
    )

    clean_signals = torch.from_numpy(clean_signals).float()
    f = torch.from_numpy(f).float()

    noisy_signals = noise_torch(clean_signals, args.res_snr, "gaussian_blind_constant")

    return noisy_signals, f


def create_data(args):
    data = {}
    print("Creating data...")
    for const_sep in args.res_list:
        data[const_sep] = create_data_tuple(const_sep, args)
    return data


def freq_to_idx(f_i, args):
    xgrid = np.linspace(-0.5, 0.5, args.fr_size, endpoint=False)

    idx = []
    idx_exact = []

    for f_n in f_i:
        i = int(np.argmin(np.abs(f_n - xgrid)))
        i_min = max([0, i - args.res_tol])
        i_max = min([i + args.res_tol, args.fr_size])
        
        idx.append((i_min, i_max))
        idx_exact.append(i)

    return idx, idx_exact


def compute_resolution(model, data_tuple, args):
    x, f = data_tuple

    y_pred = util.test_basic_SR(model, x, args=args)

    results = []
    for y_pred_i, f_i in zip(y_pred, f):
        idx_i, idx_exact_i = freq_to_idx(f_i, args)

        amp_i = torch.tensor(
            [
                y_pred_i[idx_i[0][0] : idx_i[0][1]].max(),
                y_pred_i[idx_i[1][0] : idx_i[1][1]].max(),
            ]
        )

        amp_mid = y_pred_i[(idx_exact_i[0] + idx_exact_i[1]) // 2]

        results.append(amp_mid < (amp_i.min() / np.sqrt(2)))

    results = torch.tensor(results)

    prob_res = results.sum() / results.numel()

    return prob_res


def get_results(args, data):
    results = {}
    for method, model in args.models.items():
        results[method] = {}

        print(f"Computing probability of resolution for method={method}")
        for sep in tqdm(args.res_list):
            if method.lower() == "music" or method.lower() == "omp":
                model.force_nfreqs = 2
            results[method][sep] = compute_resolution(model, data[sep], args)
    return results


def print_results(args, results):
    seps = np.array(args.res_list).reshape(-1, 1)
    results_mat = np.zeros((seps.size, len(args.methods)))

    save_dict = {"seperation": seps.flatten()}

    for i, method in enumerate(args.methods):
        save_dict[method] = np.zeros(seps.shape[0])
        for j, sep in enumerate(args.res_list):
            results_mat[j, i] = results[method][sep].detach().cpu().numpy()
            save_dict[method][j] = results_mat[j, i]

    table = np.hstack((seps, results_mat))
    headers = ["Separation"] + args.methods

    print(tabulate(table, headers=headers))

    df = pd.DataFrame(table, columns=headers)
    df.to_csv(f"./paper_results/exp2.csv")

    savemat("./paper_results/exp2.mat", save_dict)


def experiment2(args):
    print("Starting experiment 2...")

    # Create data for all methods
    data = create_data(args)

    # Get results
    results = get_results(args, data)

    # Print results
    print_results(args, results)
