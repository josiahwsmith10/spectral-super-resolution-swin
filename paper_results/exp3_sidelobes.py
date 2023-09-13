"""
Experiment 3: sample of sidelobes
"""

import torch
import numpy as np
from scipy.io import savemat
from tqdm import tqdm

from data import gen_signal_res
from data.noise import noise_torch
import util


def create_sidelobe_data(snr, args):
    clean_signals0, f0 = gen_signal_res(
        args.exp3_num_samples//2,
        args.signal_dim,
        args.sidelobe_sep_list[0],
        args.amplitude,
        args.floor_amplitude,
    )
    
    clean_signals1, f1 = gen_signal_res(
        args.exp3_num_samples//2,
        args.signal_dim,
        args.sidelobe_sep_list[1],
        args.amplitude,
        args.floor_amplitude,
    )
    
    f = np.concatenate((f0, f1), axis=0)
    clean_signals = np.concatenate((clean_signals0, clean_signals1), axis=0)

    clean_signals = torch.from_numpy(clean_signals).float()

    noisy_signals = noise_torch(clean_signals, snr, "gaussian_blind_constant")

    return noisy_signals, f


def create_data(args):
    data = {}
    f = {}
    print("Creating data...")
    for snr in args.sidelobes_snr:
        data[snr], f[snr] = create_sidelobe_data(snr, args)
    return data, f


def get_results(args, data):
    results = {}
    for method, model in args.models.items():
        results[method] = {}

        print(f"Computing SR line spectra for method={method}")
        for snr in tqdm(args.sidelobes_snr):
            if method.lower() == "music" or method.lower() == "omp":
                model.force_nfreqs = 2
            results[method][snr] = util.test_basic_SR(model, data[snr], args)
    return results


def save_results(args, results, f):
    snr = np.array(args.sidelobes_snr).reshape(-1, 1)
    save_dict = {"snr": snr}

    for method in args.methods:
        save_dict[method] = np.zeros(
            (args.fr_size, args.exp3_num_samples, len(args.sidelobes_snr))
        )
        for i, snr_i in enumerate(args.sidelobes_snr):
            save_dict[method][:, :, i] = results[method][snr_i].detach().cpu().numpy().T

    save_dict["f"] = np.zeros(
        (args.exp3_num_samples, args.sidelobes_max_n_freq, len(args.sidelobes_snr))
    )
    for i, snr_i in enumerate(args.sidelobes_snr):
        save_dict["f"][:, :, i] = f[snr_i]

    savemat("./paper_results/exp3.mat", save_dict)


def experiment3(args):
    print("Starting experiment 3...")

    # Create data for all methods
    data, f = create_data(args)

    # Get results
    results = get_results(args, data)

    # Save results
    save_results(args, results, f)
