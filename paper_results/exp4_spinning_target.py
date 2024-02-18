"""
Experiment 4: spinning target data from cResFreq data
"""

import torch
from scipy.io import loadmat, savemat

import util


def load_data(args):
    data = loadmat(args.exp4_data_path)["x"]
    data = torch.from_numpy(data).type(dtype=torch.cfloat)
    return data


def get_results(args, data):
    results = {}
    for method, model in args.models.items():
        print(f"Computing SR line spectra for method={method}")
        results[method] = torch.fft.fftshift(
            util.test_basic_SR(model, data, args), dim=1
        )
    return results


def save_results(results):
    savemat("./paper_results/exp4.mat", results)


def experiment4(args):
    print("Starting experiment 4...")

    # Load data
    data = load_data(args)

    # Get results
    results = get_results(args, data)

    # Save results
    save_results(results)
