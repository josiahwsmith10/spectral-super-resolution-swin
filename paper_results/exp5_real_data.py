"""
Experiment 5: real plane data
"""

import torch
from scipy.io import loadmat, savemat

import util


def load_data(args):
    data = loadmat(args.exp5_data_path)["x"]
    data = torch.from_numpy(data).type(dtype=torch.cfloat)
    return data


def get_results(args, data):
    results = {}
    for method, model in args.models.items():
        print(f"Computing SR line spectra for method={method}")
        results[method] = util.test_basic_SR(model, data, args)
    return results


def save_results(results):
    savemat("./paper_results/exp5.mat", results)


def experiment5(args):
    print("Starting experiment 5...")

    # Load data
    data = load_data(args)

    # Get results
    results = get_results(args, data)

    # Save results
    save_results(results)
