import numpy as np
import torch


def noise_torch(t, snr=-10.0, kind="gaussian_blind_constant", n_corr=None, args=None):
    """
    Modified from : https://github.com/sreyas-mohan/DeepFreq
    """

    if args is not None:
        snr = args.snr
        kind = args.noise

    kind = kind.lower()
    if kind == "gaussian":
        return gaussian_noise(t, snr)
    elif kind == "gaussian_blind_deepfreq":
        return gaussian_blind_noise_deepfreq(t, snr)
    elif kind == "gaussian_blind_cresfreq":
        return gaussian_blind_noise_cresfreq(t, snr)
    elif kind == "gaussian_blind_constant":
        return gaussian_blind_noise_constant(t, snr)
    elif kind == "gaussian_blind_batch":
        return gaussian_blind_noise_batch(t, [args.min_snr_db, args.max_snr_db])
    elif kind == "gaussian_blind_strict":
        return gaussian_blind_noise_batch_strict(t, [args.min_snr_db, args.max_snr_db])
    elif kind == "sparse":
        return sparse_noise(t, n_corr)
    elif kind == "variable_sparse":
        return variable_sparse_noise(t, n_corr)
    else:
        raise NotImplementedError(f"{kind} noise type not implemented!")


def gaussian_noise(s, snr):
    """
    Courtesy of: https://github.com/sreyas-mohan/DeepFreq
    Add Gaussian noise to the input signal.
    """
    bsz, _, signal_dim = s.size()
    s = s.view(s.size(0), -1)
    sigma = np.sqrt(1.0 / snr)
    noise = torch.randn(s.size(), device=s.device, dtype=s.dtype)
    mult = sigma * torch.norm(s, 2, dim=1) / (torch.norm(noise, 2, dim=1))
    noise = noise * mult[:, None]
    return (s + noise).view(bsz, -1, signal_dim)


def gaussian_blind_noise_deepfreq(s, snr):
    """
    Courtesy of: https://github.com/sreyas-mohan/DeepFreq
    Add Gaussian noise to the input signal. The std of the gaussian noise is uniformly chosen between 0 and 1/sqrt(snr).
    """
    bsz, _, signal_dim = s.size()
    s = s.view(bsz, -1)
    sigma_max = np.sqrt(1.0 / snr)
    sigmas = sigma_max * torch.rand(bsz, device=s.device, dtype=s.dtype)
    noise = torch.randn(s.size(), device=s.device, dtype=s.dtype)
    mult = sigmas * torch.norm(s, 2, dim=1) / (torch.norm(noise, 2, dim=1))
    noise = noise * mult[:, None]
    return (s + noise).view(bsz, -1, signal_dim)


def gaussian_blind_noise_cresfreq(s, snr):
    """
    Courtesy of: https://github.com/panpp-git/cResFreq
        - Different implementation from DeepFreq
        - Assumes signal power is always 1
    """
    bsz, _, signal_dim = s.size()
    s = s.view(bsz, -1)
    low = snr
    high = 40
    # sigma_max = np.sqrt(1. / snr)
    # sigmas = sigma_max * torch.rand(bsz, device=s.device, dtype=s.dtype)
    scpu = s.cpu().numpy()
    snr_array = (low + (high - low) * torch.rand(bsz))[:, None]
    snr_array = snr_array.cpu().numpy()
    noise = torch.randn(s.size(), device=s.device, dtype=s.dtype)
    s = torch.from_numpy(scpu * (10 ** (snr_array / 20))).to(s.device)
    # mult = sigmas * torch.norm(s, 2, dim=1) / (torch.norm(noise, 2, dim=1))
    # noise = noise * mult[:, None]
    return (s + noise).view(bsz, -1, signal_dim)


def gaussian_blind_noise_constant(s, snr):
    s_iq = s[:, 0] + 1j * s[:, 1]
    s_power = s_iq.abs().pow(2).mean(dim=1, keepdim=True)
    snr_linear = 10 ** (snr / 10)
    sigma = (s_power / (snr_linear * 2)).sqrt()
    noise = sigma * torch.randn(s_iq.shape, device=s.device, dtype=torch.cfloat)
    s[:, 0] += noise.real
    s[:, 1] += noise.imag
    return s


def gaussian_blind_noise_batch(s, snr_min_max=[-10, 40]):
    bsz = s.shape[0]
    s_iq = s[:, 0] + 1j * s[:, 1]
    s_power = s_iq.abs().pow(2).mean(dim=1, keepdim=True)
    snr_db_array = (
        snr_min_max[0]
        + (snr_min_max[1] - snr_min_max[0])
        * torch.rand(bsz, device=s.device, dtype=s.dtype)
    )[:, None]
    snr_array = 10 ** (snr_db_array / 10)
    sigma = (s_power / (snr_array * 2)).sqrt()
    noise = sigma * torch.randn(s_iq.shape, device=s.device, dtype=torch.cfloat)
    s[:, 0] += noise.real
    s[:, 1] += noise.imag
    return s


def gaussian_blind_noise_batch_strict(s, snr_min_max=[-10, 40]):
    bsz, _, signal_dim = s.size()
    s = s.view(bsz, -1)
    noise = torch.randn(s.size(), device=s.device, dtype=s.dtype)
    mult = torch.norm(s, 2, dim=1) / torch.norm(noise, 2, dim=1)
    noise *= mult[:, None]
    snr_array = (
        snr_min_max[0]
        + (snr_min_max[1] - snr_min_max[0])
        * torch.rand(bsz, device=s.device, dtype=s.dtype)
    )[:, None]
    s = (s * (10 ** (snr_array / 20))).to(s.device)
    return (s + noise).view(bsz, -1, signal_dim)


def sparse_noise(s, n_corr):
    """
    Courtesy of: https://github.com/sreyas-mohan/DeepFreq
    Add sparse noise to the input signal. The number of corrupted elements is equal to n_corr.
    """
    noisy_signal = s.clone()
    corruption = 0.5 * torch.randn(
        (s.size(0), s.size(1), n_corr), device=s.device, dtype=s.dtype
    )
    for i in range(s.size(0)):
        idx = torch.multinomial(torch.ones(s.size(-1)), n_corr, replacement=False)
        noisy_signal[i, :, idx] += corruption[i, :]
    return noisy_signal


def variable_sparse_noise(s, max_corr):
    """
    Courtesy of: https://github.com/sreyas-mohan/DeepFreq
    Add sparse noise to the input signal. The number of corrupted elements is drawn uniformaly between 1 and
    max_corruption.
    """
    noisy_signal = s.clone()
    corruption = 0.5 * torch.randn(
        (s.size(0), s.size(1), max_corr), device=s.device, dtype=s.dtype
    )
    n_corr = np.random.randint(1, max_corr + 1, (s.size(0)))
    for i in range(s.size(0)):
        idx = torch.multinomial(
            torch.ones(s.size(-1)), int(n_corr[i]), replacement=False
        )
        noisy_signal[i, :, idx] += corruption[i, :, : n_corr[i]]
    return noisy_signal
