import numpy as np
from tqdm import tqdm


def frequency_generator(f, nf, min_sep, dist_distribution):
    """
    Courtesy of: https://github.com/sreyas-mohan/DeepFreq
    """
    if dist_distribution == "random":
        random_freq(f, nf, min_sep)
    elif dist_distribution == "jittered":
        jittered_freq(f, nf, min_sep)
    elif dist_distribution == "normal":
        normal_freq(f, nf, min_sep)


def random_freq(f, nf, min_sep):
    """
    Courtesy of: https://github.com/sreyas-mohan/DeepFreq
    Generate frequencies uniformly.
    """
    for i in range(nf):
        f_new = np.random.rand() - 1 / 2
        condition = True
        while condition:
            f_new = np.random.rand() - 1 / 2
            condition = (
                (np.min(np.abs(f - f_new)) < min_sep)
                or (np.min(np.abs((f - 1) - f_new)) < min_sep)
                or (np.min(np.abs((f + 1) - f_new)) < min_sep)
            )
        f[i] = f_new


def jittered_freq(f, nf, min_sep, jit=1):
    """
    Courtesy of: https://github.com/sreyas-mohan/DeepFreq
    Generate jittered frequencies.
    """
    l, r = -0.5, 0.5 - nf * min_sep * (1 + jit)
    s = l + np.random.rand() * (r - l)
    c = np.cumsum(min_sep * (np.ones(nf) + np.random.rand(nf) * jit))
    f[:nf] = (s + c - min_sep + 0.5) % 1 - 0.5


def normal_freq(f, nf, min_sep, scale=0.05):
    """
    Courtesy of: https://github.com/sreyas-mohan/DeepFreq
    Distance between two frequencies follows a normal distribution
    """
    f[0] = np.random.uniform() - 0.5
    for i in range(1, nf):
        condition = True
        while condition:
            d = np.random.normal(scale=scale)
            f_new = (d + np.sign(d) * min_sep + f[i - 1] + 0.5) % 1 - 0.5
            condition = (
                (np.min(np.abs(f - f_new)) < min_sep)
                or (np.min(np.abs((f - 1) - f_new)) < min_sep)
                or (np.min(np.abs((f + 1) - f_new)) < min_sep)
            )
        f[i] = f_new


def amplitude_generation(dim, amplitude, floor_amplitude=0.1):
    """
    Modified from: https://github.com/sreyas-mohan/DeepFreq
    Generate the amplitude associated with each frequency.
    """
    if amplitude == "uniform":
        return np.random.rand(*dim) * (1 - floor_amplitude) + floor_amplitude
    elif amplitude == "normal":
        return np.abs(np.random.randn(*dim))
    elif amplitude == "normal_floor":
        return np.abs(np.random.randn(*dim)) + floor_amplitude
    elif amplitude == "alternating":
        return np.random.rand(*dim) * 0.5 + 20 * np.random.rand(
            *dim
        ) * np.random.randint(0, 2, size=dim)


def gen_signal(
    num_samples,
    signal_dim,
    num_freq,
    min_sep,
    distance="normal",
    amplitude="normal_floor",
    floor_amplitude=0.1,
    variable_num_freq=False,
):
    """
    Courtesy of: https://github.com/sreyas-mohan/DeepFreq
    Generate the amplitude associated with each frequency.
    """
    s = np.zeros((num_samples, 2, signal_dim))
    xgrid = np.arange(signal_dim)[:, None]
    f = np.ones((num_samples, num_freq)) * np.inf
    r = amplitude_generation((num_samples, num_freq), amplitude, floor_amplitude)
    theta = np.random.rand(num_samples, signal_dim) * 2 * np.pi
    d_sep = min_sep / signal_dim
    if variable_num_freq:
        nfreq = np.random.randint(1, num_freq + 1, num_samples)
    else:
        nfreq = np.ones(num_samples, dtype="int") * num_freq
    for n in tqdm(range(num_samples)):
        frequency_generator(f[n], nfreq[n], d_sep, distance)
        for i in range(nfreq[n]):
            sin = r[n, i] * np.exp(1j * theta[n, i] + 2j * np.pi * f[n, i] * xgrid.T)
            s[n, 0] = s[n, 0] + sin.real
            s[n, 1] = s[n, 1] + sin.imag
        s[n] = s[n] / np.sqrt(np.mean(np.power(s[n], 2)))
    f.sort(axis=1)
    f[f == float("inf")] = -10
    return s.astype("float32"), f.astype("float32"), nfreq


def res_freq(f, d_sep):
    max_freq = 0.5 - d_sep

    f[0] = -0.5 + (0.5 + max_freq) * np.random.rand()
    f[1] = f[0] + d_sep


def gen_signal_res(
    num_samples,
    signal_dim,
    const_sep,
    amplitude="normal_floor",
    floor_amplitude=0.1,
):
    """
    Modified from: https://github.com/sreyas-mohan/DeepFreq
    Generates randomly placed frequencies with constant separation.
    """

    # Always use 2 frequencies
    num_freq = 2

    s = np.zeros((num_samples, 2, signal_dim))
    xgrid = np.arange(signal_dim)[:, None]
    f = np.ones((num_samples, num_freq)) * np.inf
    r = amplitude_generation((num_samples, num_freq), amplitude, floor_amplitude)
    theta = np.random.rand(num_samples, signal_dim) * 2 * np.pi
    d_sep = const_sep / signal_dim
    for n in tqdm(range(num_samples)):
        res_freq(f[n], d_sep)
        for i in range(num_freq):
            sin = r[n, i] * np.exp(1j * theta[n, i] + 2j * np.pi * f[n, i] * xgrid.T)
            s[n, 0] = s[n, 0] + sin.real
            s[n, 1] = s[n, 1] + sin.imag
        s[n] = s[n] / np.sqrt(np.mean(np.power(s[n], 2)))
    f.sort(axis=1)
    f[f == float("inf")] = -10
    return s.astype("float32"), f.astype("float32")
