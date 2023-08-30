import torch
import numpy as np

from tqdm import tqdm


def freq2fr(f, xgrid, kernel_type="gaussian", param=None):
    """
    Courtesy of: https://github.com/sreyas-mohan/DeepFreq
    Convert an array of frequencies to a frequency representation discretized on xgrid.
    """
    if kernel_type == "gaussian":
        return gaussian_kernel(f, xgrid, param)
    elif kernel_type == "triangle":
        return triangle(f, xgrid, param)


def gaussian_kernel(f, xgrid, sigma):
    """
    Courtesy of: https://github.com/sreyas-mohan/DeepFreq
    Create a frequency representation with a Gaussian kernel.
    """
    fr = np.zeros((f.shape[0], xgrid.shape[0]))
    for i in tqdm(range(f.shape[1])):
        dist = np.abs(xgrid[None, :] - f[:, i][:, None])
        rdist = np.abs(xgrid[None, :] - (f[:, i][:, None] + 1))
        ldist = np.abs(xgrid[None, :] - (f[:, i][:, None] - 1))
        dist = np.minimum(dist, rdist, ldist)
        fr += np.exp(-(dist**2) / sigma**2)
    return fr


def gaussian_kernel_cuda(f, xgrid, sigma):
    """
    Courtesy of: https://github.com/sreyas-mohan/DeepFreq
    Create a frequency representation with a Gaussian kernel.
    """
    fr = np.zeros((f.shape[0], xgrid.shape[0]))
    for i in tqdm(range(f.shape[1])):
        dist = np.abs(xgrid[None, :] - f[:, i][:, None])
        rdist = np.abs(xgrid[None, :] - (f[:, i][:, None] + 1))
        ldist = np.abs(xgrid[None, :] - (f[:, i][:, None] - 1))
        dist = np.minimum(dist, rdist, ldist)
        dist = torch.from_numpy(dist).to("cuda")
        fr_temp = torch.exp(-(dist**2) / sigma**2).cpu().numpy()
        fr += fr_temp
    return fr


def triangle(f, xgrid, slope):
    """
    Courtesy of: https://github.com/sreyas-mohan/DeepFreq
    Create a frequency representation with a triangle kernel.
    """
    fr = np.zeros((f.shape[0], xgrid.shape[0]))
    for i in tqdm(range(f.shape[1])):
        dist = np.abs(xgrid[None, :] - f[:, i][:, None])
        rdist = np.abs(xgrid[None, :] - (f[:, i][:, None] + 1))
        ldist = np.abs(xgrid[None, :] - (f[:, i][:, None] - 1))
        dist = np.minimum(dist, rdist, ldist)
        fr += np.clip(1 - slope * dist, 0, 1)
    return fr


def periodogram(signal, xgrid):
    """
    Courtesy of: https://github.com/sreyas-mohan/DeepFreq
    Compute periodogram.
    """
    js = np.arange(signal.shape[1])
    return (
        np.abs(
            np.exp(-2.0j * np.pi * xgrid[:, None] * js).dot(signal.T) / signal.shape[1]
        )
        ** 2
    ).T


def periodogram_torch(signal, xgrid):
    """
    Courtesy of: https://github.com/sreyas-mohan/DeepFreq
    Compute periodogram.
    """
    js = torch.arange(signal.shape[1])
    return (
        torch.abs(
            torch.exp(-2.0j * np.pi * xgrid[:, None] * js) @ signal.T / signal.shape[1]
        )
        ** 2
    ).T


def make_hankel(signal, m):
    """
    Courtesy of: https://github.com/sreyas-mohan/DeepFreq
    Auxiliary function used in MUSIC.
    """
    n = len(signal)
    h = np.zeros((m, n - m + 1), dtype="complex128")
    for r in range(m):
        for c in range(n - m + 1):
            h[r, c] = signal[r + c]
    return h


def music(signal, xgrid, nfreq, m=20):
    """
    Courtesy of: https://github.com/sreyas-mohan/DeepFreq
    Compute frequency representation obtained with MUSIC.
    """
    music_fr = np.zeros((signal.shape[0], len(xgrid)))
    for n in range(signal.shape[0]):
        hankel = make_hankel(signal[n], m)
        _, _, V = np.linalg.svd(hankel)
        v = np.exp(
            -2.0j
            * np.pi
            * np.outer(xgrid[:, None], np.arange(0, signal.shape[1] - m + 1))
        )
        u = V[nfreq[n] :]
        fr = -np.log(np.linalg.norm(np.tensordot(u, v, axes=(1, 1)), axis=0) ** 2)
        music_fr[n] = fr
    return music_fr


def make_hankel_torch(signal, m):
    """
    Modified from: https://github.com/sreyas-mohan/DeepFreq
    Auxiliary function used in MUSIC.
    """
    n = len(signal)
    h = torch.zeros((m, n - m + 1), dtype=torch.cfloat, device=signal.device)
    for r in range(m):
        for c in range(n - m + 1):
            h[r, c] = signal[r + c]
    return h


def music_torch(signal, xgrid, nfreq, m=20):
    """
    Modified from: https://github.com/sreyas-mohan/DeepFreq
    Compute frequency representation obtained with MUSIC.
    """
    music_fr = torch.zeros((signal.shape[0], len(xgrid)))
    
    # Precompute to save time
    v = torch.exp(
        -2.0j
        * torch.pi
        * torch.outer(xgrid, torch.arange(0, signal.shape[1] - m + 1))
    )
    
    for n in range(signal.shape[0]):
        hankel = make_hankel_torch(signal[n], m)
        _, _, V = torch.linalg.svd(hankel)
        u = V[nfreq[n] :]
        fr = torch.linalg.norm(u @ v.T.type(torch.cfloat), axis=0) ** -2
        music_fr[n] = (fr-fr.min())/(fr.max() - fr.min()) # Scale between 0 and 1
    return music_fr


def omp_torch(D, X, L):
    """Sparse coding of a group of signals based on a given dictionary
    and specified number of atoms to use.
    ||X - DA||
    Uses MATLAB style indexing

    Args:
        D (torch.Tensor): the dictionary (its columns MUST be normalized)
        X (torch.Tensor): the signals to represent
        L (torch.Tensor): the max. number of coefficients for each signal
        
    Outputs:
        A (torch.Tensor): sparse coefficients matrix
    """
    
    _, P = X.shape
    _, K = D.shape # D.shape = (n, K)
    
    A = torch.zeros(K, P, dtype=torch.cfloat)
    
    for k in range(P):
        a = []
        x = X[:, k]             # k-th signal sample
        residual = x            # initialize the residual vector
        indx = torch.zeros(L[k], dtype=int)   # initialize the index vector
        
        for j in range(L[k]):
            # compute the inner product
            proj = D.H @ residual
            
            indx[j] = int(proj.abs().argmax())
            
            # solve the least squares problem
            a = torch.pinverse(D[:, indx[:(j+1)]]) @ x
            
            # compute the residual in the new dictionary
            residual = x - D[:, indx[:(j+1)]] @ a
            
        temp = torch.zeros(K, dtype=torch.cfloat)
        temp[indx[:(j+1)]] = a
        A[:, k] = temp
        
    return A