import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIM1d(nn.Module):
    """
    Typical SSIM Loss Function modified for 1-D tensors

    Code modified from: https://gitlab.com/computational-imaging-lab/perp_loss
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03) -> None:
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size) / win_size)
        NP = win_size
        self.cov_norm = NP / (NP - 1)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        data_range=None,
        full: bool = False,
    ) -> torch.Tensor:
        assert isinstance(self.w, torch.Tensor)
        assert x.dim() == 1 and y.dim() == 1, "x and y must be vectors"

        x = x[None, None, :]
        y = y[None, None, :]

        if data_range is None:
            data_range = y.max()

        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv1d(x, self.w)
        uy = F.conv1d(y, self.w)
        uxx = F.conv1d(x * x, self.w)
        uyy = F.conv1d(y * y, self.w)
        uxy = F.conv1d(x * y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux**2 + uy**2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        if full:
            return S
        else:
            return S.mean()


class RMSE1d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        assert x.dim() == 1 and y.dim() == 1, "x and y must be vectors"

        return (x - y).pow(2).mean().sqrt()


class PSNR1d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, data_range=None):
        assert x.dim() == 1 and y.dim() == 1, "x and y must be vectors"

        if data_range is None:
            data_range = y.max()

        mse = (x - y).pow(2).mean()

        return 10 * torch.log10(data_range / mse)
