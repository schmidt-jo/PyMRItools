import numpy as np
import torch
import logging
log_module = logging.getLogger(__name__)


def interpolate(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """
    Interpolate a function fp at points xp in a multidimensional context

    Parameters:
    x (torch.Tensor): Tensor of the new sampling points with shape [batch, a, b]
    xp (torch.Tensor): 1D Tensor of original sample points with shape [c]
    fp (torch.Tensor): 2D Tensor of function values at xp with shape [a, c]

    Returns:
    torch.Tensor: Interpolated values with shape [batch, a, b]
    """
    while len(x.shape) < 3:
        # in case there is no batch dim we fill
        x = x.unsqueeze(0)
    batch, a, b = x.shape
    # find closest upper adjacent indices of x in xp, then the next lower one
    indices = torch.searchsorted(xp, x.view(-1, b))
    indices = torch.clamp(indices, 1, xp.shape[0] - 1)
    # find adjacent left and right points on originally sampled axes xp
    x0 = xp[indices - 1]
    x1 = xp[indices]
    # find values of originally sampled function considering its differing for each idx_a
    fp_expanded = fp.unsqueeze(0).expand(batch, -1, -1)
    y0 = fp_expanded.gather(2, indices.view(batch, a, b) - 1)
    y1 = fp_expanded.gather(2, indices.view(batch, a, b))
    # get the slope
    slope = (y1 - y0) / (x1 - x0).view(batch, a, b)
    interpolated_values = slope * (x - x0.view(batch, a, b)) + y0
    return interpolated_values


def normalize_data(data: torch.Tensor, dim_t: int = -1) -> (torch.Tensor, torch.Tensor):
    norm_factor = torch.linalg.norm(data, dim=dim_t, keepdim=True)
    # normalize
    norm_data = torch.nan_to_num(
        torch.divide(data, norm_factor),
        nan=0.0, posinf=0.0, neginf=0.0
    )
    return norm_data, torch.squeeze(norm_factor)


def fft(
        input_data: np.ndarray | torch.Tensor,
        img_to_k: bool = False, axes: tuple | int = (-1, -2)):
    if isinstance(axes, int):
        # make tuple
        axes = (axes,)
    if torch.is_tensor(input_data):
        if img_to_k:
            func = torch.fft.ifftn
        else:
            func = torch.fft.fftn
        return torch.fft.fftshift(
            func(
                torch.fft.ifftshift(
                    input_data,
                    dim=axes
                ),
                dim=axes
            ),
            dim=axes
        )
    else:
        if img_to_k:
            func = np.fft.ifftn
        else:
            func = np.fft.fftn
        return np.fft.fftshift(
            func(
                np.fft.ifftshift(
                    input_data,
                    axes=axes
                ),
                axes=axes
            ),
            axes=axes
        )


def root_sum_of_squares(input_data: np.ndarray | torch.Tensor, dim_channel: int = -1):
    if torch.is_tensor(input_data):
        return torch.sqrt(
            torch.sum(
                torch.abs(input_data) ** 2,
                dim=dim_channel
            )
        )
    else:
        return np.sqrt(
            np.sum(
                np.abs(input_data) ** 2,
                axis=dim_channel
            )
        )


def gaussian_window(size: int, sigma: float, center: int = None):
    if center is None:
        center = int(np.round(size / 2))
    ax = torch.arange(size).float()
    return torch.exp(-((ax - center) ** 2) / (2 * sigma ** 2))



def gaussian_2d_kernel(size_x: int, size_y: int, center_x: int=None, center_y:int=None,
                       sigma: float | tuple[float, float] = 1.0):
    """
    Generates a 2D Gaussian kernel.

    Parameters:
    - size_x (int): Kernel size in the x dimension.
    - size_y (int): Kernel size in the y dimension.
    - center_x (float): Center of the Gaussian in the x dimension. If None, defaults to the center of the kernel.
    - center_y (float): Center of the Gaussian in the y dimension. If None, defaults to the center of the kernel.
    - sigma (float | tuple of floats): Standard deviation of the Gaussian.
        If a tuple is given the function uses different gaussian scale per dim respectively.

    Returns:
    - kernel (torch.Tensor): 2D Gaussian kernel.
    """
    if isinstance(sigma, float | int):
        sigma = (sigma, sigma)
    gauss_x = gaussian_window(size_x, sigma[0], center=center_x)
    guass_y = gaussian_window(size_y, sigma[1], center=center_y)

    gauss_2d = torch.matmul(
        gauss_x.unsqueeze(-1), guass_y.unsqueeze(0)
    )
    # normalize
    gauss_2d /= gauss_2d.sum()

    return gauss_2d

