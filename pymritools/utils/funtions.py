import numpy as np
import torch
import logging
log_module = logging.getLogger(__name__)


def fft(
        input_data: np.ndarray | torch.Tensor,
        inverse: bool = False, axes: tuple | int = (-1, -2)):
    """
    from image space to k-space: inverse - False
    """
    if isinstance(axes, int):
        # make tuple
        axes = (axes,)
    if torch.is_tensor(input_data):
        if inverse:
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
        if inverse:
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

