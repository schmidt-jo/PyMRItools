import numpy as np
import torch
import logging
log_module = logging.getLogger(__name__)


def fft(
        input_data: np.ndarray | torch.Tensor,
        inverse: bool = False, axes: tuple = (-1, -2)):
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
