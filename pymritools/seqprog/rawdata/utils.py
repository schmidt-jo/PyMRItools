import numpy as np
import logging

import torch

from pymritools.utils import fft

log_module = logging.getLogger(__name__)


def remove_oversampling(data: np.ndarray | torch.Tensor, data_in_k_space: bool = True,
                        read_dir: int = 0, os_factor: int = 2):
    if np.abs(os_factor - 1) < 1e-9:
        return data
    log_module.debug(f"remove oversampling")
    # get read direction dimension
    nx = data.shape[read_dir]

    if data_in_k_space:
        # need to transform
        data = fft(input_data=data, img_to_k=False, axes=read_dir)

    # data in freq domain, do removal
    lower_idx = int((os_factor - 1) / (2 * os_factor) * nx)
    upper_idx = int((os_factor + 1) / (2 * os_factor) * nx)
    if not torch.is_tensor(data):
        move_func = np.moveaxis
    else:
        move_func = torch.movedim
    data = move_func(data, read_dir, 0)[lower_idx:upper_idx]
    data = move_func(data, 0, read_dir)
    if data_in_k_space:
        # data was in k domain originally, hence we move back
        data = fft(input_data=data, img_to_k=True, axes=read_dir)
    return data
