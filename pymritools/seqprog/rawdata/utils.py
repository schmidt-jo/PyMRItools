import numpy as np
import logging
from pymritools.utils import fft

log_module = logging.getLogger(__name__)


def remove_oversampling(data: np.ndarray, data_input_sampled_in_time: bool = True, read_dir: int = 0, os_factor: int = 2):
    log_module.info(f"remove oversampling")
    # get read direction dimension
    nx = data.shape[read_dir]

    if data_input_sampled_in_time:
        # data not in frequency domain, need to transform
        data = fft(input_data=data, inverse=False, axes=read_dir)

    # data in freq domain, do removal
    lower_idx = int((os_factor - 1) / (2 * os_factor) * nx)
    upper_idx = int((os_factor + 1) / (2 * os_factor) * nx)
    data = np.moveaxis(data, read_dir, 0)[lower_idx:upper_idx]
    data = np.moveaxis(data, 0, read_dir)

    if data_input_sampled_in_time:
        # data was in time domain originally, hence we move back
        data = fft(input_data=data, inverse=True, axes=read_dir)
    return data
