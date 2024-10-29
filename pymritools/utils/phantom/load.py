import pathlib as plib
import logging
from PIL import Image
import numpy as np
from scipy.ndimage import zoom
import torch

from pymritools.utils import fft

log_module = logging.getLogger(__name__)


class SheppLogan:
    def __init__(self):
        path = plib.Path(__file__).absolute().parent.joinpath("phantom").with_suffix(".png")
        im = Image.open(path).convert("L")
        self._im = np.array(im, dtype=np.float32)

    def get_2D_image(self, shape: tuple, as_torch_tensor: bool = True) -> np.ndarray | torch.Tensor:
        im = self._im
        zero_mask = self._im == 0
        if np.sum(np.abs(np.array(shape[:2]) - np.array([400, 400]))) > 1e-3:
            zf = np.array(shape[:2]) / np.array([400, 400])
            im = zoom(im, zf, order=0)
            zero_mask = zoom(zero_mask, zf, order=0)
        max_img = np.max(im, axis=(0, 1))
        # normalize to max 1
        im /= max_img
        im[zero_mask] = 0
        if as_torch_tensor:
            im = torch.from_numpy(im)
        return im

    def get_2D_k_space(self, shape: tuple, as_torch_tensor: bool = True) -> np.ndarray | torch.Tensor:
        im = self.get_2D_image(shape=shape, as_torch_tensor=as_torch_tensor)
        return fft(input_data=im, img_to_k=True, axes=(0, 1))

    def get_subsampled_k_space(self, shape: tuple, acceleration: int, ac_lines: int = 20,
                               as_torch_tensor: bool = True) -> np.ndarray | torch.Tensor:
        k = self.get_2D_k_space(shape=shape, as_torch_tensor=as_torch_tensor)
        nx, ny = shape
        y_center = int(ny / 2)
        y_l = y_center - int(ac_lines / 2)
        y_u = y_center + int(ac_lines / 2)

        for y in range(shape[1]):
            if y_l < y < y_u:
                continue
            else:
                if y % acceleration != 0:
                    k[:, y] = 0
        return k

    # ToDo: can implement coil sensitivities as well



