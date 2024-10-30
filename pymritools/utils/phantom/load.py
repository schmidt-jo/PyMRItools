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

    def get_sub_sampled_k_space(
            self, shape: tuple, acceleration: int, ac_lines: int = 20, mode: str = "skip",
            as_torch_tensor: bool = True) -> np.ndarray | torch.Tensor:
        """
        Generate a sub-sampled k-space of Shepp Logan phantom.
        The sub-sampling has an autocalibration region, i.e. fully sampled central lines,
        and the mode defines the sampling.
        There are the options:
        - 'skip' to skip every line except multiples of the acceleration factors
        - 'weighted' pseudo random sampling with a window function weighting the central lines more than outer ones.
        :param shape:
        :param acceleration:
        :param ac_lines:
        :param as_torch_tensor:
        :param mode:
        :return:
        """
        modes = ["skip", "weighted"]
        # allocate
        nx, ny = shape
        k = np.zeros((nx, ny), dtype=np.complex128)
        y_center = int(ny / 2)
        # calculate upper and lower edges of ac region
        y_l = y_center - int(ac_lines / 2)
        y_u = y_center + int(ac_lines / 2)
        # build indices of ac lines
        indices_ac = np.arange(y_l, y_u)
        if mode == "skip":
             # build indices of remaining lines sampled outer lines
            indices = np.concatenate(
                (indices_ac, np.arange(0, y_l, acceleration), np.arange(y_u, shape[1], acceleration)),
                axis=0
            )
        elif mode == "weighted":
            # set some factor for weighting central lines preferably
            weighting_factor = 0.3
            # build weighting function and normalize
            weighting = np.clip(
                np.power(
                    np.linspace(0, 1, y_l), weighting_factor
                ),
                1e-5, 1
            )
            weighting /= np.sum(weighting)
            # draw samples from the outer lines
            num_outer_lines = int((shape[1] - ac_lines) / acceleration)
            rng = np.random.default_rng(0)
            indices = rng.choice(
                np.arange(0, y_l),
                size=num_outer_lines,
                replace=False,
                p=weighting
            )
            # move half of them to other side of ac lines
            indices[::2] =shape[1] - 1 - indices[::2]
            indices = np.concatenate(
                (indices_ac, np.sort(indices)),
                axis=0
            )
        else:
            err = f"Mode {mode} not supported, should be one of: {modes}"
            log_module.error(err)
            raise ValueError(err)
        k[:, indices] = self.get_2D_k_space(shape=shape, as_torch_tensor=False)[:, indices]
        if as_torch_tensor:
            return torch.from_numpy(k)
        return k

    # ToDo: can implement coil sensitivities as well



