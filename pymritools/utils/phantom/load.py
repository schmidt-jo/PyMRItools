import pathlib as plib
import logging
from PIL import Image
import numpy as np
from scipy.ndimage import zoom
import torch

log_module = logging.getLogger(__name__)


def shepp_logan_phantom(size_xy: tuple = (400, 400), as_torch_tensor: bool = True) -> np.ndarray | torch.Tensor:
    # hard code image path
    path = plib.Path(__file__).absolute().parent.joinpath("phantom").with_suffix(".png")
    im = Image.open(path).convert("L")
    im = np.array(im, dtype=np.float32)
    max_img = np.max(im)
    zero_mask = im == 0
    if np.sum(np.abs(np.array(size_xy) - np.array([400, 400]))) > 1e-3:
        zf = np.array(size_xy) / np.array([400, 400])
        im = zoom(im, zf, order=0)
        zero_mask = zoom(zero_mask, zf, order=0)
    # normalize to max 1
    im /= max_img
    im[zero_mask] = 0
    if as_torch_tensor:
        im = torch.from_numpy(im)
    return im
