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
    im = np.array(im)
    if np.sum(np.abs(np.array(size_xy) - np.array([400, 400]))) > 1e-3:
        zf = np.array(size_xy) / np.array([400, 400])
        im = zoom(im, zf)
    if as_torch_tensor:
        im = torch.from_numpy(im)
    return im
