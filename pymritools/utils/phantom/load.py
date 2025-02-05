import pathlib as plib
import logging
from typing import Tuple

from PIL import Image
import numpy as np
from scipy.ndimage import zoom
import torch

from pymritools.utils import fft, gaussian_2d_kernel

log_module = logging.getLogger(__name__)
seed = 10


class Phantom:
    def __init__(
            self, shape: tuple[int, int], num_coils: int = 1, num_echoes: int = 1,
            mode: str = "sl", cs_mode: str = "random"):
        self.shape = shape
        self.num_coils = num_coils
        self.num_echoes = num_echoes

        # we can load different phantom data defined by mode
        match mode:
            case "sl":
                path = plib.Path(__file__).absolute().parent.joinpath("phantom").with_suffix(".png")
                im = Image.open(path).convert("L")
            case "jupiter":
                path = plib.Path(__file__).absolute().parent.joinpath("jupiter_512").with_suffix(".png")
                im = Image.open(path).convert("L")
            case _:
                raise ValueError(f"Unknown phantom mode: {mode}, choose one of 'sl' (shepp logan), 'jupiter'")
        self._im = np.array(im, dtype=np.float32)
        # save 0 points
        zero_mask = self._im == 0

        native_shape = self._im.shape
        # scale image to shape
        if np.sum(np.abs(np.array(shape[:2]) - np.array(native_shape))) > 1e-3:
            zf = np.array(shape[:2]) / np.array(native_shape)
            self._im = zoom(self._im, zf, order=0)
            zero_mask = zoom(zero_mask, zf, order=0)

        # extract unique values as labels
        labels = np.unique(self._im)
        label_img = self._im.copy()

        # want to assign rates to the label images
        rng = np.random.default_rng(seed)
        # assign rates, make them relatively small
        random_rates = rng.random(len(labels)) * 0.2
        for i, r in enumerate(random_rates.tolist()):
            label_img[np.where(label_img == labels[i])] = r

        max_img = np.max(self._im, axis=(0, 1))
        # normalize to max 1
        self._im /= max_img
        # reset zeros (changed due to interpolation method in above scaling
        self._im[zero_mask] = 0
        self._im = torch.from_numpy(self._im)
        label_img[zero_mask] = 0
        label_img = torch.from_numpy(label_img)

        if num_echoes > 1:
            # if we want echo images we use the label / rates image and create a decay
            self._im = self._im[:, :, None] * torch.exp(-label_img[:, :, None] * torch.arange(num_echoes)[None, None, :])
        else:
            self._im = self._im[:, :, None]

        if num_coils > 1:
            match cs_mode:
                case "random":
                    self._cs = self._get_random_cs(num_coils=num_coils)
                case "terra":
                    if num_coils > 64:
                        raise AttributeError(f"num_coils ({num_coils}) exceeds available coils (64) "
                                             f"of terra sensitivity profile")
                    path = plib.Path(__file__).absolute().parent.joinpath("terra_cs").with_suffix(".png")
                    log_module.info(f"Random CS mode: {cs_mode}, using Terra coil sensitivity profile, "
                                    f"with 64 coils and picking {num_coils} randomly")
                    im = Image.open(path).convert("L")
                    native_shape = im.size
                    self._cs = torch.from_numpy(im).permute(dims=(-1,))[:num_coils]
                    zf = np.array(shape[:2]) / np.array(native_shape)
                    self._cs = zoom(self._cs, zf, order=0)
                case _:
                    raise ValueError(f"Unknown phantom mode: {cs_mode}, choose one of 'random', 'terra'")
            if num_echoes > 1:
                self._cs = self._cs.unsqueeze(-1)

            self._im = self._im[:, :, None] * self._cs
            self.data = self._im

    def _get_random_cs(self, num_coils: int):
        nx, ny = self.shape[:2]
        torch.manual_seed(seed)
        coil_sens = torch.ones((*self.shape, num_coils))
        for i in range(num_coils):
            center_x = torch.randint(low=int(nx / 12), high=int(11 * nx / 12), size=(1,))
            center_y = torch.randint(low=int(ny / 12), high=int(11 * ny / 12), size=(1,))
            gw = gaussian_2d_kernel(
                size_x=nx, size_y=ny,
                center_x=center_x.item(), center_y=center_y.item(), sigma=(40, 60)
            )
            gw /= torch.max(gw)
            coil_sens[:, :, i] = gw
        return coil_sens

    def get_2D_image(self) -> torch.Tensor:
        return self.data

    def get_2D_k_space(self) -> torch.Tensor:
        im = self.get_2D_image()
        return fft(input_data=im, img_to_k=True, axes=(0, 1))

    def get_sub_sampled_k_space(
            self, acceleration: int, ac_lines: int = 20, mode: str = "skip") -> np.ndarray | torch.Tensor:
        """
        Generate a sub-sampled k-space of Shepp Logan phantom.
        The sub-sampling has an autocalibration region, i.e. fully sampled central lines,
        and the mode defines the sampling.
        There are the options:
        - 'skip' to skip every line except multiples of the acceleration factors
        - 'weighted' pseudo random sampling with a window function weighting the central lines more than outer ones.
        - 'random' random sampled points in image, no AC region.
        :param acceleration:
        :param ac_lines:
        :param mode:
        :return:
        """
        modes = ["grappa", "skip", "weighted", "random"]
        # allocate
        nx, ny = self.shape[:2]

        # get fs k - space
        k_fs = self.get_2D_k_space()
        k_us = np.zeros_like(k_fs)

        y_center = int(ny / 2)
        # calculate upper and lower edges of ac region
        y_l = y_center - int(ac_lines / 2)
        y_u = y_center + int(ac_lines / 2)
        # fill ac region
        k_us[:, y_l:y_u] = k_fs[:, y_l:y_u]
        if mode == "grappa":
            k_us[:, :y_l:acceleration] = k_fs[:, :y_l:acceleration]
            k_us[:, y_u::acceleration] = k_fs[:, y_u::acceleration]
        elif mode == "skip":
            # we fill every acc._factor line in outer k_space and move one step for every echo
            for e in range(self.num_echoes):
                k_us[:, e:y_l:acceleration, ..., e] = k_fs[:, e:y_l:acceleration, ..., e]
                k_us[:, y_u + e::acceleration, ..., e] = k_fs[:, y_u+e::acceleration, ..., e]

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
            num_outer_lines = int((self.shape[1] - ac_lines) / acceleration)
            rng = np.random.default_rng(seed)
            indices = np.stack([
                rng.choice(
                    np.arange(0, y_l),
                    size=(num_outer_lines),
                    replace=False,
                    p=weighting
                ) for e in range(self.num_echoes)
                ],
                axis=1
            )
            # move half of them to other side of ac lines
            indices[::2] = self.shape[1] - 1 - indices[::2]
            indices = np.sort(indices, axis=0)
            for e in range(self.num_echoes):
                k_us[:, indices[:, e], ..., e] = k_fs[:, indices[:, e], ..., e]
        elif mode == "random":
            rng = np.random.default_rng(seed)
            i = np.array([[x, y] for x in np.arange(256) for y in np.arange(256)])
            for e in range(self.num_echoes):
                indices = rng.choice(i, size=int(i.shape[0] / acceleration), replace=False)
                k_us[indices[:, 0], indices[:, 1], ..., e] = k_fs[indices[:, 0], indices[:, 1], ..., e]
        else:
            err = f"Mode {mode} not supported, should be one of: {modes}"
            log_module.error(err)
            raise ValueError(err)
        k_us = torch.from_numpy(np.squeeze(k_us))
        return k_us


class JupiterImage:
    def __init__(self):
        image_path = plib.Path(__file__).absolute().parent.joinpath("jupiter_512").with_suffix(".png")
        mask_path = plib.Path(__file__).absolute().parent.joinpath("jupiter_mask_512").with_suffix(".pnm")
        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        self._image = np.array(image, dtype=np.float32)
        self._mask = np.array(mask, dtype=np.bool)

    def as_torch_tensor(self) -> Tuple[torch.Tensor, ...]:
        t_im = torch.from_numpy(self._image)
        t_mask = torch.from_numpy(self._mask).to(dtype=torch.bool).unsqueeze(0)
        return (
            t_im,
            fft(input_data=t_im, img_to_k=True, axes=(0, 1)),
            t_mask
        )
