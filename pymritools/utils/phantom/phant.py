import pathlib as plib
import logging
from typing import Tuple

from PIL import Image
import numpy as np
from scipy.ndimage import zoom
import torch

from pymritools.utils import fft, gaussian_2d_kernel

log_module = logging.getLogger(__name__)


class Phantom:
    def __init__(
            self, shape: tuple, num_coils: int = 1, num_echoes: int = 1):
        self.shape: tuple = shape
        self.num_coils: int = num_coils
        self.num_echoes: int = num_echoes
        # set seed for reproducibility
        self.seed: int = 10

        self._img: np.ndarray = np.empty(shape)
        self._cs: np.ndarray = np.ones((*shape, num_coils))
        self._e: np.ndarray = np.ones((*shape, num_echoes))

    # __ private
    def _set_image(self, img: Image):
        self._img = np.array(img, dtype=np.float32)
        # get all pts with no data
        zero_mask = self._img == 0
        # get shape of loaded image
        native_shape = self._img.shape
        # scale image to shape
        if np.sum(np.abs(np.array(self.shape[:2]) - np.array(native_shape))) > 1e-3:
            # calculate zoom factor to get wanted shape
            zf = np.array(self.shape[:2]) / np.array(native_shape)
            self._img = zoom(self._img, zf, order=0)
            zero_mask = zoom(zero_mask, zf, order=0)
            # reset zeros (changed due to interpolation method in above scaling
            self._img[zero_mask] = 0

        max_img = np.max(self._img, axis=(0, 1))
        # normalize to max 1
        self._img /= max_img

    def _build_echo_imgs(self):
        # build echo images if wanted
        if self.num_echoes > 1:
            # extract unique values as labels
            labels = np.unique(self._img)
            label_img = self._img.copy()
            zero_mask = self._img == 0

            # want to assign rates to the label images
            rng = np.random.default_rng(self.seed)
            # assign rates, make them relatively small
            random_rates = rng.random(len(labels)) * 0.2
            for i, r in enumerate(random_rates.tolist()):
                label_img[np.where(label_img == labels[i])] = r

            label_img[zero_mask] = 0
            # if we want echo images we use the label / rates image and create a decay
            self._e = np.exp(-label_img[:, :, None] * np.arange(self.num_echoes)[None, None, :])

    def _build_coil_imgs(self, cs_mode: str = "random"):
        # build coil images if wanted
        if self.num_coils > 1:
            match cs_mode:
                case "random":
                    self._cs = self._get_random_cs()
                case "terra":
                    self._cs = self._get_terra_cs()
                case _:
                    raise ValueError(f"Unknown phantom mode: {cs_mode}, choose one of 'random', 'terra'")

    def _get_terra_cs(self) -> np.ndarray:
        if self.num_coils > 64:
            raise AttributeError(f"num_coils ({self.num_coils}) exceeds available coils (64) "
                                 f"of terra sensitivity profile")
        path = plib.Path(__file__).absolute().parent.joinpath("terra_cs").with_suffix(".png")
        log_module.info(f"Using Terra coil sensitivity profile, "
                        f"with 64 coils and picking {self.num_coils} randomly")
        im = Image.open(path).convert("L")
        native_shape = im.size
        self._cs = torch.from_numpy(im).permute(dims=(-1,))[:self.num_coils]
        zf = np.array(self.shape[:2]) / np.array(native_shape)
        return zoom(self._cs, zf, order=0)

    def _get_random_cs(self) -> np.ndarray:
        nx, ny = self.shape[:2]
        torch.manual_seed(self.seed)
        coil_sens = torch.ones((*self.shape, self.num_coils))
        for i in range(self.num_coils):
            center_x = torch.randint(low=int(nx / 12), high=int(11 * nx / 12), size=(1,))
            center_y = torch.randint(low=int(ny / 12), high=int(11 * ny / 12), size=(1,))
            gw = gaussian_2d_kernel(
                size_x=nx, size_y=ny,
                center_x=center_x.item(), center_y=center_y.item(), sigma=(40, 60)
            )
            gw /= torch.max(gw)
            coil_sens[:, :, i] = gw
        return coil_sens.numpy()

    def _get_ac_sampled_k(self, ac_lines: int) -> (torch.Tensor, torch.Tensor, int, int):
        # allocate
        nx, ny = self.shape[:2]

        # get fs k - space
        k_fs = self.get_2d_k_space()
        k_us = torch.zeros_like(k_fs)

        y_center = int(ny / 2)
        # calculate upper and lower edges of ac region
        y_l = y_center - int(ac_lines / 2)
        y_u = y_center + int(ac_lines / 2)
        # fill ac region
        k_us[:, y_l:y_u] = k_fs[:, y_l:y_u]
        if self.num_echoes <= 1:
            k_us.unsqueeze_(-1)
            k_fs.unsqueeze_(-1)
        return k_us, k_fs, y_l, y_u

    @classmethod
    def get_shepp_logan(cls, shape: tuple, num_coils: int = 1, num_echoes: int = 1, cs_mode: str = "random"):
        phantom = cls(shape=shape, num_coils=num_coils, num_echoes=num_echoes)
        # load shepp logan image
        path = plib.Path(__file__).absolute().parent.joinpath("phantom").with_suffix(".png")
        # set image
        phantom._set_image(Image.open(path).convert("L"))
        phantom._build_coil_imgs(cs_mode=cs_mode)
        phantom._build_echo_imgs()
        return phantom

    @classmethod
    def get_jupiter(cls, shape: tuple, num_coils: int = 1, num_echoes: int = 1):
        phantom = cls(shape=shape, num_coils=num_coils, num_echoes=num_echoes)
        # set image
        path = plib.Path(__file__).absolute().parent.joinpath("jupiter_512").with_suffix(".png")
        for s in shape:
            if s > 512:
                path = plib.Path(__file__).absolute().parent.joinpath("jupiter").with_suffix(".png")
                break
        phantom._set_image(Image.open(path).convert("L"))
        phantom._build_coil_imgs()
        phantom._build_echo_imgs()
        return phantom

    # __ public
    def get_2d_image(self) -> torch.Tensor:
        data = self._img[:, :, None, None] * self._cs[:, :, :, None] * self._e[:, :, None, :]
        return torch.squeeze(torch.from_numpy(data))

    def get_2d_k_space(self) -> torch.Tensor:
        im = self.get_2d_image()
        return fft(input_data=im, img_to_k=True, axes=(0, 1))

    # __ subsampling
    def sub_sample_ac_skip_lines(self, acceleration: int, ac_lines: int) -> torch.Tensor:
        k_us, k_fs, y_l, y_u = self._get_ac_sampled_k(ac_lines=ac_lines)
        # we fill every acc._factor line in outer k_space and move one step for every echo
        for e in range(self.num_echoes):
            k_us[:, e:y_l:acceleration, ..., e] = k_fs[:, e:y_l:acceleration, ..., e]
            k_us[:, y_u + e::acceleration, ..., e] = k_fs[:, y_u + e::acceleration, ..., e]
        return torch.squeeze(k_us)

    def sub_sample_ac_random_lines(self, acceleration: int, ac_lines: int) -> torch.Tensor:
        return self.sub_sample_ac_weighted_lines(acceleration=acceleration, ac_lines=ac_lines, weighting_factor=0.0)

    def sub_sample_ac_weighted_lines(self, acceleration: int, ac_lines: int, weighting_factor: float = 0.3) -> torch.Tensor:
        k_us, k_fs, y_l, y_u = self._get_ac_sampled_k(ac_lines=ac_lines)
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
        rng = np.random.default_rng(self.seed)
        indices = np.stack([
            rng.choice(
                np.arange(0, y_l),
                size=num_outer_lines,
                replace=False,
                p=weighting
            ) for e in range(self.num_echoes)
        ],
            axis=1
        )
        # move half of them to other side of ac lines
        indices[::2] = self.shape[1] - 1 - indices[::2]
        indices = torch.from_numpy(np.sort(indices, axis=0))
        for e in range(self.num_echoes):
            k_us[:, indices[:, e], ..., e] = k_fs[:, indices[:, e], ..., e]
        return torch.squeeze(k_us)

    def sub_sample_ac_grappa(self, acceleration: int, ac_lines: int) -> torch.Tensor:
        k_us, k_fs, y_l, y_u = self._get_ac_sampled_k(ac_lines=ac_lines)
        k_us[:, :y_l:acceleration] = k_fs[:, :y_l:acceleration]
        k_us[:, y_u::acceleration] = k_fs[:, y_u::acceleration]
        return torch.squeeze(k_us)

    def sub_sample_random(self, acceleration: int, ac_central_radius: int = 5) -> torch.Tensor:
        k_fs = self.get_2d_k_space()
        k_us = torch.zeros_like(k_fs)
        if self.num_echoes <= 1:
            k_us.unsqueeze_(-1)
            k_fs.unsqueeze_(-1)
        rng = np.random.default_rng(self.seed)
        # always fill center
        i_central = np.array(
            [
                [x + int(self.shape[0] / 2), y + int(self.shape[1] / 2)]
                for x in np.arange(-ac_central_radius, ac_central_radius)
                for y in np.arange(-ac_central_radius, ac_central_radius)
                if x**2 + y**2 <= ac_central_radius**2
            ])
        i = np.array([[x, y] for x in np.arange(self.shape[0]) for y in np.arange(self.shape[1])])
        for e in range(self.num_echoes):
            indices = torch.from_numpy(rng.choice(i, size=int(i.shape[0] / acceleration), replace=False))
            k_us[i_central[:, 0], i_central[:, 1], ..., e] = k_fs[i_central[:, 0], i_central[:, 1], ..., e]
            k_us[indices[:, 0], indices[:, 1], ..., e] = k_fs[indices[:, 0], indices[:, 1], ..., e]
        return torch.squeeze(k_us)


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


def main():
    psl = Phantom.get_shepp_logan(shape=(256, 256), num_coils=4, num_echoes=2)
    pj = Phantom.get_jupiter(shape=(256, 256))

    psl_us = psl.sub_sample_ac_weighted_lines(acceleration=3, ac_lines=30, weighting_factor=0.3)
    pj_us = pj.sub_sample_random(acceleration=3)
    print("done")

if __name__ == '__main__':
    main()