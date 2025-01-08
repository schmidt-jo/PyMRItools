import pathlib as plib
import logging

from PIL import Image
import numpy as np
from scipy.ndimage import zoom
import torch
from scipy.io import savemat

from pymritools.utils import fft, gaussian_2d_kernel, torch_save

log_module = logging.getLogger(__name__)
seed = 10
torch.manual_seed(seed)


class SheppLogan:
    def __init__(self):
        path = plib.Path(__file__).absolute().parent.joinpath("phantom").with_suffix(".png")
        im = Image.open(path).convert("L")
        self._im = np.array(im, dtype=np.float32)

    def get_2D_image(
            self, shape: tuple, as_torch_tensor: bool = True, num_coils: int = None, num_echoes: int = None
    ) -> np.ndarray | torch.Tensor:
        # load the image
        im = self._im
        # extract unique values as labels
        labels = np.unique(im)
        label_img = im.copy()
        # want to assign rates to the label images
        rng = np.random.default_rng(seed)
        # assign rates, make them relatively small
        random_rates = rng.random(len(labels)) * 0.2
        for i, r in enumerate(random_rates.tolist()):
            label_img[np.where(label_img == labels[i])] = r

        # save 0 points
        zero_mask = self._im == 0
        # scale image to shape
        if np.sum(np.abs(np.array(shape[:2]) - np.array([400, 400]))) > 1e-3:
            zf = np.array(shape[:2]) / np.array([400, 400])
            im = zoom(im, zf, order=0)
            zero_mask = zoom(zero_mask, zf, order=0)
            label_img = zoom(label_img, zf, order=0)

        max_img = np.max(im, axis=(0, 1))
        # normalize to max 1
        im /= max_img
        # reset zeros (changed due to interpolation method in above scaling
        im[zero_mask] = 0
        im = torch.from_numpy(im)
        label_img[zero_mask] = 0
        label_img = torch.from_numpy(label_img)

        # save within slice shape
        nx, ny = shape[:2]

        if num_echoes is not None:
            # if we want echo images we use the label / rates image and create a decay
            im = im[:, :, None] * torch.exp(-label_img[:, :, None] * torch.arange(num_echoes)[None, None, :])
        else:
            im = im[:, :, None]
        # include fake coil sensitivities randomly set up
        if num_coils is not None:
            coil_sens = torch.ones((*shape, num_coils))
            if num_coils > 1:
                for i in range(num_coils):
                    center_x = torch.randint(low=int(nx / 12), high=int(11 * nx / 12), size=(1,))
                    center_y = torch.randint(low=int(ny / 12), high=int(11 * ny / 12), size=(1,))
                    gw = gaussian_2d_kernel(
                        size_x=nx, size_y=ny,
                        center_x=center_x.item(), center_y=center_y.item(), sigma=(40, 60)
                    )
                    gw /= torch.max(gw)
                    coil_sens[:, :, i] = gw
            if num_echoes is not None:
                coil_sens = coil_sens.unsqueeze(-1)
            im = im[:, :, None] * coil_sens
        if not as_torch_tensor:
            im = im.numpy()
        return im

    def get_2D_k_space(self, shape: tuple, as_torch_tensor: bool = True, num_coils: int = None, num_echoes: int = None) -> np.ndarray | torch.Tensor:
        im = self.get_2D_image(shape=shape, as_torch_tensor=as_torch_tensor, num_coils=num_coils, num_echoes=num_echoes)
        return fft(input_data=im, img_to_k=True, axes=(0, 1))

    def get_sub_sampled_k_space(
            self, shape: tuple, acceleration: int, ac_lines: int = 20, mode: str = "skip",
            as_torch_tensor: bool = True, num_coils: int = None, num_echoes: int = None) -> np.ndarray | torch.Tensor:
        """
        Generate a sub-sampled k-space of Shepp Logan phantom.
        The sub-sampling has an autocalibration region, i.e. fully sampled central lines,
        and the mode defines the sampling.
        There are the options:
        - 'skip' to skip every line except multiples of the acceleration factors
        - 'weighted' pseudo random sampling with a window function weighting the central lines more than outer ones.
        - 'random' random sampled points in image, no AC region.
        :param shape:
        :param acceleration:
        :param ac_lines:
        :param as_torch_tensor:
        :param mode:
        :return:
        """
        modes = ["skip", "weighted", "random"]
        # allocate
        nx, ny = shape
        # get fs k - space
        if num_echoes is None:
            # set at least one echo
            num_echoes = 1
        k_fs = self.get_2D_k_space(shape=shape, as_torch_tensor=False, num_coils=num_coils, num_echoes=num_echoes)
        k_us = np.zeros_like(k_fs)

        y_center = int(ny / 2)
        # calculate upper and lower edges of ac region
        y_l = y_center - int(ac_lines / 2)
        y_u = y_center + int(ac_lines / 2)
        # fill ac region
        k_us[:, y_l:y_u] = k_fs[:, y_l:y_u]
        if mode == "skip":
            # we fill every acc._factor line in outer k_space and move one step for every echo
            for e in range(num_echoes):
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
            num_outer_lines = int((shape[1] - ac_lines) / acceleration)
            rng = np.random.default_rng(seed)
            indices = rng.choice(
                np.arange(0, y_l),
                size=(num_outer_lines, num_echoes),
                replace=False,
                p=weighting
            )
            # move half of them to other side of ac lines
            indices[::2] = shape[1] - 1 - indices[::2]
            indices = np.sort(indices, axis=0)
            for e in range(num_echoes):
                k_us[:, indices[:, e], ..., e] = k_fs[:, indices[:, e], ..., e]
        elif mode == "random":
            rng = np.random.default_rng(seed)
            i = np.array([[x, y] for x in np.arange(256) for y in np.arange(256)])
            for e in range(num_echoes):
                indices = rng.choice(i, size=int(i.shape[0] / acceleration), replace=False)
                k_us[indices[:, 0], indices[:, 1], ..., e] = k_fs[indices[:, 0], indices[:, 1], ..., e]
        else:
            err = f"Mode {mode} not supported, should be one of: {modes}"
            log_module.error(err)
            raise ValueError(err)
        k_us = np.squeeze(k_us)
        if as_torch_tensor:
            return torch.from_numpy(k_us)
        return k_us


def main():
    logging.basicConfig(level=logging.INFO)
    k_us = SheppLogan().get_sub_sampled_k_space(
        shape=(256, 256), acceleration=2, ac_lines=30, mode="skip", as_torch_tensor=True, num_echoes=2
    )
    k_us_cs = SheppLogan().get_sub_sampled_k_space(
        shape=(256, 256), acceleration=2, ac_lines=30, mode="weighted", as_torch_tensor=True, num_coils=32
    )
    path = plib.Path("./test_data/loraks").absolute()
    path.mkdir(exist_ok=True, parents=True)
    mat_file_name = path.joinpath("phantom").with_suffix(".mat")
    logging.info(f"Write file: {mat_file_name.as_posix()}")
    savemat(
        mat_file_name.as_posix(),
        {"k_us": k_us.numpy(), "k_us_cs": k_us_cs.numpy()}
    )
    torch_save(data=k_us, path_to_file=path, file_name="phantom_k_us.pt")
    torch_save(data=k_us_cs, path_to_file=path, file_name="phantom_k_us_cs.pt")



if __name__ == '__main__':
    main()
