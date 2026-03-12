import logging
import pathlib as plib

import numpy as np
import torch
from dataclasses import dataclass
from simple_parsing import field
from scipy.ndimage import gaussian_filter

from pymritools.config import setup_program_logging, setup_parser
from pymritools.config.basic import BaseClass
from pymritools.modeling.espirit.functions import map_estimation
from pymritools.utils import nifti_load, nifti_save, torch_load, torch_save, root_sum_of_squares, ifft_to_k, fft_to_img
from pymritools.processing.denoising.lcpca import extract_noise_mask, extract_noise_stats_from_mask

log_module = logging.getLogger(__name__)


@dataclass
class Settings(BaseClass):
    input_file: str = field(
        alias="-i", default="",
        help="Input file path to combined AFI echo images."
    )
    input_affine: str = field(
        alias="-ia", default="",
        help="Input affine matrix, necessary if input file is .pt to output .nii file."
    )
    flip_angle: float = field(
        alias="-fa", default=60.0,
        help="Flip angle in degrees."
    )
    ratio_tr2_tr1: int = field(
        alias="-n", default=5,
        help="Ratio of TR2 / TR1."
    )
    smoothing_kernel: float = field(
        alias="-s", default=3,
    )
    noise_mask: str = field(
        alias="-m", default="",
        help="Noise mask file. Otherwise We try to deduce this from the input"
    )
    input_in_image_domain: bool = field(
        alias="-iimgd", default=True,
        help="If input is in image domain, otherwise in k-space domain."
    )


def check_cast_signals(signal_e1: float | torch.Tensor, signal_e2: float | torch.Tensor):
    if isinstance(signal_e1, (float, int)):
        signal_e1 = torch.tensor([signal_e1])
    if isinstance(signal_e2, (float, int)):
        signal_e2 = torch.tensor([signal_e2])
    if signal_e1.shape != signal_e2.shape:
        msg = f"afi signal echo images need to have equal shape, but found: {signal_e1.shape} and {signal_e2.shape}"
        raise AttributeError(msg)
    return signal_e1, signal_e2


def variance_r(signal_e1: float | torch.Tensor, signal_e2: float| torch.Tensor,
               sig_se1: float, sig_se2: float) -> torch.Tensor:
    # calculate the variance in the r value / map from the afi signal equation (r = sig1 / sig2)
    signal_e1, signal_e2 = check_cast_signals(signal_e1, signal_e2)
    div = signal_e1.abs()**2
    div[div < 1e-9] = 1
    return torch.divide(
        sig_se1 * signal_e2.abs() + sig_se2 * signal_e1.abs(),
        div)**2


def variance_alpha(
        signal_e1: float | torch.Tensor, signal_e2: float | torch.Tensor,
        sig_se1: float, sig_se2: float, ratio_tr_n: float = 10.0) -> torch.Tensor:
    # calculate the variance in the apparent flip angle, as error propagation through the afi equation
    signal_e1, signal_e2 = check_cast_signals(signal_e1, signal_e2)
    signal_e1[torch.abs(signal_e1) < 1e-9] = 1
    # get r
    ratio_signal_r = torch.divide(
        signal_e2.abs(), signal_e1.abs()
    )

    a = - (ratio_tr_n - 1) * (ratio_tr_n + 1)
    b = (ratio_signal_r - 1) * (ratio_signal_r + 1) * (ratio_tr_n - ratio_signal_r) ** 2
    b[torch.abs(b) < 1e-9] = 1
    tmp = torch.abs(a/b) * variance_r(signal_e1=signal_e1, signal_e2=signal_e2, sig_se1=sig_se1, sig_se2=sig_se2)
    return tmp


def variance_b1(value_sigma_alpha, value_set_alpha: float = 60.0):
    return (18000 / np.pi / value_set_alpha)**2 * value_sigma_alpha


def calculate_error_map(
        b1_data: torch.Tensor, mask: torch.Tensor, flip_angle_set_deg: float,  ratio_tr_n: float | int,
        path_visuals: plib.Path | str = None):
    # get individual echoes
    signal_afi_e1 = b1_data[..., 0]
    signal_afi_e2 = b1_data[..., 1]

    # get noise voxels and noise stats estimate to estimate the mean noise value
    noise_sigma_1, noise_n = extract_noise_stats_from_mask(input_data=signal_afi_e1, mask=mask,
                                                           path_visuals=path_visuals, )
    # get noise voxels and noise stats estimate to estimate the mean noise value
    noise_sigma_2, noise_n_2 = extract_noise_stats_from_mask(input_data=signal_afi_e2, mask=mask,
                                                             path_visuals=path_visuals, )
    # calculate error map
    map_err_alpha = variance_alpha(
        signal_e1=signal_afi_e1, signal_e2=signal_afi_e2,
        sig_se1=noise_sigma_1, sig_se2=noise_sigma_2,
        ratio_tr_n=ratio_tr_n
    )

    return torch.sqrt(variance_b1(value_sigma_alpha=map_err_alpha, value_set_alpha=flip_angle_set_deg))


def calculate_b1(b1_data: torch.Tensor, r_tr21: float, smoothing_kernel: float = 3, ) -> torch.Tensor:
    # calculate ratio
    b1_data[..., 0][torch.abs(b1_data[..., 0]) < 1e-9] = 1
    r = torch.divide(
        b1_data[..., 1].abs(), b1_data[..., 0].abs(),
    )
    rtr = r_tr21 - r
    rtr[torch.abs(rtr) < 1e-9] = 1
    arg = torch.divide(
        r * r_tr21 - 1, rtr
    )
    alpha = torch.arccos(torch.clip(arg, -1, 1))
    alpha *= 180 / np.pi

    # smooth
    alpha_filtered = smooth_b1(alpha=alpha, smoothing_kernel=smoothing_kernel)
    return alpha_filtered


def smooth_b1(alpha: torch.Tensor, smoothing_kernel: float) -> torch.Tensor:
    return torch.from_numpy(gaussian_filter(alpha.numpy(), sigma=smoothing_kernel, axes=(0, 1, 2)))


def processing(settings: Settings):
    # load file
    path_in = plib.Path(settings.input_file).absolute()
    if ".nii" in path_in.suffixes:
        b1_data, b1_img = nifti_load(settings.input_file)
        b1_data = torch.from_numpy(b1_data)
        aff = b1_img.affine
        nii = True
    elif ".pt" in path_in.suffixes:
        b1_data = torch_load(settings.input_file)
        path_aff = plib.Path(settings.input_affine).absolute()
        if path_aff.is_file():
            aff = torch_load(path_aff)
        else:
            log_module.warning("No affine matrix provided, using identity matrix.")
            aff = torch.eye(4)
        nii = False
    else:
        err = f"Suffix not supported ({path_in.suffixes})."
        log_module.error(err)
        raise AttributeError(err)
    # check domain
    if settings.input_in_image_domain:
        b1_k = ifft_to_k(b1_data, dims=(0, 1, 2))
    else:
        log_module.info(
            "Input data is in k-space domain, "
            "converting to image domain using first 3 dimensions as spatial dimensions."
        )
        b1_k = b1_data
        b1_data = fft_to_img(input_data=b1_k, dims=(0, 1, 2))
    # estimate noise voxels
    noise_mask_path = plib.Path(settings.noise_mask).absolute()
    if not noise_mask_path.is_file():
        log_module.info(f"No valid Noise Mask input, deducing from AFI data")
        noise_mask = extract_noise_mask(input_data=b1_data, erode_iter=1)
        nifti_save(data=noise_mask.to(torch.int), img_aff=aff, path_to_dir=settings.out_path, file_name="noise_mask")
    else:
        if ".pt" in noise_mask_path.suffixes:
            noise_mask = torch_load(noise_mask_path)
        else:
            noise_mask, _ = nifti_load(noise_mask_path)
    # calculate b1
    b1 = calculate_b1(
        b1_data=b1_data, r_tr21=settings.ratio_tr2_tr1,
        smoothing_kernel=settings.smoothing_kernel
    ) / settings.flip_angle * 100

    # b1 correction luke
    p = torch.tensor([0.000012295234437, -0.0017655889077654, 0.981394299349869, 3.067045680657626])
    po = torch.arange(p.shape[0]).__reversed__()
    b1_corr = torch.sum(
        p[None, None, None] * torch.pow(b1[..., None], po[None, None, None]),
        dim=-1
    )

    # calculate error map
    b1_err = calculate_error_map(
        b1_data=b1_data, mask=noise_mask, flip_angle_set_deg=settings.flip_angle, path_visuals=settings.out_path,
        ratio_tr_n=settings.ratio_tr2_tr1
    )
    # calculate relative error - we clamp at 200 %
    b1_rel_err = torch.clamp(
        torch.nan_to_num(
            torch.divide(
                b1_err, b1
            ),
            nan=0.0, posinf=0.0, neginf=0.0
        ),
        min=-2, max = 2
    )

    # if torch
    if not nii:
        save_fn = torch_save
    else:
        def save_fn(data, path_to_file, file_name):
            nifti_save(data=data, img_aff=aff, path_to_dir=path_to_file, file_name=file_name)

    names = ["b1_afi", "b1_afi_ref", "b1_rel_err", "b1_afi_corr-poly"]
    for i, d in enumerate([b1, b1_data[..., 0], b1_rel_err, b1_corr]):
        save_fn(data=d, path_to_file=settings.out_path, file_name=names[i])

    if not nii:
        b1_ref = root_sum_of_squares(b1_data[..., 0])
        nifti_save(data=b1_ref, img_aff=aff, path_to_dir=settings.out_path, file_name="b1_afi_ref")

        espirit_maps = map_estimation(
            k_rpsc=b1_k, kernel_size=6, num_ac_lines=10,
            rank_fraction_ac_matrix=0.01, eigenvalue_cutoff=0.05,
            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        sensitivity_maps = espirit_maps[0].abs()
        nifti_save(data=sensitivity_maps, img_aff=aff, path_to_dir=settings.out_path, file_name="coil_sensitivities")

        b1_map = torch.sum(sensitivity_maps * b1.abs(), dim=-1) / torch.sum(sensitivity_maps, dim=-1)
        nifti_save(data=b1_map, img_aff=aff, path_to_dir=settings.out_path, file_name="b1_map_wavg")


def main():
    setup_program_logging(name="AFI B1 creation")
    parser, args = setup_parser(prog_name="AFI B1 creation", dict_config_dataclasses={"settings": Settings})
    # get cli args
    settings = Settings.from_cli(args=args.settings, parser=parser)
    settings.display()
    try:
        processing(settings=settings)
    except Exception as e:
        logging.exception(e)
        parser.print_help()


if __name__ == '__main__':
    main()

