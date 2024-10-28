import logging
import pathlib as plib

import torch

from pymritools.config import setup_program_logging, setup_parser
from pymritools.config.processing import DenoiseSettingsMPK
from pymritools.utils import torch_load, torch_save, nifti_save, fft, root_sum_of_squares
from pymritools.processing.denoising.mppca_k_filter.functions import matched_filter_noise_removal


log_module = logging.getLogger(__name__)


def main(settings: DenoiseSettingsMPK):
    # check path
    path_out = plib.Path(settings.out_path).absolute()
    log_module.info(f"Setting up output path {path_out}")
    path_out.mkdir(exist_ok=True, parents=True)
    # figures
    if settings.visualize:
        path_figs = path_out.joinpath("figs/")
        log_module.info(f"Setting up visualization path {path_figs}")
        path_figs.mkdir(exist_ok=True, parents=True)
    # load data
    k_space = torch_load(settings.in_k_space)
    noise_scans = torch_load(settings.in_noise_scans)
    affine = torch_load(settings.in_affine)
    sampling_mask = torch_load(settings.in_sampling_mask) if settings.in_sampling_mask is not None else None
    # assuming k-space dims [x, y, z, ch, t], cast if not the case
    while k_space.shape.__len__() < 5:
        k_space.unsqueeze(-1)

    # deduce sampling mask
    if sampling_mask is None:
        sampling_mask = torch.sum(torch.abs(k_space), dim=-1, keepdim=True) > 1e-10
    # expand to k_space size
    while sampling_mask.shape.__len__() < k_space.shape.__len__():
        # we assume here channels and slices are missing, this is a specific usecase not true for e.g. 3D sequences
        l = sampling_mask.shape.__len__()
        exp = [-1] * (l + 1)
        exp[-2] = k_space.shape[l-1]
        sampling_mask = sampling_mask.unsqueeze(-2).expand(exp)
    # deduce read direction, assuming fully sampled read
    read_dir = -1
    if torch.sum(torch.abs(sampling_mask[:, int(sampling_mask.shape[1] / 2), 0, 0, 0].to(torch.int)), dim=0) < sampling_mask.shape[0]:
        read_dir = 0
    if torch.sum(torch.abs(sampling_mask[int(sampling_mask.shape[0] / 2), :, 0, 0, 0].to(torch.int)), dim=0) < sampling_mask.shape[1]:
        if read_dir == 0:
            msg = f"found k - space to be undersampled in x and y direction. Can choose either direction for processing."
            log_module.info(msg)
        else:
            read_dir = 1
    # move read dim to front
    input_filter = torch.movedim(k_space, read_dir, 0)
    nr, np, ns, nch, nt = input_filter.shape
    sampling_mask = torch.movedim(sampling_mask, read_dir, 0)

    # reduce undersampled dimension
    input_filter = torch.reshape(input_filter[sampling_mask], (nr, -1, ns, nch, nt))
    filtered_k = torch.zeros_like(input_filter)

    # pass to function
    filtered_k[sampling_mask] = matched_filter_noise_removal(
        noise_data_n_ch_samp=noise_scans, k_space_lines_read_ph_sli_ch_t=input_filter,
        settings=settings
    ).fatten()

    # save
    if settings.visualize:
        # do quick naive fft rsos recon
        img = fft(filtered_k, axes=(0, 1))
        img = root_sum_of_squares(img, dim_channel=-2)
        nifti_save(img, img_aff=affine, path_to_dir=settings.out_path, file_name="filt_naive_rsos_recon")

    torch_save(filtered_k, path_to_file=path_out, file_name='filtered_k-space')



if __name__ == '__main__':
    # setup logging
    setup_program_logging(name="MP distribution k-space filter denoising", level=logging.INFO)
    # setup parser
    parser, args = setup_parser(
        prog_name="MP distribution k-space filter denoising",
        dict_config_dataclasses={"settings": DenoiseSettingsMPK}
    )
    # get config
    settings = DenoiseSettingsMPK.from_cli(args=args.settings, parser=parser)
    settings.display()

    try:
        main(settings=settings)
    except Exception as e:
        parser.print_help()
        logging.exception(e)
        exit(-1)
