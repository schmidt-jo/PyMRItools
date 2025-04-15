"""
estimate a low pass filtered very approximate coil sensitivity map and coil sensitivity
correlations from k_space or image input.
Essentially we just divide the individual coils by a rSoS combined image and smooth the output.
"""
import json
import logging
import pathlib as plib

import torch
import tqdm
import plotly.graph_objects as go

from pymritools.config.processing.coil_sens import CoilSensConfig
from pymritools.config import setup_program_logging, setup_parser
from pymritools.utils import nifti_save, nifti_load, torch_load, root_sum_of_squares, fft, ifft

log_module = logging.getLogger(__name__)


def smap(settings: CoilSensConfig):
    # setup path
    path_out = plib.Path(settings.out_path).absolute()
    log_module.info(f"Setting up output directory: {path_out}")
    if not path_out.exists():
        log_module.info(f"\t\tmkdir")
        path_out.mkdir(exist_ok=True, parents=True)

    path_in = plib.Path(settings.input_data).absolute()
    if ".nii" in path_in.suffixes:
        nii_input = True
        img, data = nifti_load(path_in)
        data = torch.from_numpy(data)
        affine = img.affine
    elif ".pt" in path_in.suffixes:
        nii_input = False
        data = torch_load(path_in)
        affine = torch_load(settings.input_affine)
    else:
        err = f"Suffix not supported ({path_in.suffixes})."
        log_module.error(err)
        raise AttributeError(err)
    dim_channels = settings.coil_dimension

    # check dims
    if not len(data.shape) > 3:
        err = "Input data must be at least 4D."
        log_module.error(err)
        raise AttributeError(err)

    # setup device
    if settings.use_gpu:
        device = torch.device(f"cuda:{settings.gpu_device}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    log_module.info(f"Using device: {device}")

    # allocate
    shape = data.shape
    if len(shape) > 4:
        shape = shape[:-1]
    smap = torch.zeros(shape, dtype=torch.float64)
    # batch z dim
    for idx_z in tqdm.trange(data.shape[2], desc="Batch Processing"):
        batch_data = data[:, :, idx_z].to(device)
        # put in image space
        if not settings.input_in_img_space:
            batch_data = torch.abs(ifft(batch_data, dims=(0, 1)))
        # do rsos
        batch_data_rsos = root_sum_of_squares(batch_data, dim_channel=dim_channels-1)

        # divide
        batch_data_smap = torch.nan_to_num(
            batch_data / batch_data_rsos.unsqueeze(dim_channels-1),
            nan=0.0, posinf=0.0, neginf=0.0
        )

        # if time dim collapse
        if len(batch_data_smap.shape) > 3:
            batch_data_smap = torch.mean(batch_data_smap, dim=-1)

        # smooth
        nx, ny = batch_data_smap.shape[:2]
        bd_smap_k = fft(batch_data_smap, dims=(0, 1))
        nxh = int(nx / 2)
        nyh = int(ny / 2)
        kh = int(settings.smoothing_kernel / 2)
        bd_smap_k[:nxh-kh] = 0
        bd_smap_k[nxh+kh:] = 0
        bd_smap_k[:, :nyh-kh] = 0
        bd_smap_k[:, nyh+kh:] = 0
        bd_smap_smoothed = torch.abs(ifft(bd_smap_k, dims=(0, 1)))

        # fill in
        smap[:, :, idx_z] = bd_smap_smoothed.cpu()

    # want to calculate a covariance matrix of the individual coil sensitivities
    cov = torch.corrcoef(torch.reshape(smap, (-1, smap.shape[-1])).T)

    # plot cov
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(z=cov.cpu().numpy())
    )
    fig_name = path_out.joinpath("smap_cov_plot").with_suffix(".html")
    log_module.info(f"Saving figure to {fig_name}")
    fig.write_html(fig_name.as_posix())

    file_path = path_out.joinpath("smap_cov").with_suffix(".json")
    log_module.info(f"Saving covariance matrix to {file_path}")
    with open(file_path.as_posix(), "w") as j_file:
        json.dump(cov.tolist(), j_file, indent=2)
    # save
    nifti_save(data=smap.to(torch.float32), img_aff=affine, path_to_dir=path_out, file_name="smap")


def main():
    setup_program_logging("Coil Sensitivity Map Estimation")
    parser, args = setup_parser(
        prog_name="Coil Sensitivity Map Estimation",
        dict_config_dataclasses={"settings": CoilSensConfig}
    )

    settings = CoilSensConfig.from_cli(args=args.settings, parser=parser)
    settings.display()

    try:
        smap(settings=settings)
    except Exception as e:
        parser.print_usage()
        log_module.exception(e)



if __name__ == '__main__':
    main()
