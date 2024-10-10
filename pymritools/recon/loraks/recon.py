import logging
import pathlib as plib

import torch
import plotly.graph_objects as go

from pymritools.config.recon import PyLoraksConfig
from pymritools.config import setup_program_logging, setup_parser
from pymritools.utils import torch_save, torch_load, root_sum_of_squares, nifti_save, fft
from pymritools.processing.coil_compression import compress_channels
from pymritools.recon.loraks.ac_loraks import ACLoraks

log_module = logging.getLogger(__name__)


def load_data(settings: PyLoraksConfig):
    logging.debug("Load data")
    k_space = torch_load(settings.in_k_space)
    affine = torch_load(settings.in_affine)
    sampling_pattern = torch_load(settings.in_sampling_mask)

    if settings.read_dir > 0:
        k_space = torch.swapdims(k_space, 0, 1)
        sampling_pattern = torch.swapdims(sampling_pattern, 0, 1)

    logging.debug(f"For debug reduce dims")
    if settings.debug:
        # for debugging take one coil
        k_space = k_space[:, :, :, 0, None, :]
        # also take one slice. if not set anyway, we set it
        settings.process_slice = True

    logging.debug(f"Check single slice toggle set")
    if settings.process_slice:
        mid_slice = int(k_space.shape[2] / 2)
        logging.info(f"single slice processing: pick slice {mid_slice + 1}")
        k_space = k_space[:, :, mid_slice, None]

    logging.debug(f"Check sampling pattern shape")
    if sampling_pattern.shape.__len__() < 3:
        # sampling pattern supposed to be x, y, t
        sampling_pattern = sampling_pattern[:, :, None]

    if settings.coil_compression is not None:
        k_space = compress_channels(
            input_k_space=k_space,
            sampling_pattern=sampling_pattern,
            num_compressed_channels=settings.coil_compression,
            use_ac_data=True, use_gcc_along_read=False
        )
    # get shape
    while k_space.shape.__len__() < 5:
        # probably when processing single slice or debugging
        k_space = k_space[..., None]
    read, phase, sli, ch, t = k_space.shape

    # flatten xy dims
    s_xy = sampling_pattern.shape[0] * sampling_pattern.shape[1]
    if abs(s_xy - read * phase) > 1e-3:
        err = f"sampling pattern dimensions do not match input k-space data"
        logging.error(err)
        raise ValueError(err)

    logging.debug(f"Set sampling indices as matrix (fhf)")
    f_indexes = torch.squeeze(
        torch.nonzero(
            torch.reshape(
                sampling_pattern.to(torch.int),
                (s_xy, -1)
            )
        )
    )

    return k_space, f_indexes, affine


def recon(settings: PyLoraksConfig):
    # setup
    log_module.info(f"Set output path: {settings.out_path}")
    path_out = plib.Path(settings.out_path).absolute()

    path_figs = path_out.joinpath("figs/")
    if settings.visualize:
        log_module.info(f"Set figure path for visualizations: {path_figs}")
        path_figs.mkdir(parents=True, exist_ok=True)

    # set up device
    if settings.use_gpu and torch.cuda.is_available():
        logging.info(f"configuring gpu::  cuda:{settings.gpu_device}")
        device = torch.device(f"cuda:{settings.gpu_device}")
    else:
        device = torch.device("cpu")
    torch.manual_seed(0)
    
    # load data
    k_space, f_indexes, affine = load_data(settings=settings)

    log_module.info(f"___ Loraks Reconstruction ___")
    log_module.info(f"{settings.flavour}; Radius - {settings.radius}; ")
    log_module.info(
        f"Rank C - {settings.c_rank}; Lambda C - {settings.c_lambda}; "
        f"Rank S - {settings.s_rank}; Lambda S - {settings.s_lambda}; "
        f"coil compression - {settings.coil_compression}")

    # set up name
    loraks_name = f"loraks_k_space_recon_r-{settings.radius}"
    if settings.c_lambda > 1e-6:
        loraks_name = f"{loraks_name}_lc-{settings.c_lambda:.3f}_rank-c-{settings.c_rank}"
    if settings.s_lambda > 1e-6:
        loraks_name = f"{loraks_name}_ls-{settings.s_lambda:.3f}_rank-s-{settings.s_rank}"
    loraks_name = loraks_name.replace(".", "p")

    # recon sos and phase coil combination
    solver = ACLoraks(
        k_space_input=k_space, mask_indices_input=f_indexes,
        radius=settings.radius,
        rank_c=settings.c_rank, lambda_c=settings.c_lambda,
        rank_s=settings.s_rank, lambda_s=settings.s_lambda,
        max_num_iter=settings.max_num_iter, conv_tol=settings.conv_tol,
        fft_algorithm=False, device=device, fig_path=path_figs,
        channel_batch_size=settings.batch_size
    )
    solver.reconstruct()

    # print stats
    residuals, stats = solver.get_residuals()
    logging.info(f"Minimum residual l2: {stats['norm_res_min']:.3f}")
    logging.info(f"save optimizer loss plot")
    # quick plot of residual sum
    fig = go.Figure()
    for idx_slice in range(solver.dim_slice):
        fig.add_trace(
            go.Scattergl(y=residuals[idx_slice], name=f"slice: {idx_slice}")
        )
    fig_name = solver.fig_path.joinpath(f"{loraks_name}_residuals.html")
    logging.info(f"write file: {fig_name}")
    fig.write_html(fig_name.as_posix())

    # get k-space
    loraks_recon = solver.get_k_space()
    # ToDo implement (aspire) phase reconstruction

    # switch back
    if settings.read_dir > 0:
        loraks_recon = torch.swapdims(loraks_recon, 0, 1)

    if settings.process_slice:
        loraks_recon = torch.squeeze(loraks_recon)[:, :, None, :]

    logging.info(f"Save k-space reconstruction")
    file_name = path_out.joinpath(loraks_name).with_suffix(".pt")
    logging.info(f"write file: {file_name}")
    torch.save(loraks_recon, file_name.as_posix())

    # loraks_phase = torch.angle(loraks_recon)
    # loraks_phase = torch.mean(loraks_phase, dim=-2)
    # loraks_mag = torch.abs(loraks_recon)

    # loraks_recon_k = loraks_mag * torch.exp(1j * loraks_phase)
    if settings.process_slice:
        loraks_recon = torch.squeeze(loraks_recon)[:, :, None, :]

    # save data as tensors, for further usage of whole data
    torch_save(data=loraks_recon, path_to_file=path_out, file_name=f"{loraks_name}_k-space")

    logging.info("FFT into image space")
    # fft into real space
    loraks_recon_img = fft(loraks_recon, img_to_k=False, axes=(0, 1))

    logging.info("rSoS channels")
    # for nii we rSoS combine channels
    loraks_recon_mag = root_sum_of_squares(input_data=loraks_recon_img, dim_channel=-2)

    loraks_phase = torch.angle(loraks_recon_img)
    loraks_phase = torch.mean(loraks_phase, dim=-2)

    nii_name = loraks_name.replace("k_space", "image")
    nifti_save(data=loraks_recon_mag, img_aff=affine, path_to_dir=path_out, file_name=f"{nii_name}_mag")
    nifti_save(data=loraks_phase, img_aff=affine, path_to_dir=path_out, file_name=f"{nii_name}_phase")


def main():
    # setup  logging
    setup_program_logging(name="PyLoraks", level=logging.INFO)
    # setup parser
    parser, args = setup_parser(prog_name="PyLoraks", dict_config_dataclasses={"settings": PyLoraksConfig})
    # get cli args
    settings = PyLoraksConfig.from_cli(args=args.settings, parser=parser)
    settings.display()

    try:
        recon(settings=settings)
    except Exception as e:
        logging.exception(e)
        parser.print_help()


if __name__ == '__main__':
    main()
