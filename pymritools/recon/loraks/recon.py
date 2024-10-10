import logging
import pathlib as plib

import torch
from torch.onnx.symbolic_opset9 import logit

from pymritools.config.recon import PyLoraksConfig
from pymritools.config import setup_program_logging, setup_parser
from pymritools.utils import torch_load
from pymritools.processing.coil_compression import compress_channels

log_module = logging.getLogger(__name__)


def setup_data(settings: PyLoraksConfig):
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

    k_space, f_indexes, affine = setup_data(settings=settings)

    log_module.info(f"___ Loraks Reconstruction ___")
    log_module.info(f"{settings.flavour}; Radius - {settings.radius}; ")
    log_module.info(
        f"Rank C - {settings.rank_c}; Lambda C - {settings.lambda_c}; Rank S - {settings.rank_s}; Lambda S - {settings.lambda_s}; "
                 f"coil compression - {opts.coil_compression}")

    loraks_name = f"loraks_k_space_recon_r-{opts.radius}"
    if opts.lambda_c > 1e-6:
        loraks_name = f"{loraks_name}_lc-{opts.lambda_c:.3f}_rank-c-{opts.rank_c}"
    if opts.lambda_s > 1e-6:
        loraks_name = f"{loraks_name}_ls-{opts.lambda_s:.3f}_rank-s-{opts.rank_s}"
    loraks_name = loraks_name.replace(".", "p")

    # recon sos and phase coil combination
    solver = algorithm.ACLoraks(
        k_space_input=k_space, mask_indices_input=f_indexes,
        radius=opts.radius,
        rank_c=opts.rank_c, lambda_c=opts.lambda_c,
        rank_s=opts.rank_s, lambda_s=opts.lambda_s,
        max_num_iter=opts.max_num_iter, conv_tol=opts.conv_tol,
        fft_algorithm=False, device=device, fig_path=fig_path,
        channel_batch_size=opts.batch_size
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
    # ToDo implement aspire phase reconstruction

    # switch back
    if opts.read_dir > 0:
        loraks_recon = torch.swapdims(loraks_recon, 0, 1)

    if opts.process_slice:
        loraks_recon = torch.squeeze(loraks_recon)[:, :, None, :]

    logging.info(f"Save k-space reconstruction")
    file_name = out_path.joinpath(loraks_name).with_suffix(".pt")
    logging.info(f"write file: {file_name}")
    torch.save(loraks_recon, file_name.as_posix())

    # save recon as nii for now to look at
    # rSoS k-space-data for looking at it
    loraks_recon_mag = torch.sqrt(
        torch.sum(
            torch.square(
                torch.abs(loraks_recon)
            ),
            dim=-2
        )
    )

    loraks_phase = torch.angle(loraks_recon)
    loraks_phase = torch.mean(loraks_phase, dim=-2)

    loraks_recon_k = loraks_recon_mag * torch.exp(1j * loraks_phase)
    if opts.process_slice:
        loraks_recon = torch.squeeze(loraks_recon)[:, :, None, :]

    utils.save_data(out_path=out_path, name=loraks_name, data=loraks_recon_k, affine=affine)

    logging.info("FFT into image space")
    # fft into real space
    loraks_recon_img = torch.fft.fftshift(
        torch.fft.fft2(
            torch.fft.ifftshift(
                loraks_recon, dim=(0, 1)
            ),
            dim=(0, 1)
        ),
        dim=(0, 1)
    )

    logging.info("rSoS channels")
    # for nii we rSoS combine channels
    loraks_recon_mag = torch.sqrt(
        torch.sum(
            torch.square(
                torch.abs(loraks_recon_img)
            ),
            dim=-2
        )
    )

    loraks_phase = torch.angle(loraks_recon_img)
    loraks_phase = torch.mean(loraks_phase, dim=-2)

    loraks_recon_img = loraks_recon_mag * torch.exp(1j * loraks_phase)

    nii_name = loraks_name.replace("k_space", "image")
    utils.save_data(out_path=out_path, name=nii_name, data=loraks_recon_img, affine=affine)


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
