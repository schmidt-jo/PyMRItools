import sys
import pathlib as plib
import timeit

import torch
torch.manual_seed(0)
import logging

import plotly.graph_objects as go
import plotly.subplots as psub
import plotly.colors as plc
import polars as pl

from scipy.io import loadmat, savemat
from pymritools.utils import Phantom, fft_to_img, calc_nmse, calc_ssim, calc_psnr
from pymritools.config.basic import setup_program_logging
from pymritools.recon.loraks.utils import (
    check_channel_batch_size_and_batch_channels, prepare_k_space_to_batches, unprepare_batches_to_k_space,
    pad_input, unpad_output
)
from pymritools.recon.loraks.loraks import Loraks
from pymritools.recon.loraks.ac_loraks import AcLoraksOptions

p_tests = plib.Path(__file__).absolute().parent.parent.parent.parent
sys.path.append(p_tests.as_posix())
from tests.utils import get_test_result_output_dir, ResultMode
from tests.experiments.loraks.utils import (
    run_ac_loraks_matlab_script,
    prep_k_space, unprep_k_space, DataType, load_data
)


logger = logging.getLogger(__name__)


def recon_pyloraks(
        k_us: torch.Tensor,
        rank: int, reg_lambda: float, max_num_iter: int):
    # insert slice dim
    k_us = k_us.unsqueeze(2)
    batch_size_channels = k_us.shape[-2]

    timer = timeit.Timer(
        "torch.randn(1000)**2",
        setup="import torch"
    )
    t = timer.repeat(5, number=5)

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {torch.cuda.get_device_name(device)}")

    # batching
    batch_channel_indices = check_channel_batch_size_and_batch_channels(
        k_space_rpsct=k_us, batch_size_channels=batch_size_channels
    )
    k_batched, input_shape = prepare_k_space_to_batches(
        k_space_rpsct=k_us, batch_channel_indices=batch_channel_indices
    )
    # padding
    k_batched, padding = pad_input(k_batched)


    logger.info(f"Setup")
    loraks_opts = AcLoraksOptions(
        regularization_lambda=reg_lambda, max_num_iter=max_num_iter, device=device
    )
    loraks_opts.rank.value = rank

    loraks = Loraks.create(loraks_opts)

    logger.info("Reconstruction")
    k_recon = loraks.reconstruct(k_batched)

    logger.info("Unprepare")
    k_recon = unpad_output(k_space=k_recon, padding=padding)

    logger.info("Unbatch / Reshape")
    k_recon = unprepare_batches_to_k_space(
        k_batched=k_recon, batch_channel_indices=batch_channel_indices, original_shape=input_shape
    )
    return k_recon


def recon_matlab(
        k_us: torch.Tensor, path_matlab: plib.Path,
        rank: int, reg_lambda: float, max_num_iter: int,
        force_compute: bool = False):
    # build matlab data - for joint echo reconstruction, we just combine channel and echo data
    k = k_us.view(*k_us.shape[:2], -1)
    mask = (k.abs() > 1e-12)

    mat_results_file = path_matlab.joinpath("val_output.mat")
    if not mat_results_file.is_file() or force_compute:
        # write to file as input for matlab routine
        matlab_input_file = path_matlab.joinpath("val_input").with_suffix(".mat")
        logger.info(f"save matlab input data: {matlab_input_file}")
        matlab_input_data = {
            "k_data": k.numpy(), "mask": mask.numpy(),
            "rank": rank, "lambda": reg_lambda, "max_num_iter": max_num_iter
        }
        savemat(matlab_input_file, matlab_input_data)

        logger.info("Calling MATLAB routine")
        subproc = run_ac_loraks_matlab_script(
            script_dir=path_matlab,
            script_name_func="recon_validation",
            profile_memory=False
        )
        logger.info(subproc)

    logger.info("Fetching results")
    results = loadmat(mat_results_file.as_posix())

    k_recon = torch.from_numpy(results["k_recon"])
    k_recon = torch.reshape(k_recon, k_us.shape)
    return k_recon


def main():
    # get output dirs
    path_out = plib.Path(
        get_test_result_output_dir("mloraks_recon_validation", mode=ResultMode.EXPERIMENT)
    ).absolute()
    path_data = plib.Path(
        get_test_result_output_dir("mloraks_recon_validation", mode=ResultMode.DATA)
    ).absolute()
    path_matlab = plib.Path(__file__).absolute().parent.joinpath("matlab").absolute()

    # build phantom
    sl_phantom = Phantom.get_shepp_logan(shape=(200, 200), num_coils=8, num_echoes=2)
    k_gt = sl_phantom.get_2d_k_space()

    # subsample
    k_us = sl_phantom.sub_sample_ac_random_lines(acceleration=3, ac_lines=32)
    noise = torch.randn_like(k_us) * k_us.abs().max() / 200
    noise[k_us.abs() < 1e-12] = 0
    k_us += noise

    img_us = fft_to_img(k_us, dims=(0, 1))
    img_gt = fft_to_img(k_gt, dims=(0, 1))
    # reconstruct using PyLORAKS
    k_recon_py = recon_pyloraks(k_us, rank=50, reg_lambda=0.0, max_num_iter=30).squeeze()
    img_recon_py = fft_to_img(k_recon_py, dims=(0, 1))

    k_recon_mat = recon_matlab(k_us, rank=50, reg_lambda=0.0, max_num_iter=20, path_matlab=path_matlab, force_compute=True)
    img_recon_mat = fft_to_img(k_recon_mat, dims=(0, 1))

    # plot
    fig =psub.make_subplots(
        rows=4, cols=5,
        row_titles=[f"Channel: {1+k}" for k in range(4)],
        column_titles=["GT", "US", "P", "M", "ΔPM"],
        horizontal_spacing=0.02, vertical_spacing=0.01
    )
    zmax = img_gt.abs().max().item()*0.6
    for i, d in enumerate([img_gt, img_us, img_recon_py, img_recon_mat, 10 * (img_recon_py - img_recon_mat)]):
        for c in range(4):
            img = d[:, :, c, 0].abs()
            fig.add_trace(
                go.Heatmap(
                    z=img,
                    zmin=0, zmax=zmax,
                    coloraxis="coloraxis"
                ),
                row=1+c, col=i+1
            )
            xaxis = fig.data[-1].xaxis
            fig.update_xaxes(visible=False, row=1+c, col=i+1)
            fig.update_yaxes(visible=False, scaleanchor=xaxis, row=1+c, col=i+1)

    fig.update_layout(
        width=800,
        height=550,
        margin=dict(t=25, b=55, l=65, r=5),
        coloraxis=dict(
            colorscale="Inferno",
            cmin=0,
            cmax=zmax,
            colorbar=dict(
                title=dict(text="Intensity [a.u.]", side="right"),
                len=1,
                thickness=15,

            )
        )
    )
    fn = path_out.joinpath("recon_comp").with_suffix(".html")
    logger.info(f"save html file: {fn}")
    fig.write_html(fn)
    for suff in [".png", ".pdf"]:
        fn = path_out.joinpath("recon_comp").with_suffix(suff)
        logger.info(f"save file: {fn}")
        fig.write_image(fn)

    img_py_stats = torch.reshape(torch.permute(img_recon_py.abs(), (3, 2, 1, 0)), (-1, 200, 200))
    img_mat_stats = torch.reshape(torch.permute(img_recon_mat.abs(), (3, 2, 1, 0)), (-1, 200, 200))
    stats = []
    for i in range(img_py_stats.shape[0]):
        nmse = calc_nmse(img_py_stats[i], img_mat_stats[i])
        ssim = calc_ssim(img_py_stats[i], img_mat_stats[i])
        psnr = calc_psnr(img_py_stats[i], img_mat_stats[i])
        stats.append({"batch": i, "nmse": nmse, "ssim": ssim, "psnr": psnr})
    df = pl.DataFrame(stats)
    logger.info(df)
    logger.info(f"mean: {df.mean()}")
    fn = path_out.joinpath("recon_comp").with_suffix(".json")
    logger.info(f"save file: {fn}")
    df.write_json(fn)


if __name__ == '__main__':
    setup_program_logging(name="MLORAKS validation")
    main()
