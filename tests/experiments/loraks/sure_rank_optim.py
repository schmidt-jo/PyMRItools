import sys
import logging
import pathlib as plib
import torch

from pymritools.recon.loraks.ac_loraks import get_ac_matrix
from pymritools.config.basic.basic import setup_program_logging
import pickle

import polars as pl

import numpy as np
from scipy.ndimage import zoom, gaussian_filter1d
from scipy.signal import savgol_filter

import plotly.graph_objects as go
import plotly.subplots as psub
import plotly.colors as plc
import tqdm

import wandb

from pymritools.recon.loraks.ac_loraks import AcLoraksOptions
from pymritools.recon.loraks.loraks import RankReduction, RankReductionMethod, Loraks
from pymritools.recon.loraks.operators import Operator, OperatorType
from pymritools.utils.algorithms import rank_estimator_adaptive_rsvd
from pymritools.utils import fft_to_img, torch_load, Phantom, ifft_to_k, calc_nmse, calc_psnr, root_sum_of_squares, \
    calc_ssim, torch_save

p_tests = plib.Path(__file__).absolute().parent.parent.parent.parent
sys.path.append(p_tests.as_posix())
from tests.utils import get_test_result_output_dir, create_phantom, ResultMode
from tests.experiments.loraks.utils import prep_k_space, DataType, load_data, unprep_k_space

logger = logging.getLogger(__name__)
logger.manager.loggerDict["tests.experiments.loraks.utils"].setLevel(logging.DEBUG)
logger.manager.loggerDict["pymritools.recon.loraks.utils"].setLevel(logging.DEBUG)


def get_fig(img: torch.Tensor):
    fig = psub.make_subplots(
        rows=2, cols=4
    )
    for row in range(2):
        for col in range(4):
            fig.add_trace(
                go.Heatmap(
                    z=img[..., col, row].abs().detach().cpu().numpy(),
                    showscale=row==0 and col==0,
                    colorscale="Inferno"
                ),
                row=row + 1, col=col + 1
            )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def get_trace(op, rank, batch: torch.Tensor, num_mc_samples=50, eps=1e-7):
    # find the trace
    # Trace via Monte Carlo (efficient for large matrices)
    trace_est = 0
    for mc in tqdm.trange(num_mc_samples, desc="Monte Carlo trace estimate"):
        v = torch.randn_like(batch)
        v /= torch.linalg.norm(v)

        pos = sure_operator(r=rank, k_space=v, op=op)
        neg = sure_operator(r=rank, k_space=-v, op=op)
        Jv = (pos - neg) / (2 * eps)  # Jacobian approx
        trace_est += torch.sum(torch.real(v * Jv))
    trace_op = trace_est / num_mc_samples
    return trace_op


def sure_operator(r: int, k_space: torch.Tensor, op: Operator):
    # find the denoised data representation
    m = op.forward(k_space)
    u, s, vh = torch.linalg.svd(m, full_matrices=False)
    s[r:] = 0
    m_lr = u @ torch.diag_embed(s) @ vh
    return op.adjoint(m_lr)


def main():
    # set path
    path = plib.Path(get_test_result_output_dir(
        f"sure_rank_optim".lower(),
        mode=ResultMode.EXPERIMENT)
    )
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Device: {torch.cuda.get_device_name(device)}")

    # k, _, bet = load_data(data_type=DataType.INVIVO)
    # bet = bet[:,:,0].to(torch.bool)
    sl_phantom = Phantom.get_shepp_logan(shape=(192, 168), num_coils=4, num_echoes=2)
    k = sl_phantom.get_2d_k_space()
    noise = torch.randn_like(k) * k.abs().max() / 100
    sigma = torch.std(noise)

    k += noise
    img = fft_to_img(k, dims=(0, 1))
    fig = get_fig(img)

    fn = path.joinpath(f"sure_rank_optim_input").with_suffix(".html")
    logger.info(f"Write fie: {fn}")
    fig.write_html(fn)
    # we want to use only the AC region
    num_ac = 36
    start = (k.shape[1] - num_ac) // 2
    end = (k.shape[1] + num_ac) // 2
    k_ac = k[:, start:end]
    logger.info(f"Prep k-space")
    data_in, in_shape, padding, batch_channel_idx = prep_k_space(k_ac.unsqueeze(2).to(device), batch_size_channels=8)

    op = Operator(
        k_space_shape=data_in.shape[1:], nb_side_length=5, device=device, operator_type=OperatorType.S
    )

    for i, batch in enumerate(data_in[:1]):
        # we first need to estimate noise or get this from pre-scan
        kk = unprep_k_space(batch[None], padding=padding, batch_idx=batch_channel_idx, input_shape=in_shape).squeeze()
        img = fft_to_img(kk, dims=(0, 1))
        fig = get_fig(img)

        fn = path.joinpath(f"sure_batch_in").with_suffix(".html")
        logger.info(f"Write fie: {fn}")
        fig.write_html(fn)
        # we compute the S-matrix for AC only
        m_ac = get_ac_matrix(batch, operator=op)
        # we compute SVD
        u, s, v = torch.linalg.svd(m_ac, full_matrices=False)

        # for a number of rank parameters, we calculate the optimal LORAKS approximation
        num_singular_vals = s.shape[0]
        losses = []
        num_mc_samples = 100
        eps = 1e-6 * torch.linalg.norm(batch)

        for j, r in enumerate(tqdm.trange(num_singular_vals, desc="Iterate rank")):
            # find the denoised data representation
            rank = torch.ones_like(s)
            rank[r:] = 0
            t = u @ torch.diag_embed(rank) @ u.mH
            s_hat = t @ m_ac
            sure_op = op.adjoint(s_hat)
            sl_1 = torch.linalg.norm(sure_op - batch).item()

            trace_op = get_trace(op=op, rank=r, batch=batch, num_mc_samples=num_mc_samples, eps=eps)
            sl_2 = 2*sigma**2 * trace_op.item()
            sure_loss = sl_1 + sl_2
            losses.append({"SURE": sure_loss, "SL1": sl_1, "SL2": sl_2})

            if j % 20 == 0 and j < 101:
                den_k = unprep_k_space(sure_op[None], padding=padding, batch_idx=batch_channel_idx,
                                       input_shape=in_shape).squeeze()
                img = fft_to_img(den_k, dims=(0, 1))
                fig = get_fig(img)
                fn = path.joinpath(f"sure_rank_optim_denoised_img_{j}").with_suffix(".html")
                logger.info(f"Write fie: {fn}")
                fig.write_html(fn)
        fig = go.Figure()
        df = pl.DataFrame(losses)
        for i, name in enumerate(["SL1", "SL2", "SURE"]):
            fig.add_trace(go.Scatter(x=rank_vals, y=df[name].abs(), name=name))
        fig.update_layout(title="Loss vs rank")
        fn = path.joinpath(f"sure_rank_optim_{i}").with_suffix(".html")
        logger.info(f"Write fie: {fn}")
        fig.write_html(fn)



if __name__ == '__main__':
    setup_program_logging("SURE rank optim")
    main()
