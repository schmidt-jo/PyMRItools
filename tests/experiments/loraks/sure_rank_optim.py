import sys
import logging
import pathlib as plib
from enum import auto, Enum

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


class SvdMode(Enum):
    SVD = auto()
    LR_SVD = auto()


class TraceMode(Enum):
    MC = auto()
    HUTCHINSON = auto()


def get_trace(op, rank, batch: torch.Tensor, num_samples=50, eps=1e-7, mode: TraceMode = TraceMode.MC, svd_mode: SvdMode = SvdMode.SVD, lr_q: int = None):
    # find the trace
    match mode:
        case TraceMode.MC:
            trace_op = mc_trace(op=op, rank=rank, batch=batch, num_samples=num_samples, eps=eps, svd_mode=svd_mode, lr_q=lr_q)
        case TraceMode.HUTCHINSON:
            trace_op = hutchinson_trace(k_space=batch, r=rank, op=op, num_samples=num_samples, svd_mode=svd_mode, lr_q=lr_q)
        case _:
            raise ValueError("Unknown trace mode")
    return trace_op


def svd(in_matrix: torch.tensor, mode: SvdMode = SvdMode, lr_q: int = None):
    match mode:
        case SvdMode.SVD:
            u, s, vh = torch.linalg.svd(in_matrix, full_matrices=False)
        case SvdMode.LR_SVD:
            u, s, v = torch.svd_lowrank(in_matrix, q=lr_q, niter=2)
            vh = v.mH
        case _:
            raise ValueError(f"Unknown svd mode ({SvdMode.name})")
    return u, s, vh


def sure_operator(r: int, k_space: torch.Tensor, op: Operator, svd_mode: SvdMode = SvdMode.SVD, lr_q: int = None):
    # find the denoised data representation
    m = op.forward(k_space)
    u, s, vh = svd(m, svd_mode, lr_q=lr_q)
    weighting = torch.ones_like(s)
    weighting[r:] = 0
    m_lr = u @ torch.diag_embed(s * weighting) @ vh
    return op.adjoint(m_lr) / op.count_matrix


def mc_trace(op, rank, batch: torch.Tensor, num_samples=50, eps=1e-7, svd_mode: SvdMode = SvdMode.SVD, lr_q: int = None):
    # Trace via Monte Carlo (efficient for large matrices)
    trace_est = 0
    # for mc in tqdm.trange(num_mc_samples, desc="Monte Carlo trace estimate"):
    for mc in range(num_samples):
        v = torch.randn_like(batch)
        v /= torch.linalg.norm(v)

        pos = sure_operator(r=rank, k_space=batch + eps * v, op=op, svd_mode=svd_mode, lr_q=lr_q)
        neg = sure_operator(r=rank, k_space=batch - eps * v, op=op, svd_mode=svd_mode, lr_q=lr_q)
        Jv = (pos - neg) / (2 * eps)  # Jacobian approx
        trace_est += torch.real(torch.sum(v.flatten().conj() * Jv.flatten()))
        del pos, neg, Jv
        torch.cuda.empty_cache()
    trace_op = trace_est / num_samples
    return trace_op


def hutchinson_trace(k_space: torch.Tensor, r: int, op: Operator, num_samples=100, svd_mode:SvdMode=SvdMode.SVD, lr_q: int = None):
    trace_est = 0
    # k_space.requires_grad_(True)
    # if k_space.grad is not None:
    #     k_space.grad.zero_()

    def p_forward(x):
        return sure_operator(r=r, k_space=x, op=op, svd_mode=svd_mode, lr_q=lr_q)

    def single_sample(v):
        _, jvp = torch.func.jvp(
                p_forward, (k_space,), (v,),
                # create_graph=False
            )
        return torch.real(torch.vdot(v.flatten(), jvp.flatten())).item()

    for _ in tqdm.trange(num_samples, desc="hutchinson trace estimation"):
        v = torch.randn_like(k_space)
        trace_est += single_sample(v)
    torch.cuda.empty_cache()
    return trace_est / num_samples


def main():
    # set path
    path = plib.Path(get_test_result_output_dir(
        f"sure_rank_optim".lower(),
        mode=ResultMode.EXPERIMENT)
    )
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Device: {torch.cuda.get_device_name(device)}")

    k, _, bet = load_data(data_type=DataType.INVIVO)
    bet = bet[:,:,0].to(torch.bool)
    img = fft_to_img(k, dims=(0, 1))
    noise = torch.concatenate([
        img[:30, :30],
        img[:30, -30:],
        img[-30:, :30],
        img[-30:, -30:]
    ]).flatten()
    noise /= np.sqrt(k.shape[0] * k.shape[1])

    # sl_phantom = Phantom.get_shepp_logan(shape=(192, 168), num_coils=4, num_echoes=2)
    # k = sl_phantom.get_2d_k_space()
    # noise = torch.randn_like(k) * k.abs().max() / 100
    sigma = torch.std(noise)

    img = fft_to_img(k, dims=(0, 1))
    fig = get_fig(img)

    fn = path.joinpath(f"sure_rank_optim_input").with_suffix(".html")
    logger.info(f"Write file: {fn}")
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
    count_matrix = op.count_matrix
    svd_mode = SvdMode.LR_SVD

    losses = []
    for i, batch in enumerate(data_in[:1]):
        den_k = unprep_k_space(batch[None].expand(data_in.shape), padding=padding, batch_idx=batch_channel_idx,
                               input_shape=in_shape).squeeze()
        img = fft_to_img(den_k, dims=(0, 1))
        fig = get_fig(img)
        fn = path.joinpath(f"sure_rank_optim_input_img-batch-{i+1}").with_suffix(".html")
        logger.info(f"Write fie: {fn}")
        fig.write_html(fn)
        # we compute the S-matrix for AC only
        m_ac = get_ac_matrix(batch, operator=op)
        # we compute SVD
        u, s, v = svd(m_ac, mode=SvdMode.SVD)

        # for a number of rank parameters, we calculate the optimal LORAKS approximation
        num_samples = 100
        eps = 1e-3 * torch.linalg.norm(batch)
        ranks = torch.arange(20, min(m_ac.shape) // 10, 25)
        m = torch.prod(torch.tensor(batch.shape)).item()

        bar = tqdm.tqdm(ranks, desc="Iterate rank")
        for j, r in enumerate(bar):
            # find the denoised data representation
            rank = torch.ones_like(s)
            rank[r:] = 0
            t = u @ torch.diag_embed(rank) @ u.mH
            s_hat = t @ m_ac
            sure_op = op.adjoint(s_hat) / count_matrix
            sl_1 = torch.sum(torch.abs(sure_op - batch)**2) / m
            lr_q = max(r + 10, min(m_ac.shape) // 10)
            bar.postfix = f"compute hutchinson"
            trace_op_h = get_trace(
                op=op, rank=r, batch=batch, num_samples=num_samples, eps=eps, mode=TraceMode.HUTCHINSON, svd_mode=svd_mode, lr_q=lr_q
            ) / m
            torch.cuda.empty_cache()
            # bar.postfix = f"compute mc"
            # trace_op_mc = get_trace(op=op, rank=r, batch=batch, num_samples=num_samples, eps=eps, mode=TraceMode.MC, svd_mode=svd_mode, lr_q=lr_q)
            # torch.cuda.empty_cache()
            # sl_2 = 2*sigma**2 * trace_op_mc
            sl_2_h = 2*sigma**2 * trace_op_h
            sure_loss = sl_1 + sl_2_h
            losses.append({
                "batch": i,
                "SURE": sure_loss,
                "loss_fidelity": sl_1,
                # "loss_trace": sl_2,
                "loss_trace_hutchinson": sl_2_h})

            if j % 2 == 0 and j < 20:
                den_k = unprep_k_space(sure_op[None].expand(data_in.shape), padding=padding, batch_idx=batch_channel_idx,
                                       input_shape=in_shape).squeeze()
                img = fft_to_img(den_k, dims=(0, 1))
                fig = get_fig(img)
                fn = path.joinpath(f"sure_rank_optim_denoised_img_batch-{i+1}_rank-{r}").with_suffix(".html")
                logger.info(f"Write fie: {fn}")
                fig.write_html(fn)
        fig = go.Figure()
        df = pl.DataFrame(losses)
        for i, name in enumerate(["SURE", "loss_fidelity", "loss_trace_hutchinson"]):
            fig.add_trace(go.Scatter(x=ranks.detach().cpu(), y=df[name].abs(), name=name))
        fig.update_layout(title="Loss vs rank")
        fn = path.joinpath(f"sure_rank_optim").with_suffix(".html")
        logger.info(f"Write fie: {fn}")
        fig.write_html(fn)



if __name__ == '__main__':
    setup_program_logging("SURE rank optim")
    main()
