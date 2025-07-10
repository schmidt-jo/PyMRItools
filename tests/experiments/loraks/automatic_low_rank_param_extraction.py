import sys
import logging
import pathlib as plib
import pickle
from enum import Enum, auto
from typing import Tuple

import polars as pl

import torch
import plotly.graph_objects as go
import plotly.subplots as psub
import plotly.colors as plc
import tqdm

from pymritools.recon.loraks.operators import Operator, OperatorType
from pymritools.recon.loraks.utils import (
    prepare_k_space_to_batches, pad_input,
    check_channel_batch_size_and_batch_channels, unpad_output, unprepare_batches_to_k_space
)
from pymritools.utils.algorithms import rank_estimator_adaptive_rsvd
from pymritools.utils import fft_to_img, torch_load

p_tests = plib.Path(__file__).absolute().parent.parent.parent.parent
sys.path.append(p_tests.as_posix())
from tests.utils import get_test_result_output_dir, create_phantom, ResultMode

logger = logging.getLogger(__name__)


class DataType(Enum):
    SHEPPLOGAN = auto()
    PHANTOM = auto()
    INVIVO = auto()



#__ misc functionalities
def load_data(data_type: DataType):
    match data_type:
        case DataType.SHEPPLOGAN:
            # set some shapes
            nx, ny, nc, ne = (156, 140, 4, 2)
            k = create_phantom(shape_xyct=(nx, ny, nc, ne), acc=1).unsqueeze(2)
            bet = None
        case DataType.PHANTOM:
            raise NotImplementedError("Phantom data not yet provided")
        case DataType.INVIVO:
            # load input data fully sampled
            path = plib.Path(
                get_test_result_output_dir(f"data", mode=ResultMode.EXPERIMENT)
            )
            k = torch_load(path.joinpath("fs_data_slice.pt"))
            # ensure read dimension is correct
            k = torch.swapdims(k, 0, 1)
            bet = torch_load(path.joinpath("bet.pt"))
        case _:
            raise ValueError(f"Data Type {data_type.name} not supported")
    return k, bet


#__ methods for autograd LORAKS iteration & prep
def prep_k_space(k: torch.Tensor, batch_size_channels: int = -1):
    # we need to prepare the k-space
    batch_channel_idx = check_channel_batch_size_and_batch_channels(
        k_space_rpsct=k, batch_size_channels=batch_size_channels
    )
    prep_k, in_shape = prepare_k_space_to_batches(
        k_space_rpsct=k, batch_channel_indices=batch_channel_idx
    )
    prep_k, padding = pad_input(prep_k, sampling_dims=(-2, -1))
    return prep_k, in_shape, padding, batch_channel_idx


def unprep_k_space(k: torch.Tensor, padding: Tuple[int, int], batch_idx: torch.tensor, input_shape: Tuple):
    k = unpad_output(k_space=k, padding=padding)

    return unprepare_batches_to_k_space(
        k_batched=k, batch_channel_indices=batch_idx, original_shape=input_shape
    )

def estimate_rank(svd_vals: torch.Tensor, area_thr_factor: float = 0.95):
    # calculate area under cumulatively
    total_area = torch.sum(svd_vals)
    cum_area = torch.cumsum(svd_vals, dim=-1)
    # find threshold
    # estimate % of area under curve
    return torch.where(cum_area > area_thr_factor * total_area)[0][0]


def process_svd(k: torch.Tensor, op: Operator, area_thr_factor: float = 0.95):
    # build s-matrix
    matrix = op.forward(k)

    # extract eigenvalue spectrum
    svd_vals = torch.linalg.svdvals(matrix)
    # scale to max 1
    svd_vals /= svd_vals.max()

    th = estimate_rank(svd_vals, area_thr_factor=area_thr_factor)
    return svd_vals, th

def process_rsvd(k: torch.Tensor, op: Operator, area_thr_factor: float = 0.95):
    # build s-matrix
    matrix = op.forward(k)

    th, svd_vals = rank_estimator_adaptive_rsvd(
        matrix=matrix, energy_threshold=area_thr_factor
    )
    return svd_vals, th


def try_arsvd():
    k, k_us = create_phantom(shape_xyct=(192, 168, 4, 2), acc=3, ac_lines=32)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    k_in_fs, _, _, _ = prep_k_space(k.unsqueeze_(2).to(device), batch_size_channels=-1)
    k_in_us, _, _, _ = prep_k_space(k_us.unsqueeze_(2).to(device), batch_size_channels=-1)
    # we want to build the operators from fully sampled and undersampled k-space
    op = Operator(
        k_space_shape=k_in_fs.shape[1:], nb_side_length=5,
        device=device,
        operator_type=OperatorType.S
    )
    # get matrix size
    matrix_size = 5**2 * 4 * 2 * 2
    marix_fs = op.forward(k_in_fs)
    marix_us = op.forward(k_in_us)

    # find rank
    rank_fs, s_vals_fs = rank_estimator_adaptive_rsvd(marix_fs)
    rank_us, s_vals_us = rank_estimator_adaptive_rsvd(marix_us)

    fig = go.Figure()
    for i, s in enumerate([s_vals_fs, s_vals_us]):
        fig.add_trace(
            go.Scatter(
                y=s
            )
        )
        r = [rank_fs, rank_us][i]
        fig.add_trace(
            go.Scatter(x=[r, r], y=[0, s.max()], mode="lines")
        )
    fig.update_layout(
        width=1000,
        height=550,
        margin=dict(t=25, b=55, l=65, r=5)
    )
    fn = "/data/pt_np-jschmidt/code/PyMRItools/scratches/figures/adaptive_rSVD.html"
    logger.info(f"Write file: {fn}")
    fig.write_html(fn)



def automatic_low_rank_param_extraction(data_type: DataType, force_processing: bool = False):
    # get path
    path = plib.Path(get_test_result_output_dir(
        f"automatic_low_rank_param_extraction/{data_type.name}".lower(),
        mode=ResultMode.EXPERIMENT)
    )
    path_out = path.joinpath("result_data").with_suffix(".pkl")
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    nb_size = 5 ** 2

    if path_out.is_file() and not force_processing:
        logger.info(f"Load file: {path_out}")
        with open(path_out, "rb") as f:
            results = pickle.load(f)
    else:
        results = []
        if data_type == DataType.SHEPPLOGAN:
            for i, c in enumerate(torch.arange(4, 36, 3)):
                k, _ = create_phantom(shape_xyct=(192, 168, c, 4), acc=3, ac_lines=32)
                # calculate matrix size
                mat_size = nb_size * c * 4
                # first for fully sampled input
                # prep / batch input k-space
                k_in_fs, _, _, _ = prep_k_space(k.unsqueeze_(2).to(device), batch_size_channels=-1)
                # we want to build the operators from fully sampled and undersampled k-space
                op = Operator(
                    k_space_shape=k_in_fs.shape, nb_side_length=5,
                    device=device,
                    operator_type=OperatorType.S
                )
                # iterate through for fs data
                svd_vals, th = process_rsvd(k_in_fs, op)
                # calculate acc - factor from data
                results.append({
                    "name": "Fully-Sampled", "acc": 1, "calc_acc": 1,
                    "svd_vals": svd_vals.cpu(), "thresh": th.item(),
                    "nc": c, "ne": 4, "mat_size": mat_size.item()
                })
                # iterate for some acceleration factors, i.e. subsampled k-space
                for i, acc in enumerate(torch.linspace(2, 12, 8)):
                    logger.info(f"Processing Acc. {acc:.2f}")
                    _, k_us = create_phantom(shape_xyct=k.shape, acc=acc.item(), ac_lines=32)
                    # calc acc from input
                    calc_acc = torch.prod(torch.tensor(k_us.shape[:2])) / torch.count_nonzero(k_us[..., 0, 0])
                    # prep / batch input k-space
                    k_in, _, _, _ = prep_k_space(k_us.unsqueeze_(2).to(device), batch_size_channels=-1)
                    svd_vals, th = process_rsvd(k_in, op)
                    results.append({
                        "name": f"Sub-Sampled", "acc": acc.item(), "svd_vals": svd_vals.cpu(), "thresh": th.item(),
                        "nc": c.item(), "ne": 4, "mat_size": mat_size.item(), "calc_acc": calc_acc
                    })
            plot_results_svals(results, path)
        else:
            k, bet = load_data(data_type=data_type)
            nx, ny, nc, ne = k.shape
            for acc in torch.linspace(1, 12, 8):
                # create sampling mask
                _, k_us = create_phantom(shape_xyct=(nx, ny, nc, ne), acc=acc, ac_lines=32)
                mask = k_us.abs() > 1e-10
                k_in = k.clone()
                k_in[~mask] = 0
                calc_acc = torch.prod(torch.tensor(mask.shape)) / torch.count_nonzero(mask)
                # reduce available data
                for ic in tqdm.tqdm(torch.arange(4, nc+1, 4)):
                    # some randomization runs
                    for rc in range(3):
                        idx_c = torch.randint(low=0, high=nc, size=(ic,))
                        for ie in torch.arange(2, ne+1, 2):
                            if ic * ie > 32*6:
                                break
                            # some randomization runs
                            for re in range(3):
                                idx_e = torch.randint(low=0, high=ne, size=(ie,))
                                data = k_in[:, :, idx_c, :][..., idx_e]
                                data_in, _, _, _ = prep_k_space(data.unsqueeze_(2), batch_size_channels=-1)
                                op = Operator(
                                    k_space_shape=data_in.shape[1:], nb_side_length=5, device=device, operator_type=OperatorType.S
                                )
                                _, th = process_rsvd(data_in.to(device), op)
                                # calculate matrix size
                                mat_size = nb_size * ic * ie
                                results.append({
                                    "name": f"Sub-Sampled", "acc": acc.item(), "thresh": th, "random_c": rc, "random_e": re,
                                    "nc": ic.item(), "ne": ie.item(), "mat_size": mat_size.item(), "calc_acc": calc_acc.item()
                                })

        logger.info(f"Write file: {path_out}")
        with open(path_out.as_posix(), "wb") as f:
            pickle.dump(results, f)

    plot_results_vs_acc_mat_size(results, path)


def plot_results_svals(results: list, path: plib.Path):
    mat_sizes = []
    acc = []
    for i, r in enumerate(results):
        if r["mat_size"] not in mat_sizes:
            mat_sizes.append(r["mat_size"])
        rac = r["acc"]
        if rac not in acc:
            if torch.is_tensor(rac):
                rac = rac.item()
            acc.append(rac)
    print(acc)
    cmap = plc.sample_colorscale("Turbo", len(acc))
    fig = psub.make_subplots(
        rows=len(mat_sizes), cols=2,
        row_titles=mat_sizes,
        shared_xaxes=True,
        vertical_spacing=0.015, horizontal_spacing=0.1,
        y_title="Singular Values / max(Singular Values)"
    )
    for i, r in enumerate(results):
        row = 1 + mat_sizes.index(r["mat_size"])
        fig.add_trace(
            go.Scatter(
                y=r["svd_vals"],
                name=r["name"],
                legendgroup=i, showlegend=False,
                marker=dict(color=cmap[acc.index(r['acc'])])
            ),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[r["thresh"], r["thresh"]], y=[0, 1],
                opacity=0.6,
                name=f"{r['name']} - Acc. {r['acc']:.2f} - Threshold: {r['thresh']:.1f}",
                legendgroup=i,
                showlegend=False,
                mode="lines", line=dict(color=cmap[acc.index(r['acc'])])
            ),
            row=row, col=1
        )
        if row == len(mat_sizes):
            fig.update_xaxes(title="Singular value number", range=(0, 2*min(mat_sizes)), row=row, col=1)
        else:
            fig.update_xaxes(range=(0, 2*min(mat_sizes)), row=row, col=1)

        fig.add_trace(
            go.Scatter(
                x=[r["acc"]], y=[r["thresh"]],
                name=f"{r['name']} - Acc. {r['acc']:.2f} - Threshold: {r['thresh']:.1f}",
                legendgroup=i,
                showlegend=False,
                mode="markers", line=dict(color=cmap[acc.index(r['acc'])])
            ),
            row=row, col=2
        )
        if row == len(mat_sizes):
            fig.update_xaxes(title="Acceleration", range=(0, max(acc)+1), row=row, col=2)
        else:
            fig.update_xaxes(range=(0, max(acc)+1), row=row, col=2)
        fig.update_yaxes(title="Rank", row=row, col=2)

    fig.update_layout(
        width=1000,
        height=550,
        margin=dict(t=25, b=55, l=65, r=5)
    )
    fn = path.joinpath("rank_vs_sub_sampling").with_suffix(".html")
    logger.info(f"Write file: {fn}")
    fig.write_html(fn)
    for suff in [".png", ".pdf"]:
        fn = fn.with_suffix(suff)
        logger.info(f"Write file: {fn}")
        fig.write_image(fn)


def plot_results_vs_acc_mat_size(results, path):
    for r in results:
        if "svd_vals" in r.keys():
            del r["svd_vals"]
    df = pl.DataFrame(results)
    # print(df)
    # we want to plot the rank against acceleration per matrix size
    matrix_sizes = df["mat_size"].unique().to_list()
    num_mat_sizes = len(matrix_sizes)
    num_cols = 0
    for i, nm in enumerate(matrix_sizes):
        df_tmp = df.filter(pl.col("mat_size") == nm)
        for h, nc in enumerate(df_tmp["nc"].unique()):
            num_cols += 1
    cmap = plc.sample_colorscale("Turbo", num_cols)
    row_titles = [
        "Rank vs Matrix size per number of Channels",
        "Rank vs Matrix Size per Acceleration",
        "Normalized Rank vs Matrix Size per Acceleration"
    ]

    fig = psub.make_subplots(
        rows=3, cols=1,
        y_title="Estimated Rank", x_title="Matrix size",
        shared_xaxes=True,
        vertical_spacing=0.02, horizontal_spacing=0.1,
    )
    ps = 20
    for irc in range(3):
        for ire in range(3):
            df_plot = df.filter(
                (pl.col("random_c") == irc) & (pl.col("random_e") == ire)
            ).drop(["random_c", "random_e"])
            fig.add_trace(
                go.Scatter(
                    x=df_plot["mat_size"] - ps + 2 * ps /9 * (3 * irc + ire), y=df_plot["thresh"],
                    mode="markers",
                    showlegend=False,
                    marker=dict(color=df_plot["nc"], coloraxis="coloraxis"),
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df_plot["mat_size"] - ps + 2*ps/9 * (3 * irc + ire), y=df_plot["thresh"],
                    mode="markers",
                    showlegend=False,
                    marker=dict(color=df_plot["acc"], coloraxis="coloraxis2"),
                ),
                row=2, col=1
            )
            # get df threshold estimation normalized by acc = 1 threshold
            # create lookup reference
            reference_values = (
                df_plot.filter(pl.col("acc") == 1).group_by(
                    ["nc", "mat_size"]
                ).agg(pl.col("thresh").first().alias("ref_thresh"))
            )
            df_norm = (
                df_plot.join(reference_values, on=["nc", "mat_size"], how="left").
                with_columns(
                    (pl.col("thresh") / pl.col("ref_thresh")).alias("normalized_thresh")
                )
                .drop("ref_thresh")
            )

            fig.add_trace(
                go.Scatter(
                    x=df_norm["mat_size"] - ps + 2*ps/9 * (3 * irc + ire), y=df_norm["normalized_thresh"],
                    mode="markers",
                    showlegend=False,
                    marker=dict(color=df_norm["acc"], coloraxis="coloraxis2"),
                ),
                row=3, col=1
    )

    for i, n in enumerate(row_titles):
        fig.add_annotation(
            text=n, x=int(1.08*(max(matrix_sizes))), y=0.5 if i==2 else 0.0, row=1+i, col=1,
            showarrow=False, xanchor="right", yanchor="bottom",
            font=dict(size=14),
        )
    fig.update_layout(
        width = 800,
        height = 420,
        coloraxis=dict(
            colorscale="Inferno",
            cmin=0.0, cmax=df["nc"].max(),
            colorbar=dict(
                x=1.01, y=0.66, len=0.33, thickness=14,
                title=dict(text="Channels", side="right"),
                xanchor="left", yanchor="bottom",
            )
        ),
        coloraxis2=dict(
            colorscale="Inferno",
            cmin=0.0, cmax=df["acc"].max(),
            colorbar=dict(
                x=1.01, y=0.0, len=0.66, thickness=14,
                title=dict(text="Acceleration", side="right"),
                xanchor="left", yanchor="bottom",
            )
        ),
        margin=dict(t=20, b=50, l=65, r=10)
    )
    # fig.update_xaxes(row=2, col=1, title="Matrix Size")
    fn = path.joinpath("mat_size_rank_vs_acc").with_suffix(".html")
    logger.info(f"Write file: {fn}")
    fig.write_html(fn)

    for suff in [".png", ".pdf"]:
        fn = fn.with_suffix(suff)
        logger.info(f"Write file: {fn}")
        fig.write_image(fn)

def rank_matrix_completion(data_type: DataType,):
    # get path
    path = plib.Path(get_test_result_output_dir(
        f"rank_matrix_completion/{data_type.name}".lower(),
        mode=ResultMode.EXPERIMENT)
    )
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


    if data_type == DataType.SHEPPLOGAN:
        _, k = create_phantom(shape_xyct=(80, 50, 8, 2), acc=3, ac_lines=16)
    else:
        raise NotImplementedError(f"data_type {data_type} is not supported")
    k_in, in_shape, padding, batch_channel_idx = prep_k_space(k.unsqueeze_(2).to(device), batch_size_channels=-1)
    batch_channel_idx = batch_channel_idx.to(device)

    op = Operator(
        k_space_shape=k_in.shape[1:], nb_side_length=5,
        device=device,
        operator_type=OperatorType.S
    )

    matrix = op.forward(k_in)

    completed_matrix, losses, ranks = rank_matrix_completion_minimize_rank(
        matrix=matrix, max_num_iter=2500, lr_max=1e-4, lr_min=1e-5, device=device,
    )

    k_recon = op.adjoint(completed_matrix)
    k_recon = unprep_k_space(k_recon.unsqueeze_(0), padding=padding, batch_idx=batch_channel_idx, input_shape=in_shape)

    plot_losses(losses, ranks, path)
    plot_completion(fft_to_img(k.squeeze()[:, :, 0, 0].cpu(), dims=(0, 1)), fft_to_img(k_recon.squeeze()[:, :, 0, 0].cpu(), dims=(0, 1)), path, log=False)


def rank_matrix_completion_minimize_rank(matrix: torch.Tensor, max_num_iter: int = 20, lr_max: float = 1e-1, lr_min: float = 1e-3,
                                         device: torch.device = torch.get_default_device()):
    mask = matrix.abs() < 1e-10
    best_candidate = matrix[mask].clone()
    candidate = torch.zeros_like(best_candidate, requires_grad=True, device=device)
    losses = []
    ranks = []
    lrs = torch.linspace(lr_max, lr_min, max_num_iter)
    for i in tqdm.trange(max_num_iter):
        tmp = matrix.clone()
        tmp[mask] = candidate
        vals, _ = torch.linalg.eigh(tmp @ tmp.mH)
        # vals = torch.linalg.svdvals(tmp)
        loss = torch.sum(torch.square(vals))
        loss.backward()
        with torch.no_grad():
            candidate -= candidate.grad * lrs[i]
            ranks.append(estimate_rank(vals.__reversed__()))
            if i > 1:
                if loss.item() < min(losses):
                    best_candidate = candidate.detach()
            losses.append(loss.item())

    result = matrix.clone()
    result[mask] = best_candidate
    return result, torch.tensor(losses), torch.tensor(ranks)

def plot_completion(matrix, completed_matrix, path, log: bool = True):
    fig = psub.make_subplots(
        rows=1, cols=2
    )
    if log:
        zmax = torch.log(matrix.abs().max()).item()* 1e-1
    else:
        zmax = matrix.abs().max().item()* 0.6
    for i, d in enumerate([matrix, completed_matrix]):
        if log:
            d = torch.log(d.abs())
        else:
            d = d.abs()
        fig.add_trace(
            go.Heatmap(z=d, zmin=0, zmax=zmax, transpose=True),
            row=1, col=1+i
        )

    fig.update_layout(
            width=1000,
            height=550,
            margin=dict(t=25, b=55, l=65, r=5)
        )
    fn = path.joinpath("matrix_completion").with_suffix(".html")
    logger.info(f"Write file: {fn}")
    fig.write_html(fn)
    for suff in [".png", ".pdf"]:
        fn = fn.with_suffix(suff)
        logger.info(f"Write file: {fn}")
        fig.write_image(fn)


def plot_losses(losses, ranks, path):
    fig = psub.make_subplots(rows=2, cols=1, shared_xaxes=True, row_titles=["Loss", "Rank"])
    for i, d in enumerate([losses, ranks]):
        fig.add_trace(
            go.Scatter(
                y=d,
            ),
            row=1+i, col=1
        )
    fig.update_layout(
        width=1000,
        height=550,
        margin=dict(t=25, b=55, l=65, r=5)
    )
    fn = path.joinpath("matrix_completion_loss").with_suffix(".html")
    logger.info(f"Write file: {fn}")
    fig.write_html(fn)
    for suff in [".png", ".pdf"]:
        fn = fn.with_suffix(suff)
        logger.info(f"Write file: {fn}")
        fig.write_image(fn)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    automatic_low_rank_param_extraction(data_type=DataType.INVIVO, force_processing=False)
    # rank_matrix_completion(DataType.SHEPPLOGAN)
    # try_arsvd()
