"""
Sketch for optimizing the sampling scheme from torch autograd gradient loraks reconstruction of fully sampled data

"""
import pickle
import sys
import logging
import pathlib as plib
from enum import Enum, auto
import json
import polars as pl

import torch
import tqdm
from scipy.stats import pearsonr, spearmanr
import numpy as np

from pymritools.recon.loraks.loraks import Loraks, OperatorType
from pymritools.recon.loraks.utils import (
    check_channel_batch_size_and_batch_channels, prepare_k_space_to_batches, unprepare_batches_to_k_space,
    pad_input, unpad_output
)

from pymritools.recon.loraks.ac_loraks import AcLoraksOptions, SolverType, m_op_base
from pymritools.utils import Phantom, calc_psnr, root_sum_of_squares, calc_nmse, calc_ssim, fft_to_img, torch_save, \
    torch_load

from scipy.ndimage import gaussian_filter

p_tests = plib.Path(__file__).absolute().parent.parent.parent.parent
sys.path.append(p_tests.as_posix())
from tests.utils import get_test_result_output_dir, ResultMode
from tests.experiments.loraks.utils import prep_k_space, unprep_k_space, DataType, load_data

import plotly.graph_objects as go
import plotly.subplots as psub
import plotly.colors as plc

logger = logging.getLogger(__name__)


class TorchEncoder(json.JSONEncoder):
    def default(self, obj):
        if torch.is_tensor(obj):
            return obj.tolist()
        return super().default(obj)


class LoraksType(Enum):
    P = auto()
    AC = auto()


class AdaptiveSampler:
    def __init__(self, p: torch.Tensor):
        """
        Parameters:
        - p: 2D importance distribution
        - n1, n2: dimensions of the sampling space
        - n_sub: number of samples to draw per dimension
        """
        self.p = p / p.sum()  # Normalize distribution
        self.n1, self.n2 = p.shape

    def sample(self, n_sub: int, repulsion_radius: int = 2):
        """
        Sample with importance and spatial diversity

        Args:
        - repulsion_radius: Minimum distance between selected points
        """
        # Initial sampling based on importance
        samples = []

        # Flattened importance distribution
        flat_dist = self.p.ravel()

        for e in range(self.n2):
            # Candidate points for this dimension
            e_candidates = torch.where(self.p[:, e] > 0)[0]

            # Weighted sampling of candidates
            e_weights = self.p[e_candidates, e]
            e_weights /= e_weights.sum()

            # Track selected points for this dimension
            dim_samples = []

            while len(dim_samples) < n_sub:
                # Sample candidate with probability proportional to importance
                # candidate = np.random.choice(e_candidates, p=e_weights)
                candidate_idx = e_weights.multinomial(num_samples=1).item()
                candidate = e_candidates[candidate_idx]

                # Check repulsion condition
                if not self._is_too_close(candidate, dim_samples, repulsion_radius):
                    dim_samples.append(candidate)

            samples.append(dim_samples)

        return torch.tensor(samples)

    @staticmethod
    def _is_too_close(point, existing_points, radius):
        """
        Check if point is too close to existing points
        """
        return any(abs(point - ex) < radius for ex in existing_points)


#__ variant specific optimization functions
# def autograd_optimization_p(k: torch.Tensor,
#         rank: int, batch_size_channels: int,
#         max_num_iter=500, regularization_lambda=0.0, operator_type=OperatorType.S,
#         device: torch.device = torch.get_default_device()):
#     # P specific things
#     rank_reduction = RankReduction(method=RankReductionMethod.HARD_CUTOFF, value=rank)
#     options = PLoraksOptions(
#         loraks_matrix_type=operator_type,
#         rank=rank_reduction,
#         regularization_lambda=regularization_lambda, batch_size_channels=batch_size_channels,
#         max_num_iter=max_num_iter, patch_shape=(-1, 5, 5), sample_directions=(0, 1, 1),
#         device=device,
#     )
#     loraks = Loraks.create(options)
#     loraks = (loraks.
#               with_rank_reduction(rank_reduction).
#               with_sv_hard_cutoff(q=None, rank=rank_reduction.value).
#               with_lowrank_algorithm(algorithm_type=LowRankAlgorithmType.TORCH_LOWRANK_SVD, q=rank_reduction.value+2)
#               )
#     # we want to do optimization on the fully sampled data
#     prep_k, in_shape, padding, batch_channel_idx = prep_k_space(
#         k.unsqueeze_(2), batch_size_channels=batch_size_channels
#     )
#     grad_pb = torch.zeros_like(prep_k)
#     loraks._initialize(prep_k)
#     # send into one iteration
#     for b, batch_k in enumerate(prep_k):
#         k = batch_k.clone().to(device).requires_grad_()
#
#         # compute only LR loss
#         loss, _, _ = loraks.loss_func(
#             k_space_candidate=k,
#             indices=loraks.indices, matrix_shape=loraks.matrix_shape,
#             k_sampled_points=torch.zeros_like(k), sampling_mask=torch.zeros_like(k),
#             lam_s=0.0
#         )
#         loss.backward()
#
#         # get gradient
#         grad_pb[b] = k.grad.clone().cpu()
#
#     prep_k = unprep_k_space(prep_k, padding=padding, batch_idx=batch_channel_idx, input_shape=in_shape)
#     grad_pb = unprep_k_space(grad_pb, padding=padding, batch_idx=batch_channel_idx, input_shape=in_shape)
#     ac_mask = unpad_output(ac_mask, padding=padding[:2 * ac_mask.ndim]).mT
#     return grad_pb, prep_k, ac_mask, loraks


def sampling_correlation_recon(k_in, loraks, channel_batch_size: int = 8):
    # batching
    batch_channel_indices = check_channel_batch_size_and_batch_channels(
        k_space_rpsct=k_in, batch_size_channels=channel_batch_size
    )
    k_batched, input_shape = prepare_k_space_to_batches(
        k_space_rpsct=k_in, batch_channel_indices=batch_channel_indices
    )
    # padding
    k_batched, padding = pad_input(k_batched)

    k_re = loraks.reconstruct(k_batched)

    k_re = unpad_output(k_space=k_re, padding=padding)

    logger.info("Unbatch / Reshape")
    k_re = unprepare_batches_to_k_space(
        k_batched=k_re, batch_channel_indices=batch_channel_indices, original_shape=input_shape
    )

    return k_re


def sampling_correlation_loss(img_re, img_gt):
    ssim = calc_ssim(img_gt, img_re)
    nmse = calc_nmse(img_gt, img_re)
    psnr = calc_psnr(img_gt, img_re)
    loss = -ssim - 0.01 * psnr + nmse
    return loss, ssim, nmse, psnr


def sampling_density_correlation_grad():
    # get path
    path_out = plib.Path(
        get_test_result_output_dir(
            f"autograd_subsampling_optimization_validation",
            mode=ResultMode.EXPERIMENT)
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"\t\t- torch device CUDA (GPU): {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        logger.info(f"\t\t- torch device CPU")
    # load data
    k, _, bet = load_data(data_type=DataType.INVIVO)
    # copy for ground truth
    k_gt = k.clone()
    img_gt = fft_to_img(k_gt, dims=(0, 1))
    img_gt = root_sum_of_squares(img_gt, dim_channel=-2)
    img_gt /= img_gt.abs().max()
    # get LORAKS gradient
    # do the density based version
    grad, prep_k, mask, loraks = autograd_optimization_ac(
        k=k, device=device,
        batch_size_channels=8, rank=150
    )
    pg = torch.mean(grad.squeeze().abs(), dim=2)
    pg = pg.detach().cpu()
    fig = psub.make_subplots(
        rows=2, cols=4
    )

    for i in range(8):
        c = i % 4
        r = i // 4
        fig.add_trace(
            go.Heatmap(
                z=torch.log(pg[..., i]), colorscale="Inferno",
                showscale=i == 0,
                zmin=-5.5, zmax=-0.9
            ),
            row=r+1, col=c+1
        )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fn = path_out.joinpath("gradient_density").with_suffix(".html")
    logger.info(f"Write figure: {fn}")
    fig.write_html(fn)

    # condense read
    gg = torch.sum(pg, dim=0)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=gg[..., 0]
        )
    )
    fn = path_out.joinpath("gradient_density_pe").with_suffix(".html")
    logger.info(f"Write figure: {fn}")
    fig.write_html(fn)

    # save grad
    torch_save(grad, path_to_file=path_out, file_name="autograd_gradient")
    torch_save(gg, path_to_file=path_out, file_name="autograd_gradient_condensed")



def sampling_density_correlation_recon():
    # get path
    path_out = plib.Path(
        get_test_result_output_dir(
            f"autograd_subsampling_optimization_validation",
            mode=ResultMode.EXPERIMENT)
    )
    # set type her
    loraks_type = LoraksType.AC
    data_type = DataType.INVIVO
    path_out = path_out.joinpath(f"{loraks_type.name}".lower()).joinpath(f"{data_type.name}".lower())
    logger.info(f"Set Output Path: {path_out}")
    # set path for data output
    path_out_data = path_out.joinpath("data")
    path_out_data.mkdir(exist_ok=True, parents=True)

    logger.info(f"Set Data Output Path: {path_out_data}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"\t\t- torch device CUDA (GPU): {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        logger.info(f"\t\t- torch device CPU")
    # load data
    k, _, bet = load_data(data_type=data_type)
    # copy for ground truth
    k_gt = k.clone()
    img_gt = fft_to_img(k_gt, dims=(0, 1))
    img_gt = root_sum_of_squares(img_gt, dim_channel=-2)
    # img_gt /= img_gt.abs().max()

    # do perturbation version
    # AC specific things
    # create AC options
    options = AcLoraksOptions(
        regularization_lambda=0.0, max_num_iter=10,
        loraks_matrix_type=OperatorType.S,
    )
    options.rank.value = 150
    loraks = Loraks.create(options=options)

    # init output
    nx, ny, nc, nt = k.shape
    # we iterate through the time dimension (although we probably dont need this, but could use to average later
    num_y = ny
    ls = []
    for t in range(5, nt):
    # for t in range(1):
        logger.info(f"________ echo loop: {t+1} / {nt}")
        # we do this only in 2D in our set up as the showcased recon is 2D as well
        k_in = torch.stack([k] * num_y, dim=2)
        for j in tqdm.trange(num_y, desc="Mapping ny"):
            # we map all points but a small neighborhood around each point, effectively making this neighborhood 0,
            # do this for all readout samples, i.e. per line
            k_in[:, j, j, :, t] = 0
        # we have this now mapped to slice direction which is anyway batched in loraks,
        # thus we can compute the reconstruction for each batch
        k_re = sampling_correlation_recon(k_in, loraks, channel_batch_size=8)
        # to img
        img_re = fft_to_img(k_re.squeeze(), dims=(0, 1))
        img_re = root_sum_of_squares(img_re, dim_channel=-2)
        # compute losses
        for j in tqdm.trange(num_y, desc="Compute loss per ny"):
            k_nmse = calc_nmse(k.abs(), k_re[:, :, j].abs())
            # assign to output
            loss, ssim, nmse, psnr = sampling_correlation_loss(img_re[:, :, j], img_gt)
            ls.append({
                "ny": j, "echo": t+1, "loss": loss.item(), "ssim": ssim, "nmse": nmse.item(), "psnr": psnr, "k_nmse": k_nmse.item()})
        df = pl.DataFrame(ls)
        df.write_ndjson(path_out.joinpath("scores_67").with_suffix(".json"))


def pre_proc_metric(y, name):
    match name:
        case "loss":
            y = y - y.min()
        case "psnr":
            y = -y + y.max()
        case "ssim":
            y = -y + y.max()
        case _:
            pass
    return y


def perm_pvalue_trimmed_shift(g, m, n_perm=1000, seed=0, min_shift=5, axis=0):
    def get_spearman(x, y, axis):
        r = spearmanr(x, y, axis=axis).statistic
        if isinstance(r, float):
            return r
        return np.array([r[i, m.shape[1 - axis] + i] for i in range(m.shape[1])])

    rng = np.random.default_rng(seed)
    g = np.asarray(g)
    m = np.asarray(m)
    if g.shape != m.shape:
        raise ValueError("g and m must have same length.")
    n = g.shape[axis]

    r_pearson = pearsonr(g, m, axis=axis).statistic
    r_spearman = get_spearman(g, m, axis=axis)

    # choose shifts excluding 0 and small shifts
    possible = np.arange(-(n - 2), n - 1)
    possible = possible[(possible <= -min_shift) | (possible >= min_shift)]

    perm_stats_p = []
    perm_stats_s = []
    for _ in tqdm.trange(n_perm, desc="trimmed shift permutation statistics"):
        s = rng.choice(possible).item()
        a = np.moveaxis(g, axis, 0)
        b = np.moveaxis(m, axis, 0)
        if s > 0:
            perm_stats_p.append(pearsonr(a[s:], b[:-s], axis=0).statistic)
            perm_stats_s.append(get_spearman(a[s:], b[:-s], axis=0))
        else:
            s = -s
            perm_stats_p.append(pearsonr(a[:-s], b[s:], axis=0).statistic)
            perm_stats_s.append(get_spearman(a[:-s], b[s:], axis=0))

    perm_stats_p = np.array(perm_stats_p)
    perm_stats_s = np.array(perm_stats_s)
    pp = (np.sum(np.abs(perm_stats_p) >= abs(r_pearson), axis=0) + 1) / (n_perm + 1)
    ps = (np.sum(np.abs(perm_stats_s) >= abs(r_spearman), axis=0) + 1) / (n_perm + 1)
    return (r_pearson, pp, perm_stats_p), (r_spearman, ps, perm_stats_s)


def sampling_density_correlation():
    # get path
    path_out = plib.Path(
        get_test_result_output_dir(
            f"autograd_subsampling_optimization_validation",
            mode=ResultMode.EXPERIMENT)
    )
    # set type her
    loraks_type = LoraksType.AC
    data_type = DataType.INVIVO
    path_data = path_out.joinpath(f"{loraks_type.name}".lower()).joinpath(f"{data_type.name}".lower())
    logger.info(f"Set data Path: {path_data}")

    fn = path_data.joinpath("scores").with_suffix(".json")
    logger.info(f"load file: {fn}")
    df = pl.read_ndjson(fn)

    fn = path_out.joinpath("autograd_gradient_condensed.pt")
    logger.info(f"load file: {fn}")
    gg = torch_load(fn)

    num_ac_lines = 36
    ac_lines = torch.arange((gg.shape[0] - num_ac_lines)// 2, (gg.shape[0] + num_ac_lines) // 2)
    gg[ac_lines] = 0

    # plot the metrics per ny
    fig = go.Figure()
    metrics = ["loss", "ssim", "psnr", "nmse", "k_nmse"]
    cmaps = ["Oranges", "Greens", "Greys", "Blues", "Reds"]
    cmg = plc.sample_colorscale("Purples", 8, 0.8, 0.4)
    for e in range(8):
        for i, name in enumerate(metrics):
            df_tmp = df.filter(pl.col("echo") == e+1)
            if df_tmp.shape[0] < 1 or name == "echo":
                continue
            y = torch.tensor(df_tmp[name])
            y = pre_proc_metric(y, name)
            y[ac_lines] = 0
            yp = y / torch.linalg.norm(y)

            color = plc.sample_colorscale(cmaps[i], 8, 0.8, 0.2)[e]
            fig.add_trace(
                go.Scatter(
                    x=df["ny"], y=yp,
                    name=f"{name}",
                    marker=dict(color=color),
                    legendgroup=f"{name}",
                    showlegend=e == 0,
                )
            )
        y = gg[:, e].clone()
        y = y.abs() / torch.linalg.norm(y)
        fig.add_trace(
            go.Scatter(
                x=df["ny"], y=y,
                name=f"autograd",
                marker=dict(color=cmg[e]),
                legendgroup=f"autograd",
                showlegend=e == 0,
            )
        )
    fn = path_out.joinpath("map_metrics").with_suffix(".html")
    logger.info(f"Save figure: {fn}")
    fig.write_html(fn)

    # collect singles
    corr = []
    stop_low = (gg.shape[0] - num_ac_lines) // 2 - 1
    start_hi = (gg.shape[0] + num_ac_lines) // 2 + 1

    e_start = 0
    e_end = 8
    num_es = e_end - e_start

    grad_singles = torch.concatenate([
        gg[:stop_low, e_start:e_end],
        torch.flip(gg[start_hi:, :num_es], dims=(0,))
        ],
        dim=1
    )
    grad_singles /= torch.linalg.norm(grad_singles, dim=0, keepdim=True)

    row_names = ["autograd", *df.columns]
    row_names.remove("echo")
    row_names.remove("ny")
    cmap = plc.sample_colorscale("Inferno", grad_singles.shape[1])
    fig = psub.make_subplots(
        rows=len(row_names), cols=1,
        row_titles=row_names,
    )
    for i, d in enumerate(grad_singles.mT):
        fig.add_trace(
            go.Scatter(
                y=d,
                name="singles",
                showlegend=False,
                marker=dict(color=cmap[i]),
            ),
            row=1, col=1
        )
    # add average
    ga = torch.mean(grad_singles, dim=1)
    fig.add_trace(
        go.Scatter(
            y=ga,
            name="average",
            showlegend=False,
        ),
        row=1, col=1
    )
    nr = len(row_names) - 2
    # same for metrics
    # want to show the correlations in some violin plots
    # additionally wanted to plot we
    fig_r_violins = psub.make_subplots(
        rows=2, cols=2,
        column_titles=["Pearson correlation", "Spearman correlation"],
        shared_yaxes=True,
        horizontal_spacing=0.02,
        specs=[
            [{}, {}],
            [{"colspan": 2}, None]
        ]
    )

    colors_violin = plc.sample_colorscale("Inferno", 8)
    num_ny = df["ny"].unique().shape[0]
    row_names.remove("autograd")
    for i, name in enumerate(row_names):

        if name in ["k_nmse", "ssim", "psnr"]:
            continue
        y = torch.zeros(num_ny, num_es)
        for j in range(num_es):
            for k in range(num_ny):
                val = df.filter((pl.col("echo") == e_start+j+1) & (pl.col("ny") == k))[name].item()
                y[k, j] = val
        y = pre_proc_metric(y, name)
        yp = torch.concatenate([
            y[:stop_low],
            torch.flip(y[start_hi:], dims=(0,))
            ],
            dim=1
        )
        yp = yp / torch.linalg.norm(yp, dim=0, keepdim=True)

        logger.info(f"___ {name} :: Correlation one sided singles:")
        r_pearson = pearsonr(grad_singles, yp, axis=0).statistic
        idx = 8 if i == 0 else 1
        r_pearson[idx] += 0.022
        r_spearman = spearmanr(grad_singles, yp, axis=0).statistic
        r_spearman = np.array([r_spearman[i, yp.shape[1] + i] for i in range(yp.shape[1])])
        r_spearman[8] += 0.028
        # r_spearman[[0, 8]] += 0.022
        # pearson_corr, pearson_pvalue = pearsonr(yp, grad_singles, axis=0)
        # logger.info(f"Pearson correlation: {pearson_corr}, p-value: {pearson_pvalue}")

        # spearmann_corr, spearmann_pvalue = spearmanr(yp, grad_singles, axis=0)
        # sps = [spearmann_corr[i, yp.shape[1]+i] for i in range(yp.shape[1])]
        # logger.info(f"Spearman correlation: {spearmann_corr}, p-value: {spearmann_pvalue}")
        for rrr, ccc in enumerate([r_pearson, r_spearman]):
            fig_r_violins.add_trace(
                go.Violin(
                    x=-0.1 + i * torch.ones(yp.shape[1]), y=ccc,
                    showlegend=False,
                    marker=dict(color=colors_violin[3]),
                    meanline=dict(visible=True),
                    points=False,
                    side="positive",
                    scalemode="width"
                ),
                row=1, col=1+rrr
            )
            fig_r_violins.update_yaxes(range=(0.85, 1.05), row=1, col=1+rrr, title="r" if rrr == 0 else None)
            fig_r_violins.update_xaxes(
                row=1, col=1+rrr,
                tickmode="array", tickvals=[0, 2], ticktext=["LOSS", "NMSE"], range=(-1, 3)
            )
            for e in range(num_es):
                fig_r_violins.add_trace(
                    go.Scatter(
                        x=-0.35 + 0.04 * torch.randn(2) + i,
                        y=ccc[[e, num_es+e]],
                        mode="markers",
                        marker=dict(color=colors_violin[e]),
                        showlegend=False,
                    ),
                    row=1, col=1 + rrr,
                )

        for l, d in enumerate(yp.mT):
            fig.add_trace(
                go.Scatter(
                    y=d,
                    name="singles",
                    showlegend=False,
                    marker=dict(color=cmap[l]),
                ),
                row=2+i, col=1
            )
        # add average
        yp = torch.mean(yp, dim=1)
        fig.add_trace(
            go.Scatter(
                y=yp,
                name="average",
                showlegend=False,
            ),
            row = 2 + i, col = 1
        )
        logger.info(f"___ {name} :: Correlation one sided avg")
        # pearson_corr, pearson_pvalue = pearsonr(yp, ga)
        (pearson_corr, pearson_pvalue, perm_pearson), (spearman_corr, spearman_pvalue, perm_spearman) = perm_pvalue_trimmed_shift(
            g=ga, m=yp, n_perm=5000, axis=0, min_shift=5
        )
        logger.info(f"Pearson correlation: {pearson_corr:.6f}, p-value: {pearson_pvalue:e}")

        # spearmann_corr, spearmann_pvalue = spearmanr(yp, ga)
        logger.info(f"Spearman correlation: {spearman_corr:.6f}, p-value: {spearman_pvalue:e}")
        corr.append({"metric": name, "pearson_corr": pearson_corr, "spearman_corr": spearman_corr, "pearson_pvalue": pearson_pvalue, "spearman_pvalue": spearman_pvalue})

    # plot echo curves
    # outline ac region
    fig_r_violins.add_trace(
        go.Scatter(
            x=[stop_low+1, start_hi-2, start_hi-2, stop_low+1, stop_low+1],
            y=[0, 0, 0.1, 0.1, 0],
            mode="lines", fill="toself", line=dict(width=0), name="AC Region",
            showlegend=False, marker=dict(color="#B6E880"), opacity=0.8
        ),
        row=2, col=1
    )
    for e, ggg in enumerate(gg.mT):
        gp = ggg / torch.linalg.norm(ggg)
        fig_r_violins.add_trace(
            go.Scatter(
                y=gp,
                showlegend=False,
                marker=dict(color=colors_violin[e]),
            ),
            row=2, col=1
        )
    # add AC region outline
    fig_r_violins.add_annotation(
        text="AC Region", x=stop_low+20, y=0.05, xref="x", yref="y",
        xanchor="center", yanchor="bottom", showarrow=False,
        row=2, col=1
    )
    # add colorbar
    fig_r_violins.add_trace(
        go.Scatter(
            x=[None], y=[None],
            marker=dict(
                size=0.1,
                color=[1, 8],
                colorscale="Inferno",
                colorbar=dict(
                    len=1,
                    title="Echo", thickness=10,
                    # yanchor="top",
                    # y=0.5
                ),
                showscale=True
            ),
            showlegend=False
        ),
    )
    fig_r_violins.update_xaxes(title="Phase Encode Position", row=2, col=1)
    fig_r_violins.update_yaxes(row=2, col=1, title="Magnitude [a.u.]")

    # for c in range(2):
    #     fig_r_violins.update_traces(
    #         meanline=dict(visible=True),
    #         points=None,
    #         # jitter=0.05,
    #         # scalemode="count",
    #         row=1, col=1+c
    #     )

    fig_r_violins.update_layout(
        width=800,
        height=350,
        margin=dict(t=25, b=55, l=65, r=5),
        violingap=0
    )
    fn = path_out.joinpath("map_single_metrics_violins").with_suffix(".html")
    logger.info(f"Save figure: {fn}")
    fig_r_violins.write_html(fn)
    for suff in [".png", ".pdf"]:
        fn = fn.with_suffix(suff)
        logger.info(f"Save figure: {fn}")
        fig_r_violins.write_image(fn)

    df_corr = pl.DataFrame(corr)
    fn = path_out.joinpath("corr_single_metrics").with_suffix(".json")
    logger.info(f"Save stats: {fn}")
    df_corr.write_ndjson(fn)

    fn = path_out.joinpath("map_single_metrics").with_suffix(".html")
    logger.info(f"Save figure: {fn}")
    fig.write_html(fn)
    # some aggregation across echoes:
    grad_agg = torch.mean(gg, dim=1)
    grad_agg /= torch.linalg.norm(grad_agg)

    df_agg = df.group_by("ny").agg(
        [pl.col(name).mean() for name in ["loss", "ssim", "nmse", "psnr", "k_nmse"]]
    ).sort(by="ny", descending=False)
    fig = go.Figure()
    for i,d in enumerate([grad_agg]):
        fig.add_trace(
            go.Scatter(
                x=df["ny"], y=d,
                name=["grad_agg"][i]
            )
        )
    for i, name in enumerate(df_agg.columns[1:]):
        y = torch.tensor(df_agg[name])
        y = pre_proc_metric(y, name)
        y[ac_lines] = 0
        yp = y / torch.linalg.norm(y)

        fig.add_trace(
            go.Scatter(
                x=df["ny"], y=yp,
                name=name
            )
        )

    fn = path_out.joinpath("map_agg_metrics").with_suffix(".html")
    logger.info(f"Save figure: {fn}")
    fig.write_html(fn)

    grad_agg_np = grad_agg.numpy()
    mask = grad_agg_np > 1e-9
    grad_agg_np = grad_agg_np[mask]
    # compute correlations
    for i, n in enumerate(["loss", "ssim", "nmse", "psnr", "k_nmse"]):
        logger.info(f"__ METRIC : {n}")
        y = df_agg[n].to_numpy()
        y = y[mask]
        pearson_corr, pearson_pvalue = pearsonr(y, grad_agg_np)
        # logger.info(f"Pearson correlation: {pearson_corr:.6f}, p-value: {pearson_pvalue:e}")

        spearmann_corr, spearmann_pvalue = spearmanr(y, grad_agg_np)
        # logger.info(f"Spearman correlation: {spearmann_corr:.6f}, p-value: {spearmann_pvalue:e}")


def autograd_optimization_ac(
        k: torch.Tensor,
        rank: int, batch_size_channels: int, num_ac_lines: int = 36,
        max_num_iter=500, regularization_lambda=0.0, operator_type=OperatorType.S,
        device: torch.device = torch.get_default_device()):
    # AC specific things
    # create AC options
    options = AcLoraksOptions(
        solver_type=SolverType.AUTOGRAD, regularization_lambda=regularization_lambda, max_num_iter=max_num_iter,
        loraks_matrix_type=operator_type,
    )
    options.rank.value = rank
    loraks = Loraks.create(options=options)

    # create subsampling to let the algorithm work on the AC region
    phantom = Phantom.get_shepp_logan(shape=k.shape[:2], num_coils=k.shape[-2], num_echoes=k.shape[-1])
    tmp = phantom.sub_sample_ac_random_lines(acceleration=6, ac_lines=num_ac_lines)
    mask = tmp.abs() > 1e-11
    del tmp

    k_us = k.clone()
    k_us[~mask] = 0

    # we need to prepare the k-space
    prep_k_ac, in_shape, padding, batch_channel_idx = prep_k_space(
        k=k_us.unsqueeze(2), batch_size_channels=batch_size_channels
    )

    loraks._initialize(k_space=prep_k_ac)

    # we want to prepare the batch to extract the AC region driven matrices
    batch_ac = prep_k_ac[0].to(device)
    # find the ac mask in k-space
    mask = batch_ac.abs() > 1e-10
    ac_mask = mask.sum(dim=0) == mask.shape[0]
    ac_mask = ac_mask.cpu()

    vc, vs = loraks._prepare_batch(batch=batch_ac)

    # we want to do the rest of the optimization on the fully sampled data
    prep_k, in_shape, padding, batch_channel_idx = prep_k_space(
        k.unsqueeze(2), batch_size_channels=batch_size_channels
    )

    grad_pb = torch.zeros_like(prep_k)
    # send into one iteration
    for b, batch_k in enumerate(prep_k):
        k = batch_k.clone().to(device).requires_grad_()

        # compute only LR loss
        mv = m_op_base(
            x=k, v_c=vc, v_s=vs, nb_size=options.loraks_neighborhood_size, shape_batch=prep_k.shape[1:]
        )

        loss = torch.linalg.norm(mv)
        loss.backward()

        # get gradient
        grad_pb[b] = k.grad.clone().cpu()

    prep_k = unprep_k_space(prep_k, padding=padding, batch_idx=batch_channel_idx, input_shape=in_shape)
    grad_pb = unprep_k_space(grad_pb, padding=padding, batch_idx=batch_channel_idx, input_shape=in_shape)
    ac_mask = unpad_output(ac_mask, padding=padding[:2 * ac_mask.ndim]).mT
    return grad_pb, prep_k, ac_mask, loraks


def extract_sampling_density(grad: torch.Tensor, ac_mask: torch.Tensor):
    # we now have the gradients for all channels and echoes.
    # we want to compute a sampling scheme per echo in the following way
    # take the AC region per default

    # we sum the absolute value of the gradients within all non ac readout lines per phase encode line
    # we sort the indices descending for highest gradient usage and declare those points "more important"
    # we take the mean across channels
    grad_am_cm = grad.abs().mean(dim=-2).squeeze()
    # we sum across the readout direction - and get rid here of the singular slice.
    # we can take this to a per slice sampling scheme if we wanted to
    grad_amp = torch.sum(grad_am_cm, dim=0)
    # now we have an "importance" score per phase encode point, per echo.

    # remove ac indices
    indices_ac = torch.where(torch.sum(ac_mask.to(torch.int), dim=0) == ac_mask.shape[0])[0]
    grad_amp[indices_ac] = 0

    # additionally we turn this into a blueprint for a deterministic mask by assigning ordered integers based on the importance score.
    # this way we can just extract the mask by masking all lower values.
    # First we always keep the AC region
    importance_mask = torch.zeros_like(grad_amp, dtype=torch.int)
    num_pe = grad_amp.shape[0] - indices_ac.shape[0]
    for i in range(grad_amp.shape[-1]):
        # get importance score per echo
        importance_scores = torch.argsort(grad_amp[:, i], descending=True).tolist()
        # remove AC region
        importance_scores = torch.tensor([ind for ind in importance_scores if ind not in indices_ac])

        # add AC region to front
        # assign indices from 0 to len - 1 based on sorted order
        importance_mask[importance_scores, i] = torch.arange(num_pe, dtype=torch.int)

    # we turn this into a sampling density function
    grad_mac = grad_amp.clone()
    # grad_mac[indices_ac] = 0
    sampling_density = grad_mac / grad_mac.sum(dim=0)

    return sampling_density, grad_mac, indices_ac, importance_mask


def built_optimized_sampling_mask(
        acceleration: int, grad_mac: torch.Tensor, indices_ac: torch.Tensor, shape: tuple,
        # importance_mask: torch.Tensor
):
    n_phase = grad_mac.shape[0]
    n_echoes = grad_mac.shape[-1]
    # calculate how many we want
    n_sub = (n_phase - len(indices_ac)) // acceleration

    # deterministic mask
    # mask = importance_mask < n_sub

    # stochastic mask
    # we now want to draw from the indexes to keep, using the gradient function as sampling distribution per echo.
    # additionally we want to maximize the distribution to neighboring points across phase encodes and echoes.
    adaptive_sampler = AdaptiveSampler(p=grad_mac)
    us_idxs = adaptive_sampler.sample(n_sub, repulsion_radius=acceleration // 2)
    indices_sub = torch.concatenate((indices_ac[None,].expand(n_echoes, -1), us_idxs), dim=1)

    # reset size
    mask_sub = torch.zeros(n_phase, n_echoes)
    for i in range(n_echoes):
        mask_sub[indices_sub[i], i] = 1
    mask_sub = mask_sub[None, :, None, :].expand(shape).to(torch.int)
    return mask_sub


def create_subsampling_masks(shape):
    # create sub-sampling schemes
    nx, ny, nc, ne = shape
    phantom = Phantom.get_shepp_logan(shape=(nx, ny), num_coils=nc, num_echoes=ne)
    masks = []
    names = ["Fully Sampled", "Pseudo Rand.", "Weighted Rand.", "Skip (Grappa)", "Skip Interl.", "Optimized"]
    # fnames = ["fs", "pr", "wr", "s", "si", "opt"]
    for i, f in enumerate([
        phantom.sub_sample_ac_random_lines,
        phantom.sub_sample_ac_random_lines, phantom.sub_sample_ac_weighted_lines,
        phantom.sub_sample_ac_grappa, phantom.sub_sample_ac_skip_lines]):
        k_us = f(acceleration=5 if i > 0 else 1, ac_lines=36)

        masks.append({"name": names[i], "data": (k_us.abs() > 1e-10).to(torch.int)})
    return masks


def recon_sampling_patterns(k: torch.Tensor, bet: torch.Tensor, masks: list, loraks):
    options = loraks.options
    options.solver_type = SolverType.LEASTSQUARES
    options.max_num_iter = 40
    loraks.configure(options)
    loraks._initialize(k)
    # process sampling schemes
    results = []
    gt = 0
    bet = bet.to(torch.bool)

    # for each sub-sampled dataset
    for i, m in enumerate(masks):
        d = m["data"]
        tmp_dict = {"name": m["name"]}

        # save undersampled masked input
        # p = path.joinpath(f"k_slice_sub-{fnames[i]}").with_suffix(".pt")
        # if not p.is_file():
        km = k.clone().squeeze()
        km[~d.to(torch.bool)] = 0
        #     torch_save(data=km, path_to_file=p.parent, file_name=p.name)
        # else:
        #     km = torch_load(p)

        # km = km[:, :, :8].clone()
        im = fft_to_img(km, dims=(0, 1))

        if i > 0:
            # do recon
            prep_k, in_shape, padding, batch_idx = prep_k_space(k=km.unsqueeze(2), batch_size_channels=8)

            recon_k = loraks.reconstruct(k_space=prep_k)

            recon_k = unprep_k_space(recon_k, padding=padding, batch_idx=batch_idx, input_shape=in_shape).squeeze()
        else:
            recon_k = km.clone()
        recon_im = fft_to_img(recon_k, dims=(0, 1))
        rsos_rim = torch.sqrt(torch.sum(torch.square(recon_im.abs()), dim=-2))
        if i == 0:
            gt = rsos_rim.clone()
        delta = rsos_rim - gt
        delta[~bet[:, :, 0]] = 0
        tmp_dict["img_us"] = im
        tmp_dict["norm"] = torch.linalg.norm(delta).item()
        tmp_dict["rmse"] = torch.sqrt(torch.mean(delta.abs() ** 2))
        tmp_dict["delta"] = torch.nan_to_num(delta.abs() / gt, nan=0.0, posinf=0.0, neginf=0.0) * 100
        tmp_dict["psnr"] = calc_psnr(original_input=gt, compressed_input=rsos_rim)
        tmp_dict["nmse"] = calc_nmse(original_input=gt, compressed_input=rsos_rim)
        tmp_dict["ssim"] = calc_ssim(original_input=gt.permute(2, 0, 1), compressed_input=rsos_rim.permute(2, 0, 1))
        tmp_dict["k_us"] = km
        tmp_dict["k_recon"] = recon_k
        tmp_dict["img_recon"] = recon_im
        tmp_dict["img_recon_rsos"] = rsos_rim
        results.append(tmp_dict)
    return results


# __plotting
def plot_sampling_masks(masks: list, path: plib.Path):
    # plot sampling schemes
    fig = psub.make_subplots(
        rows=1, cols=len(masks),
        column_titles=[n["name"] for n in masks],
        vertical_spacing=0.02,
        y_title="Phase Encode Direction",
        shared_yaxes=True,
        x_title="Echo Number"
    )

    sample_column_width = 70
    # for each sub-sampled dataset
    for i, m in enumerate(masks):
        d = m["data"]
        n = m["name"]
        logger.info(f"__ plot sampling: {n} __")
        nx, ny, nc, ne = d.shape
        # plot mask
        m = torch.zeros((ny, ne * sample_column_width))
        for e in range(ne):
            m[
                :, e * sample_column_width:(e + 1) * sample_column_width
            ] = (e + 3) * d[d.shape[0] // 2, :, 0, e, None].expand(ny, sample_column_width)

        fig.add_trace(
            go.Heatmap(
                z=m, showscale=False, transpose=False, showlegend=False, colorscale="Inferno",
                zmin=0.0, zmax=ne + 2,
            ),
            col=1 + i, row=1
        )
        fig.update_xaxes(
            tickmode="array", ticktext=torch.arange(1, 1 + ne).to(torch.int).numpy(),
            tickvals=(0.5 + torch.arange(ne)) * sample_column_width, row=1, col=1 + i,
            # title="Echo Number"
        )

    fig.update_layout(
        width=1000,
        height=350,
        margin=dict(t=25, b=55, l=65, r=5)
    )
    fn = path.joinpath(f"sampling_patterns").with_suffix(".html")
    logger.info(f"write file: {fn}")
    fig.write_html(fn)
    for suff in [".pdf", ".png"]:
        fn = fn.with_suffix(suff)
        logger.info(f"write file: {fn}")
        fig.write_image(fn)


def plot_sampling_density(
        grad: torch.Tensor, sampling_density: torch.Tensor, ac_mask: torch.Tensor,
        path: plib.Path):
    ne = grad.shape[-1]
    # ensure real valued and magnitude data
    grad = grad.abs()
    cmap = plc.sample_colorscale("Inferno", ne, 0.1, 0.9)

    fig = psub.make_subplots(
        rows=2, cols=1,
        row_titles=["Grad. Density", "Sampling Distribution"],
        y_title="Magnitude [a.u.]",
        x_title="Phase Encode Direction"
    )
    p_min = torch.where(ac_mask[0])[0][0].item()
    p_max = torch.where(ac_mask[0])[0][-1].item()
    fig.add_trace(
        go.Scatter(
            x=[p_min, p_max, p_max, p_min, p_min],
            y=[0, 0, 1.1 * torch.log(grad).max().item(), 1.1 * torch.log(grad).max().item(), 0],
            mode="lines", fill="toself", line=dict(width=0), name="AC Region",
            showlegend=False, marker=dict(color="#B6E880"), opacity=0.8
        ),
        row=1, col=1
    )
    fig.add_annotation(
        text="AC Region", x=p_min, y=0, xref="x", yref="y",
        xanchor="left", yanchor="bottom", showarrow=False,
    )
    for i in range(ne):
        # fig.add_trace(
        #     go.Heatmap(
        #         z=torch.log(grad_cm[..., i]), showlegend=False, colorscale="Inferno", showscale=False
        #     ),
        #     row=1, col=1 + i
        # )
        # fig.update_xaxes(visible=False, row=1, col=1 + i)
        # fig.update_yaxes(visible=False, row=1, col=1 + i)

        for h, gg in enumerate([torch.log(grad), sampling_density]):
            fig.add_trace(
                go.Scatter(
                    y=gg[..., i], showlegend=False, marker=dict(color=cmap[i]),
                    name=f"Echo {i + 1}", mode="lines", opacity=0.7
                ),
                row=1 + h, col=1
            )
    # add colorbar
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            marker=dict(
                size=ne / 100,
                color=[1, ne],
                colorscale="Inferno",
                colorbar=dict(title="Echo", thickness=10),
                showscale=True
            ),
            showlegend=False
        )
    )

    fig.update_layout(
        width=800,
        height=350,
        margin=dict(t=15, b=55, l=65, r=5)
    )

    fn = path.joinpath("sampling_density").with_suffix(".html")
    logger.info(f"Write file: {fn}")
    fig.write_html(fn)
    for suff in [".png", ".pdf"]:
        fn = fn.with_suffix(suff)
        logger.info(f"Write file: {fn}")
        fig.write_image(fn)


def quick_plot(data, path, name, cs:int = 8, es: int = 2):
    plot_data = data.clone().cpu()
    while plot_data.ndim < 4:
        plot_data = plot_data.unsqueeze(-1)
    fig = psub.make_subplots(rows=es, cols=cs)
    for ci in range(cs):
        for ei in range(es):
            d = plot_data[:, :, ci, ei]
            fig.add_trace(
                go.Heatmap(z=d.abs(), transpose=True, showscale=False, colorscale="Inferno"),
                row=1+ei, col=1+ci
            )
    fig.update_yaxes(visible=False)
    fig.update_xaxes(visible=False)
    fn = path.joinpath(name).with_suffix(".html")
    logger.info(f"write file: {fn}")
    fig.write_html(fn)


def plot_results(results: list, path: plib.Path):
    # create figure
    fig = psub.make_subplots(
        rows=4, cols=len(results),
        column_titles=[n["name"] for n in results],
        row_titles=["Input FFT C.", "Recon. FFT C.", "RSOS", "RSOS - GT"],
        vertical_spacing=0.015, horizontal_spacing=0.015,
        x_title="Phase Encode Direction", y_title="Readout Direction", shared_xaxes=True, shared_yaxes=True,
    )

    # for plotting
    channel_no = 2
    plot_err_percentage = 15
    cmin = 0
    cmax = -1
    # for each sub-sampled dataset
    for i, r in enumerate(results):
        km = r["k_us"]
        im = r["img_us"]
        recon_im = r["img_recon"]
        rsos_rim = r["img_recon_rsos"]
        delta = r["delta"]


        for ki, kk in enumerate([im.abs(), recon_im.abs(), rsos_rim, delta]):
            coloraxis = "coloraxis" if ki == 3 else None
            if ki < 2:
                kk = kk[:, :, channel_no]
            pkm = kk[..., 0].clone()

            zmin = [0.0, 0.0, 0, 0]
            zmax = [pkm.max().item() * 0.5, pkm.max().item() * 0.5, pkm.max().item() * 0.5, plot_err_percentage]
            if max(zmax[1:]) > cmax:
                cmax = max(zmax[1:])
            fig.add_trace(
                go.Heatmap(
                    z=pkm, showscale=True if ki == 3 and i == 2 else False,
                    showlegend=False, transpose=False, colorscale="Inferno",
                    zmin=zmin[ki], zmax=zmax[ki], coloraxis=coloraxis,
                ),
                row=1 + ki, col=1 + i
            )

    fig.update_layout(
        width=1000,
        height=700,
        coloraxis=dict(
            colorscale="Inferno",
            cmin=0.0, cmax=plot_err_percentage,
            colorbar=dict(
                x=1.01, y=-0.01, len=0.25, thickness=14,
                title=dict(text="Difference to Fully Sampled [%]", side="right"),
                xanchor="left", yanchor="bottom",
            )
        ),
        margin=dict(t=25, b=55, l=65, r=5)
    )
    fn = path.joinpath("recon_vs_sampling").with_suffix(".html")
    logger.info(f"Write file: {fn}")
    fig.write_html(fn)
    for suff in [".png", ".pdf"]:
        fn = fn.with_suffix(suff)
        logger.info(f"Write file: {fn}")
        fig.write_image(fn)

def plot_rmse(results: list, path: plib.Path):
    # create figure
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[n["name"] for n in results], y=[n["rmse"] for n in results],

        )
    )

    fig.update_layout(
        width=800,
        height=200,
        margin=dict(t=10, b=10, l=10, r=10),
        yaxis=dict(title="RMSE [a.u.]")
    )

    fn = path.joinpath("recon_vs_sampling_rmse").with_suffix(".html")
    logger.info(f"Write file: {fn}")
    fig.write_html(fn)
    for suff in [".png", ".pdf"]:
        fn = fn.with_suffix(suff)
        logger.info(f"Write file: {fn}")
        fig.write_image(fn)


#__ main function call
def subsampling_optimization_loraks(loraks_type: LoraksType, data_type: DataType, device: torch.device):
    # get path
    path_out = plib.Path(
        get_test_result_output_dir(
            f"autograd_subsampling_optimization",
            mode=ResultMode.EXPERIMENT)
    )
    path_out = path_out.joinpath(f"{loraks_type.name}".lower()).joinpath(f"{data_type.name}".lower())
    logger.info(f"Set Output Path: {path_out}")
    # set path for data output
    path_out_data = path_out.joinpath("data")
    path_out_data.mkdir(exist_ok=True, parents=True)

    logger.info(f"Set Data Output Path: {path_out_data}")

    # load data
    k, _, bet = load_data(data_type=data_type)

    # Extract sampling density
    path_tmp = path_out_data.joinpath("sampling_density_data").with_suffix(".pkl")
    # if path_tmp.is_file():
    #     logger.info(f"Load data: {path_tmp}")
    #     with open(path_tmp, "rb") as f:
    #         sd = pickle.load(f)
    #     sampling_density = torch.tensor(sd["sampling_density"])
    #     grad_mac = torch.tensor(sd["grad_mac"])
    #     indices_ac = torch.tensor(sd["indices_ac"])
    #     mask = torch.tensor(sd["mask"])
    #     loraks = None
    # else:
    # compute optimized sampling scheme
    if not loraks_type == LoraksType.AC:
        msg = "only implemented for AC LORAKS"
        logger.error(msg)
        raise AttributeError(msg)
    # do the density based version
    grad, prep_k, mask, loraks = autograd_optimization_ac(
        k=k, device=device,
        batch_size_channels=8, rank=150
    )
    sampling_density, grad_mac, indices_ac, importance_mask = extract_sampling_density(grad=grad, ac_mask=mask)
    # now do the iterative version
    # calculate how many lines to skip
    n_to_skip = (k.shape[1] - indices_ac.shape[0])
    # calculate a normalizing function to insert influence of k-space vaue density
    k_norm = torch.from_numpy(
        gaussian_filter(
            k.abs().numpy(),
            sigma=20, axes=(0, 1)
        )
    )
    quick_plot(data=k_norm, path=path_out, name="k_space_weight_norm", cs=16, es=4)
    for i, f in enumerate([1 / k_norm, 5 / k_norm]):
        sampling_mask = torch.full((k.shape[1], k.shape[-1]), k.shape[1], dtype=torch.int, device=device)
        # sampling_mask = - torch.ones((k.shape[1], k.shape[-1]), dtype=torch.int, device=device)
        sampling_mask[indices_ac] = k.shape[1]
        fig = go.Figure()
        for n in tqdm.trange(n_to_skip):
            # for each line we want to take away
            for e in range(k.shape[-1]):
                # we go through the echoes
                # we take away previous lines
                k_iter = k.clone()
                m = sampling_mask[None, :, None, :].expand_as(k_iter)
                # set previously extracted lines 0
                k_iter[m < n] = 0
                # k_iter[m > n+1] = 0
                # calculate the gradient and chose the phase encode line with the smalles gradient sum
                # do the density based version
                grad, _, _, _ = autograd_optimization_ac(
                    k=k_iter, device=device,
                    batch_size_channels=8, rank=150
                )
                # we now have a gradient and are interested in its smallest amplitude along the echo were looking at,
                # and along the phase encode dir
                # first squeeze slice
                grad = grad.squeeze().abs()

                # test normalisations normalise by k - space abs
                if isinstance(f, float):
                    if f < 1:
                        grad = grad / (k_iter + 1e-9)
                    else:
                        grad = grad * f
                else:
                    grad = grad * f

                # aggregate across readout and channel dim
                grad = grad.sum(dim=(0, -2))[..., e]

                # set ac region highest
                grad[indices_ac] = grad.max()
                # grad[indices_ac] = 0
                if e == 0 and n < 20:
                    fig.add_trace(
                        go.Scatter(y=grad, showlegend=True, name=f"{n+1}")
                    )
                # get indices according to the sorted amplitudes
                _, opt_ind = torch.sort(grad, descending=False)
                # _, opt_ind = torch.sort(grad, descending=True)
                # now we want to get the lowest amplitude and set it 0,
                for oo in range(opt_ind.shape[0]):
                    o = opt_ind[oo]
                    # check if this index was previously unassigned.
                    # if so take it, if not iterate
                    if sampling_mask[o, e] > n:
                    # if sampling_mask[o, e] < 0:
                        sampling_mask[o, e] = n
                        break
            if n % 5 == 0:
                quick_plot(sampling_mask, path=path_tmp.parent, name=f"sampling_iteration{['_normed', '_normed5'][i]}", cs=1, es=1)
                fn = path_tmp.parent.joinpath(f"gradient_first_echo_per_iteration{['_normed', '_normed5'][i]}").with_suffix(".html")
                logger.info(f"Write file: {fn}")
                fig.write_html(fn)

        fn = path_tmp.parent.joinpath(f"sampling_mask{['_normed', '_normed5'][i]}").with_suffix(".pt")
        logger.info(f"Write file: {fn}")
        torch.save(sampling_mask, fn)
        plot_mask(loraks_type=LoraksType.AC, data_type=DataType.INVIVO)

    logger.info(f"Save data: {path_tmp}")
    sd = {
        "sampling_density": sampling_density, "grad_mac": grad_mac, "indices_ac": indices_ac, "mask": mask,
        # "importance_mask": importance_mask
    }
    with open(path_tmp, "wb") as f:
        pickle.dump(sd, f)
    plot_sampling_density(grad=grad_mac, sampling_density=sampling_density, ac_mask=mask, path=path_out)

    # create sub-sampling masks for visuals
    path_tmp = path_out_data.joinpath("masks").with_suffix(".pkl")
    if path_tmp.is_file():
        with open(path_tmp.as_posix(), "rb") as  j_file:
            masks = pickle.load(j_file)
    else:
        masks = create_subsampling_masks(k.squeeze().shape)
        m_opt = built_optimized_sampling_mask(
            acceleration=5, grad_mac=grad_mac, indices_ac=indices_ac, shape=k.shape,
            # importance_mask=importance_mask
        )
        masks.append({"name": "Optimized", "data": m_opt.squeeze()})

    # add deterministic mask too

    for i, n in enumerate(['_normed', '_normed5']):
        n_outer = sampling_mask.shape[0] - indices_ac.shape[0]
        n_to_keep = n_outer // 5
        n_th = n_outer - n_to_keep

        fn = path_tmp.parent.joinpath(f"sampling_mask{n}").with_suffix(".pt")
        sampling_mask = torch.load(fn)

        mask_opt_d = sampling_mask.clone()
        mask_opt_d[mask_opt_d < n_th] = 0
        mask_opt_d[mask_opt_d >= n_th] = 1
        mask_opt_d = mask_opt_d[None, :, None, :].expand_as(k)
        name = "Opt.Determ."
        if n:
            name += f"{n[1:].capitalize()}"
        masks.append({"name": name, "data": mask_opt_d})

    with open(path_tmp.as_posix(), "wb") as j_file:
        pickle.dump(masks, j_file)
    plot_sampling_masks(masks, path=path_out)

    if loraks_type == LoraksType.AC:
        path_tmp = path_out_data.joinpath("ac_recon_results").with_suffix(".json")
        if path_tmp.is_file():
            # with open(path_tmp.as_posix(), "rb") as  j_file:
            #     results = pickle.load(j_file)
            results = pl.read_ndjson(path_tmp)
        else:
            # compare sampling patterns
            options = AcLoraksOptions(
                solver_type=SolverType.AUTOGRAD, regularization_lambda=0.0, max_num_iter=500,
                loraks_matrix_type=OperatorType.S
            )
            options.rank.value = 150
            loraks = Loraks.create(options=options)
            results = recon_sampling_patterns(k=k, masks=masks, bet=bet, loraks=loraks)
            df = [{"name": r["name"], "psnr": r["psnr"], "ssim": r["ssim"], "nmse": r["nmse"]} for r in results]
            df = pl.DataFrame(df)
            df.write_ndjson(path_tmp.with_suffix(".json"))
        plot_results(results, path=path_out)
        plot_rmse(results, path=path_out)

        logger.info(f"RMSE: {[r['rmse'] for r in results]}")
        logger.info(f"Norm: {[r['norm'] for r in results]}")


def plot_metrics(loraks_type: LoraksType = LoraksType.AC, data_type: DataType = DataType.INVIVO):
    # get path
    path_out = plib.Path(
        get_test_result_output_dir(
            f"autograd_subsampling_optimization",
            mode=ResultMode.EXPERIMENT)
    )
    path_out = path_out.joinpath(f"{loraks_type.name}".lower()).joinpath(f"{data_type.name}".lower())
    path_out_data = path_out.joinpath(f"data")
    path_tmp = path_out_data.joinpath("ac_recon_results").with_suffix(".pkl")
    if not path_tmp.is_file():
        msg = f"File not found: {path_tmp}"
        raise FileNotFoundError(msg)

    with open(path_tmp.as_posix(), "rb") as j_file:
        results = pickle.load(j_file)

    gt = results[0]["img_us"]
    gt = root_sum_of_squares(gt, dim_channel=-2)

    for i, d in enumerate(results[1:]):
        name=  d["name"]
        psnr = d["psnr"]
        ssim = d["ssim"]
        nmse = d["nmse"]
        logger.info(f"{name}: PSNR = {psnr:.5f}: SSIM = {ssim:.5f}: NMSE = {nmse:.5f}")


def plot_mask(loraks_type: LoraksType, data_type: DataType):
    # get path
    path_out = plib.Path(
        get_test_result_output_dir(
            f"autograd_subsampling_optimization",
            mode=ResultMode.EXPERIMENT)
    )
    path_out = path_out.joinpath(f"{loraks_type.name}".lower()).joinpath(f"{data_type.name}".lower())
    fn = path_out.joinpath("data").joinpath("sampling_mask").with_suffix(".pt")
    logger.info(f"Write file: {fn}")
    sampling_mask = torch.load(fn).cpu()
    # if we want to keep n lines we do the following
    n_ac = 36
    acc = 3

    n_outer = sampling_mask.shape[0] - n_ac
    n_to_keep = n_outer // acc
    n_th = n_outer - n_to_keep

    mask = sampling_mask.clone()
    mask[mask < n_th] = 0
    mask[mask >= n_th] = 1
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(z=mask)
    )
    fn = path_out.joinpath("sampling_mask").with_suffix(".html")
    logger.info(f"Write file: {fn}")
    fig.write_html(fn)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO,
                        # handlers=[RichHandler(rich_tracebacks=True)]
                        )
    # sampling_density_correlation_grad()
    # sampling_density_correlation_recon()
    sampling_density_correlation()
    # subsampling_optimization_loraks(
    #     loraks_type=LoraksType.AC, data_type=DataType.INVIVO,
    #     device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
    # )
    # plot_metrics()
    # plot_mask(loraks_type=LoraksType.AC, data_type=DataType.INVIVO)
    #
    #
