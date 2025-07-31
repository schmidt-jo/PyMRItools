import sys
import logging
import pathlib as plib
import pickle

import polars as pl

import torch
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


def rank_parameter_influence():
    # get path
    path = plib.Path(get_test_result_output_dir(
        f"automatic_low_rank_param_extraction/invivo_ac_random_lines".lower(),
        mode=ResultMode.EXPERIMENT)
    )
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    k, _, bet = load_data(data_type=DataType.INVIVO)
    bet = bet[:,:,0].to(torch.bool)

    shape = k.shape
    gt = fft_to_img(k, dims=(0, 1))
    gt = root_sum_of_squares(gt, dim_channel=-2)
    gt[~bet] = 0

    name = f"recon_img_rsos_gt"
    torch_save(data=gt, path_to_file=path.joinpath("recon_data"), file_name=name)

    data_in, in_shape, padding, batch_channel_idx = prep_k_space(k.unsqueeze_(2), batch_size_channels=8)
    op = Operator(
        k_space_shape=data_in.shape[1:], nb_side_length=5, device=device, operator_type=OperatorType.S
    )
    mat = op.forward(data_in[0].to(device))
    _, th = process_rsvd(data_in.to(device), op, area_thr_factor=0.5)
    logger.info(f"S-Matrix shape: {mat.shape}")
    m = min(mat.shape)
    logger.info(f"Rank estimation: {th}")
    del mat
    torch.cuda.empty_cache()
    results = []
    # for si, sub_sample in enumerate(["ac-random-lines", "ac-random", "grappa", "interleaved-lines"]):
    for si, sub_sample in enumerate(["ac-random-lines"]):
        # for acc in [2, 3, 4, 5, 6]:
        for acc in [4]:
            # create sampling mask
            phantom = Phantom.get_shepp_logan(shape=shape[:2], num_coils=shape[-2], num_echoes=shape[-1])
            if si == 0:
                k_us = phantom.sub_sample_ac_random_lines(acceleration=acc, ac_lines=36)
            elif si == 1:
                k_us = phantom.sub_sample_random(acceleration=acc, ac_central_radius=20)
            elif si == 2:
                k_us = phantom.sub_sample_ac_grappa(acceleration=acc, ac_lines=36)
            else:
                k_us = phantom.sub_sample_ac_skip_lines(acceleration=acc, ac_lines=36)
            mask = k_us.abs() > 1e-10
            k_in = k.clone().squeeze()
            k_in[~mask] = 0
            data_in, in_shape, padding, batch_channel_idx = prep_k_space(k_in.unsqueeze(2), batch_size_channels=8)

            img_us = fft_to_img(k_in, dims=(0, 1))
            img_us = root_sum_of_squares(img_us, dim_channel=-2)

            name = f"recon_img_rsos_us"
            torch_save(data=img_us, path_to_file=path.joinpath("recon_data"), file_name=name)

            nmse = calc_nmse(gt, img_us)
            psnr = calc_psnr(gt, img_us)
            ssim = calc_ssim(gt.permute(2, 1, 0), img_us.permute(2, 1, 0))

            results.append({
                "sub-sample": sub_sample, "acc": acc,
                "lambda": None, "rank": None, "name": "us", "nmse": nmse, "psnr": psnr, "ssim": ssim
            })

            # for l in [0.0, 0.01, 0.04, 0.1]:
            for l in [0.0]:

                for i, r in enumerate(
                        # torch.linspace(20, 200, 10).to(torch.int).tolist() +
                        # torch.linspace(250, int(0.25 * m), 10).to(torch.int).tolist()
                    torch.arange(20, 300, 40).tolist()
                ):
                    # create rank grid
                    ac_opts = AcLoraksOptions(
                        loraks_neighborhood_size=5, loraks_matrix_type=OperatorType.S,
                        rank=RankReduction(method=RankReductionMethod.HARD_CUTOFF, value=r), regularization_lambda=l,
                        max_num_iter=30, device=device
                    )
                    loraks = Loraks.create(ac_opts)
                    k_recon = loraks.reconstruct(data_in)
                    k_recon = unprep_k_space(k_recon, padding=padding, batch_idx=batch_channel_idx, input_shape=in_shape)
                    img_recon = fft_to_img(k_recon, dims=(0, 1)).squeeze()
                    rsos = root_sum_of_squares(img_recon, dim_channel=-2)
                    rsos[~bet] = 0

                    nmse = calc_nmse(gt, rsos)
                    psnr = calc_psnr(gt, rsos)
                    ssim = calc_ssim(gt.permute(2, 1, 0), rsos.permute(2, 1, 0))

                    results.append({
                        "sub-sample": sub_sample, "acc": acc,
                        "lambda": l, "rank": r, "name": "recon", "nmse": nmse, "psnr": psnr, "ssim": ssim
                    })

                    df = pl.DataFrame(results)
                    fn = path.joinpath("metrics_rank").with_suffix(".json")
                    logger.info(f"Update file: {fn}")
                    df.write_ndjson(fn)
                    name = f"recon_img_rsos_sub-{sub_sample}_acc{acc}_r{r}_lambda-{l}".replace(".", "p")
                    torch_save(data=rsos, path_to_file=path.joinpath("recon_data"), file_name=name)



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

        k, _, bet = load_data(data_type=data_type)
        shape = k.shape
        k = torch.reshape(k, (*shape[:2], -1))
        k_zoom = None
        for idx_m in tqdm.trange(k.shape[-1], desc="adjust shape"):
            img = fft_to_img(k[..., idx_m], dims=(0, 1)).numpy()
            img = torch.from_numpy(
                zoom(img, zoom=(0.6, 0.6), )
            )
            ifft = ifft_to_k(img, dims=(0, 1)).unsqueeze(-1)
            if idx_m == 0:
                k_zoom = ifft
            else:
                k_zoom = torch.concatenate([k_zoom, ifft], dim=-1)
        k = torch.reshape(k_zoom, (*k_zoom.shape[:2], *shape[-2:]))
        nx, ny, nc, ne = k.shape
        logger.info(f"Shape in: {shape}, shape interp.: {k.shape}")
        phantom = Phantom.get_shepp_logan(shape=(nx, ny), num_coils=nc, num_echoes=ne)
        for acc in torch.linspace(1, 10, 8):
            logger.info(f"Process acc: {acc:.2f}")
            logger.info("_______________________")
            logger.info("_______________________")
            # create sampling mask
            # k_us = phantom.sub_sample_random(acceleration=acc, ac_central_radius=20)
            k_us = phantom.sub_sample_ac_random_lines(acceleration=acc, ac_lines=24)
            mask = k_us.abs() > 1e-10
            k_in = k.clone()
            k_in[~mask] = 0
            calc_acc = torch.prod(torch.tensor(mask.shape)) / torch.count_nonzero(mask)
            # reduce available data
            bar = tqdm.tqdm(torch.arange(4, nc+1, 4), desc="Processing Channels")
            for ic in bar:
                for ie in torch.arange(2, ne+1, 2):
                    bar.set_postfix({"nc": nc, "ne": ne})
                    if ic * ie > 32*6:
                        break
                    # some randomization runs
                    for rc in range(3):
                        idx_c = torch.randint(low=0, high=nc, size=(ic,))
                        # some randomization runs
                        for re in range(3):
                            idx_e = torch.randint(low=0, high=ne, size=(ie,))
                            data = k_in[:, :, idx_c, :][..., idx_e]
                            data_in, _, _, _ = prep_k_space(data.unsqueeze_(2), batch_size_channels=-1)
                            op = Operator(
                                k_space_shape=data_in.shape[1:], nb_side_length=5, device=device, operator_type=OperatorType.S
                            )
                            _, th = process_rsvd(data_in.to(device), op, area_thr_factor=0.6)
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
    # get df threshold estimation normalized by acc = 1 threshold
    # create lookup reference
    reference_values = (
        df.filter(pl.col("acc") == 1).group_by(
            ["nc", "mat_size", "random_c", "random_e"]
        ).agg(pl.col("thresh").first().alias("ref_thresh"))
    )
    df_norm = (
        df.join(reference_values, on=["nc", "mat_size", "random_c", "random_e"], how="left").
        with_columns(
            (pl.col("thresh") / pl.col("ref_thresh")).alias("normalized_thresh")
        )
        .drop("ref_thresh")
    )

    df_norm.write_ndjson(
        path.joinpath("df_norm").with_suffix(".json")
    )
    df_norm.filter(pl.col("mat_size") == df_norm["mat_size"].unique().min()).write_ndjson(
        path.joinpath("df_norm_mat_sizes").with_suffix(".json")
    )

    for irc in range(3):
        for ire in range(3):
            df_plot = df.filter(
                (pl.col("random_c") == irc) & (pl.col("random_e") == ire)
            ).drop(["random_c", "random_e"])
            df_plot_norm = df_norm.filter(
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
                    marker=dict(color=df_plot["calc_acc"], coloraxis="coloraxis2"),
                ),
                row=2, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df_plot_norm["mat_size"] - ps + 2*ps/9 * (3 * irc + ire), y=df_plot_norm["normalized_thresh"],
                    mode="markers",
                    showlegend=False,
                    marker=dict(color=df_plot_norm["calc_acc"], coloraxis="coloraxis2"),
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
            cmin=0.0, cmax=df["calc_acc"].max(),
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


def mask_rank(data_type: DataType):
    # get path
    path = plib.Path(get_test_result_output_dir(
        f"automatic_low_rank_param_extraction/{data_type.name}".lower(),
        mode=ResultMode.EXPERIMENT)
    )
    path_out = path.joinpath("result_data").with_suffix(".pkl")
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    k, _, bet = load_data(data_type=DataType.INVIVO)
    nx, ny, nc, ne = k.shape
    nb_size = 5**2
    phantom = Phantom.get_shepp_logan(shape=(nx, ny), num_coils=nc, num_echoes=ne)
    results = []
    for acc in torch.linspace(1, 10, 6):
        logger.info(f"Process acc: {acc:.2f}")
        logger.info("_______________________")
        logger.info("_______________________")
        # create sampling mask
        k_us = phantom.sub_sample_random(acceleration=acc, ac_central_radius=20)
        mask = (k_us.abs() > 1e-10).to(torch.int)
        calc_acc = torch.prod(torch.tensor(mask.shape)) / torch.count_nonzero(mask)
        # reduce available data
        for ic in tqdm.tqdm(torch.arange(4, nc + 1, 6)):
        # for ic in tqdm.tqdm(torch.arange(4, 4 + 1, 4)):
            # some randomization runs
            for rc in range(3):
                idx_c = torch.randint(low=0, high=nc, size=(ic,))
                for ie in torch.arange(1, ne + 1, 3):
                # for ie in torch.arange(2, 2 + 1, 2):
                    if ic * ie > 32 * 6:
                        break
                    # some randomization runs
                    for re in range(3):
                        idx_e = torch.randint(low=0, high=ne, size=(ie,))
                        data = mask[:, :, idx_c, :][..., idx_e]
                        data_in, _, _, _ = prep_k_space(data.unsqueeze_(2), batch_size_channels=-1)
                        op = Operator(
                            k_space_shape=data_in.shape[1:], nb_side_length=5, device=device,
                            operator_type=OperatorType.S
                        )
                        _, th = process_rsvd(data_in.to(device), op, area_thr_factor=0.9)
                        mat_size = nb_size * ic * ie
                        results.append({
                            "name": f"Sub-Sampled", "acc": acc.item(), "thresh": th, "random_c": rc, "random_e": re,
                            "nc": ic.item(), "ne": ie.item(), "mat_size": mat_size.item(), "calc_acc": calc_acc.item()
                        })

        logger.info(f"Write file: {path_out}")
        with open(path_out.as_posix(), "wb") as f:
            pickle.dump(results, f)

        plot_results_vs_acc_mat_size(results, path)


def plot_rank_param_influence_errors():
    path = plib.Path(get_test_result_output_dir(
        "automatic_low_rank_param_extraction", mode=ResultMode.EXPERIMENT
    )).joinpath(
        "invivo_ac_random_lines"
    )
    path_recon_data = path.joinpath("recon_data")
    cmaps = ["Purples", "Greens", "Oranges", "Greys", "Blues", "Reds"]
    df = pl.read_ndjson(path.joinpath("metrics_rank.json"))
    lambdas = df.filter(pl.col("lambda").is_not_null())["lambda"].unique()
    num_lambda = len(lambdas)
    accs = df.filter(pl.col("acc").is_not_null())["acc"].unique().sort(descending=False).to_list()
    num_acc = len(accs)
    sub_samples = df["sub-sample"].unique().to_list()
    num_subs = len(sub_samples)

    # plot fixed acc
    acc = 3
    fig = psub.make_subplots(
        rows=3, cols=num_subs,
        vertical_spacing=0.02, horizontal_spacing=0.02,
        shared_xaxes=True, shared_yaxes=True,
        x_title="Reconstruction Rank Parameter",
        row_titles=["NMSE", "PSNR", "SSIM"],
        column_titles=sub_samples,
    )
    for si, sub in enumerate(sub_samples):
        for h, err in enumerate(["nmse", "psnr", "ssim"]):
            # for g, acc in enumerate(accs):
            cmap = plc.sample_colorscale("Inferno", num_lambda)
            for i, l in enumerate(lambdas):
                df_tmp = df.filter(
                    (pl.col("lambda") == l) &
                    (pl.col("rank").is_not_null()) &
                    (pl.col("acc") == acc) &
                    (pl.col("sub-sample") == sub) &
                    (pl.col("rank") < 500)
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_tmp["rank"], y=df_tmp[err],
                        marker=dict(color=cmap[i]),
                        mode="lines+markers",
                        showlegend=(h == 0) & (si == 0), name=f"Lambda: {l}", legendgroup=l,
                        # cmin=zmin, cmax=zmax
                    ),
                    row=1+h, col=1+si
                )
    fig.update_yaxes(range=(0, 0.035), row=1, col=1)
    fig.update_yaxes(range=(20, 45), row=2, col=1)
    fig.update_yaxes(range=(0.92, 1), row=3, col=1)
    fn = path.joinpath("error_vs_rank_acc3").with_suffix(".html")
    logger.info(f"Write file: {fn}")
    fig.write_html(fn)

    for suff in [".png", ".pdf"]:
        fn = fn.with_suffix(suff)
        logger.info(f"Write file: {fn}")
        fig.write_image(fn)

    # plot fixed lambda
    l = 0.0
    fig = psub.make_subplots(
        rows=3, cols=num_subs,
        vertical_spacing=0.02, horizontal_spacing=0.02,
        shared_xaxes=True, shared_yaxes=True,
        x_title="Reconstruction Rank Parameter",
        row_titles=["NMSE", "PSNR", "SSIM"],
        column_titles=sub_samples,
    )
    for si, sub in enumerate(sub_samples):
        for h, err in enumerate(["nmse", "psnr", "ssim"]):
            # for g, acc in enumerate(accs):
            cmap = plc.sample_colorscale("Inferno", num_acc)
            for i, acc in enumerate(accs):
                df_tmp = df.filter(
                    (pl.col("lambda") == l) &
                    (pl.col("rank").is_not_null()) &
                    (pl.col("acc") == acc) &
                    (pl.col("sub-sample") == sub) &
                    (pl.col("rank") < 500)
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_tmp["rank"], y=df_tmp[err],
                        marker=dict(color=cmap[i]),
                        mode="lines+markers",
                        showlegend=(h == 0) & (si == 0), name=f"Acc.: {acc}", legendgroup=l,
                        # cmin=zmin, cmax=zmax
                    ),
                    row=1+h, col=1+si
                )
    fig.update_yaxes(range=(0, 0.015), row=1, col=1)
    fig.update_yaxes(range=(32, 47), row=2, col=1)
    fig.update_yaxes(range=(0.92, 1), row=3, col=1)
    fn = path.joinpath("error_vs_rank_lambda0").with_suffix(".html")
    logger.info(f"Write file: {fn}")
    fig.write_html(fn)

    for suff in [".png", ".pdf"]:
        fn = fn.with_suffix(suff)
        logger.info(f"Write file: {fn}")
        fig.write_image(fn)


    # plot fixed sub
    sub = "ac-random-lines"
    fig = psub.make_subplots(
        rows=3, cols=2,
        vertical_spacing=0.02, horizontal_spacing=0.02,
        shared_xaxes=True, shared_yaxes=True,
        x_title="Reconstruction Rank Parameter",
        row_titles=["NMSE", "PSNR", "SSIM"],
        column_titles=["Lambda", "Acceleration"],
    )
    a = [[[3]*len(lambdas), lambdas], [accs, [0.0]*len(accs)] ]
    cmaps = ["Agsunset", "Aggrnyl_r"]
    for h, err in enumerate(["nmse", "psnr", "ssim"]):
        for i, d in enumerate(a):
            aa, ll = d
            cmap = plc.sample_colorscale(cmaps[i], len(aa))
            for g, acc in enumerate(aa):
                for n, l in enumerate(ll):
                    c = n if i==0 else g
                    if i == 0:
                        showlegend = True if g == 0 and h==0 else False
                    else:
                        showlegend = True if n == 0 and h==0 else False
                    df_tmp = df.filter(
                        (pl.col("lambda") == l) &
                        (pl.col("rank").is_not_null()) &
                        (pl.col("acc") == acc) &
                        (pl.col("sub-sample") == sub) &
                        (pl.col("rank") < 500)
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=df_tmp["rank"], y=df_tmp[err],
                            marker=dict(color=cmap[c], coloraxis="coloraxis"),
                            mode="lines+markers",
                            showlegend=showlegend,
                            name=f"Acc.: {acc}" if i ==1 else f"Lambda: {l}"
                        ),
                        row=1+h, col=1+i
                    )
    fig.update_yaxes(range=(0, 0.035), row=1, col=1)
    fig.update_yaxes(range=(20, 45), row=2, col=1)
    fig.update_yaxes(range=(0.92, 1), row=3, col=1)
    fig.update_layout(
        width=800, height=450,
        coloraxis=dict(
            colorscale="Agsunset",
            cmin=0.0, cmax=lambdas.max(),
            colorbar=dict(
                x=0.5, y=-0.01, len=0.4, thickness=14,
                # title=dict(text="Difference to Fully Sampled [%]", side="right"),
                xref="paper", yref="paper",
                xanchor="left", yanchor="bottom",
            ),
        ),
        coloraxis2=dict(
            colorscale="Aggrnyl_r",
            cmin=min(accs), cmax=max(accs),
            colorbar=dict(
                x=0.5, y=0.5, len=0.4, thickness=14,
                # title=dict(text="Difference to Fully Sampled [%]", side="right"),
                xref="paper", yref="paper",
                xanchor="left", yanchor="bottom",
            ),
        ),
        margin=dict(t=20, b=20, l=10, r=20)
    )
    fn = path.joinpath("error_vs_rank_per_sub").with_suffix(".html")
    logger.info(f"Write file: {fn}")
    fig.write_html(fn)

    for suff in [".png", ".pdf"]:
        fn = fn.with_suffix(suff)
        logger.info(f"Write file: {fn}")
        fig.write_image(fn)

def plot_rank_param_influence_imgs():
    path = plib.Path(get_test_result_output_dir(
        "automatic_low_rank_param_extraction", mode=ResultMode.EXPERIMENT
    )).joinpath(
        "invivo_ac_random_lines"
    )
    k, _, bet = load_data(DataType.INVIVO)
    bet = bet[:,:,0].to(torch.bool)
    shape = k.shape

    # create sampling mask
    phantom = Phantom.get_shepp_logan(shape=shape[:2], num_coils=shape[-2], num_echoes=shape[-1])
    k_us = phantom.sub_sample_ac_random_lines(acceleration=4, ac_lines=36)
    mask = k_us.abs() > 1e-10
    k_in = k.clone()
    k_in[~mask] = 0
    img_us = root_sum_of_squares(fft_to_img(k_in, dims=(0, 1)), dim_channel=-2)
    img_us[~bet] = 0

    img_gt = fft_to_img(k, dims=(0, 1))
    img_gt = root_sum_of_squares(img_gt, dim_channel=-2)
    img_gt[~bet] = 0
    img_max = img_gt.max().item() * 0.6

    path_recon_data = path.joinpath("recon_data")
    lambdas = []
    ranks = []

    for f in path_recon_data.iterdir():
        if f.is_file():
            parts = f.stem.split("_")
            rank = int(parts[-2][1:])
            if rank not in ranks:
                ranks.append(rank)
            lam = float(parts[-1].split("-")[-1].replace("p", "."))
            if lam not in lambdas:
                lambdas.append(lam)
    ranks.sort()
    lambdas.sort()
    num_lambda = len(lambdas)
    num_ranks = len(ranks)

    fig = psub.make_subplots(
        rows=num_lambda, cols=num_ranks + 2,
        row_titles=lambdas, column_titles=["GT", "US"] + ranks,
        vertical_spacing=0.01, horizontal_spacing=0.01
    )
    for l in range(num_lambda):
        for i, d in enumerate([img_gt, img_us]):
            fig.add_trace(
                go.Contour(
                    z=d[:, :, 0], showlegend=False, showscale=False,
                    colorscale="Inferno",
                    zmin=0, zmax=img_max
                ),
                row=1+l, col=1+i
            )

    for f in path_recon_data.iterdir():
        if f.is_file():
            parts = f.stem.split("_")
            rank = int(parts[-2][1:])
            lam = float(parts[-1].split("-")[-1].replace("p", "."))
            d = torch_load(f)
            row = lambdas.index(lam) + 1
            col = ranks.index(rank) + 3

            fig.add_trace(
                go.Heatmap(
                    z=d[:, :, 0], showlegend=False, showscale=False,
                    colorscale="Inferno",
                    zmin=0, zmax=img_max
                ),
                row=row, col=col
            )

    fig.update_yaxes(visible=False)
    fig.update_xaxes(visible=False)

    # for i, d in enumerate(data):
    #     dp = d["data"].abs()[:, :, 0]
    #     fig.add_trace(
    #         go.Heatmap(
    #             z=dp
    #         ),
    #         row=1, col=1+i
    #     )

    fn = path.joinpath("img_vs_rank").with_suffix(".html")
    logger.info(f"Write file: {fn}")
    fig.write_html(fn)


def plot_wandb_sweep(force_api_load: bool = False):
    path = plib.Path(
        get_test_result_output_dir("automatic_low_rank_param_extraction", ResultMode.EXPERIMENT)
    ).absolute().joinpath("wandb")
    logger.info(f"Set path: {path}")

    path.mkdir(exist_ok=True, parents=True)

    fn = path.joinpath("wandb_runs").with_suffix(".json")
    if not fn.exists() or force_api_load:
        api = wandb.Api().project("loraks_rank_optimization")
        sweeps = api.sweeps()

        runs = sweeps[0].runs

        sum_list = []
        for run in tqdm.tqdm(runs, desc="process wandb runs"):
            config = {
                k: v for k, v in run.config.items() if not k.startswith("_")
            }
            tmp = {
                k: v for k, v in run.summary._json_dict.items() if not k.startswith("_")
            }
            if not tmp:
                 continue

            tmp["rank"] = config["rank"]
            tmp["sub"] = config["sub"]
            tmp["name"] = run.name
            sum_list.append(tmp)

        df = pl.DataFrame(sum_list)
        logger.info(df)
        logger.info(f"Save file: {fn}")
        df.write_ndjson(fn)
    else:
        logger.info(f"Read file: {fn}")
        df = pl.read_ndjson(fn)

    df_data = df.drop("name")
    mapping = {}
    for i, val in enumerate(df_data["sub"].unique()):
        mapping[val] = i
    df_data = df_data.with_columns(
        pl.col("sub").replace(mapping).alias("sub-num").cast(pl.Int64)
    )
    # df_data = df_data.filter(pl.col("loss") < -1.18)
    df_data = df_data.sort(by="loss", descending=True)

    logger.info(f"Optimal Rank: {df_data['rank'][-1]}, Optimal Lambda: {df_data['lambda'][-1]}, Optimal sub: {df_data['sub'][-1]}")
    logger.info(df_data)

    columns = ["rank", "sub-num", "lambda", "psnr", "nmse", "ssim", "loss"]
    logger.info(columns)
    data_n = np.zeros((len(df_data), len(columns)))

    for i, c in enumerate(columns):
        data_n[:, i] = (df_data[c].to_numpy() - df_data[c].min()) / (df_data[c].max() - df_data[c].min())

    n_ext = 60
    n_ext_edge = int(0.1 * n_ext)
    data_straight = np.zeros((data_n.shape[0], n_ext * (data_n.shape[1] - 1)))
    for i, d in enumerate(data_n):
        for r in range(data_n.shape[1]-1):
            start = data_n[i, r]
            end = data_n[i, r+1]
            data_straight[i, r*n_ext:r*n_ext+n_ext_edge] = np.linspace(start, start, n_ext_edge, endpoint=True)
            data_straight[i, r*n_ext+n_ext_edge:(r+1)*n_ext-n_ext_edge] = np.linspace(start, end, n_ext - 2*n_ext_edge, endpoint=True)
            data_straight[i, (r+1)*n_ext-n_ext_edge:(r+1)*n_ext] = np.linspace(end, end, n_ext_edge, endpoint=True)

    data_smooth = savgol_smooth_parallel_coords(data=data_straight, window_length=int(0.8*n_ext), polyorder=2)
    data_smooth = savgol_smooth_parallel_coords(data=data_smooth, window_length=5, polyorder=2)
    # data_smooth = gaussian_smooth_parallel_coords(data=data_straight, sigma=10)

    loss_th = -1.36
    plot_indices_color = df_data.with_row_index().filter(pl.col("loss") < loss_th).select("index").to_numpy().squeeze()
    plot_indices_grey = df_data.with_row_index().filter(pl.col("loss") > loss_th).select("index").to_numpy().squeeze()

    fig = go.Figure()
    for ii, idx in enumerate([plot_indices_grey, plot_indices_color]):
        data_plot = data_smooth[idx]
        dmin, dmax = data_plot[:, -1].min(), data_plot[:, -1].max()
        if ii == 1:
            cm = "Inferno_r"
            op = 0.3
            amin, amax = 0, 0.8
        else:
            cm = "Gray"
            op = 0.1
            amin, amax = 0.5, 0.8
        cmap = plc.sample_colorscale(cm, np.clip((data_plot[:, -1] - dmin) / (dmax - dmin), amin, amax))

        for i, d in enumerate(data_plot):
            fig.add_trace(
                go.Scatter(
                    y=d,
                    showlegend=False,
                    opacity=op,
                    mode="lines",
                    line=dict(color=cmap[i])
                )
            )
    for i, c in enumerate(columns[1:]):
        fig.add_trace(
            go.Scatter(
                x=[n_ext*(i+1)], y=[0, 1],
                yaxis=f"y{i+2}",
                showlegend=False,
                mode="lines",
                line=dict(color="white", width=1),
            )
        )
        ymin = df_data[c].min()
        ymax = df_data[c].max()
        fig.update_layout(**{
            f'yaxis{i + 2}': {
                "range": (-0.1, 1.1),
                "anchor": 'free',
                'title': dict(
                    text=c.upper() if 1 < i < len(columns) - 2 else c.capitalize(),
                    standoff=1,
                ),
                'overlaying': 'y',
                'side': 'left' if i < len(columns) - 2 else 'right',
                'position': 0.999 / (len(columns) - 1) * (i+1),  # Adjust positioning
                'tickmode': 'array',
                'tickvals': np.linspace(0, 1, 7) if i > 0 else (df_data["sub-num"].unique() / len(df_data["sub-num"].unique())).to_list(),
                'ticktext': [f"{x:.4f}" for x in np.linspace(ymin, ymax, 7)] if i> 0 else df_data["sub"].unique().to_list(),
                'layer': 'above traces',
            }
        })

    fig.update_layout(
        xaxis=dict(
            title="Parameter - METRIC",
            tickmode="array",
            tickvals=np.arange(n_ext*len(columns) - 1) * n_ext,
            ticktext=[columns[i].capitalize() if i in [0, 1, 2, len(columns) - 1] else columns[i].upper() for i in range(len(columns))]
        ),
        yaxis=dict(
            range =(-0.1, 1.1),
            title="Rank",
            tickmode="array",
            tickvals=np.linspace(0, 1, 13),
            ticktext=[f"{int(x)}" for x in np.linspace(df_data["rank"].min(), df_data["rank"].max(), 13)]
        ),
        width=800, height=350,
        margin=dict(t=10, b=10, l=10, r=10)
    )
    fn = path.joinpath("sweep_trendline").with_suffix(".html")
    logger.info(f"Write file: {fn}")
    fig.write_html(fn)

    for suff in [".png", ".pdf"]:
        fn = fn.with_suffix(suff)
        logger.info(f"Write file: {fn}")
        fig.write_image(fn)


def plot_low_rank_influence():
    path = plib.Path(get_test_result_output_dir("automatic_low_rank_param_extraction", ResultMode.EXPERIMENT))
    path = path.joinpath("invivo_ac_random_lines")
    path_in = path.joinpath("recon_data")
    # load metrics
    fn = path.joinpath("metrics_rank").with_suffix(".json")
    logger.info(f"Read file: {fn}")
    metrics = pl.read_ndjson(fn)

    logger.info(metrics)

    # we know that the optimal rank is ~150 from optimisation
    # load in some data
    gt = torch_load(path_in.joinpath("recon_img_rsos_gt").with_suffix(".pt"))
    mask = gt > 1e-9
    us = torch_load(path_in.joinpath("recon_img_rsos_us").with_suffix(".pt"))
    us[~mask] = 0

    zmax = gt.abs().max().item() * 0.65

    data = [gt, us]
    ranks = [0, 1]
    for i, f in enumerate(path_in.iterdir()):
        if "sub-ac" in f.stem:
            data.append(torch_load(f))
            r = int(f.stem.split("_")[-2][1:])
            ranks.append(r)
    ranks = np.array(ranks)
    inds = np.argsort(ranks)
    # sort the data according to the rank used in the recon, take the first echo only
    data = np.stack(data)[inds][..., 0]
    ranks = ranks[inds]

    data = data[:-2]
    ranks = ranks[:-2]

    zzmax = (data[2] - data[0]).max() * 0.1

    fig = psub.make_subplots(
        rows=2, cols=data.shape[0],
        column_titles=["GT", "US"] + [f"Rank: {r}" for r in ranks[2:]],
        shared_yaxes=True,
        vertical_spacing=0.02, horizontal_spacing=0.01
    )

    for i, d in enumerate(data):
        for h, dd in enumerate([d, d - data[0]]):
            # plot images
            fig.add_trace(
                go.Heatmap(
                    z=dd, showscale=False, showlegend=False,
                    colorscale="Inferno" if h ==0 else "Balance_r",
                    zmin=0.0 if h == 0 else -zzmax, zmax=zmax if h == 0 else zzmax
                ),
                row=1+h, col=1+i
            )

            xaxis=fig.data[-1].xaxis
            fig.update_yaxes(visible=False, row=1+h, col=1+i, scaleanchor=xaxis)
            fig.update_xaxes(visible=False, row=1+h, col=1+i)

    fig.update_layout(
        width=800, height = 300,
        margin=dict(t=20, b=10, l=10, r=10)
    )
    fn = path.joinpath("low_rank_influence").with_suffix(".html")
    logger.info(f"Write file: {fn}")
    fig.write_html(fn)

    for suff in [".png", ".pdf"]:
        fn = fn.with_suffix(suff)
        logger.info(f"Write file: {fn}")
        fig.write_image(fn)

    mets = ["nmse", "psnr", "ssim"]
    fig = psub.make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
    )
    tmp = metrics.filter(pl.col("name") == "recon")
    names = [f"Rank: {r}" for r in tmp["rank"].sort().to_list()]
    ranges=[(0.00, 0.008), (35, 42), (0.92, 0.98)]
    cmap = plc.sample_colorscale("Inferno", 3, 0.7, 0.3)
    for i, m in enumerate(mets):
        fig.add_trace(
            go.Bar(
                x=names, y=tmp[m],
                marker=dict(color=cmap[i]),
                showlegend=False,
            ),
            row=1+i, col=1
        )
        fig.update_yaxes(
            range=ranges[i], row=1+i, col=1, side="right", title=m.upper()
        )
    fig.update_layout(
        width=650, height=250,
        margin=dict(t=10, b=10, l=10, r=10)
    )
    fn = path.joinpath("low_rank_influence_metrics").with_suffix(".html")
    logger.info(f"Write file: {fn}")
    fig.write_html(fn)

    for suff in [".png", ".pdf"]:
        fn = fn.with_suffix(suff)
        logger.info(f"Write file: {fn}")
        fig.write_image(fn)

def savgol_smooth_parallel_coords(data, window_length=5, polyorder=2):
    """
    Apply Savitzky-Golay smoothing

    Args:
        data (np.ndarray): Original data
        window_length (int): Length of the filter window
        polyorder (int): Order of the polynomial used to fit the samples

    Returns:
        np.ndarray: Smoothed data
    """
    smoothed_data = np.copy(data)

    for dim in tqdm.trange(data.shape[0], desc="savgol smoothing"):
        smoothed_data[dim] = savgol_filter(data[dim], window_length=window_length, polyorder=polyorder)

    return smoothed_data


def gaussian_smooth_parallel_coords(data, sigma=1.0):
    """
    Apply Gaussian smoothing to each dimension

    Args:
        data (np.ndarray): Original data
        sigma (float): Standard deviation for Gaussian kernel

    Returns:
        np.ndarray: Smoothed data
    """
    smoothed_data = np.copy(data)

    for dim in range(data.shape[0]):
        smoothed_data[dim] = gaussian_filter1d(data[dim], sigma=sigma)

    return smoothed_data


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    # automatic_low_rank_param_extraction(data_type=DataType.INVIVO, force_processing=True)
    # try_arsvd()
    # mask_rank(data_type=DataType.MASK)
    # rank_parameter_influence()
    # plot_rank_param_influence_errors()
    # plot_rank_param_influence_imgs()
    # plot_wandb_sweep(force_api_load=False)
    plot_low_rank_influence()
