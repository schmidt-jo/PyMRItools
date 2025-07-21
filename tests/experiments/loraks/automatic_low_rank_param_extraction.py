import sys
import logging
import pathlib as plib
import pickle

import polars as pl

import torch
from scipy.ndimage import zoom
import plotly.graph_objects as go
import plotly.subplots as psub
import plotly.colors as plc
import tqdm

from pymritools.recon.loraks.operators import Operator, OperatorType
from pymritools.utils.algorithms import rank_estimator_adaptive_rsvd
from pymritools.utils import fft_to_img, torch_load, Phantom, ifft_to_k

p_tests = plib.Path(__file__).absolute().parent.parent.parent.parent
sys.path.append(p_tests.as_posix())
from tests.utils import get_test_result_output_dir, create_phantom, ResultMode
from tests.experiments.loraks.utils import prep_k_space, DataType, load_data

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
            k_us = phantom.sub_sample_random(acceleration=acc, ac_central_radius=20)
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
                            _, th = process_rsvd(data_in.to(device), op, area_thr_factor=0.9)
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

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    automatic_low_rank_param_extraction(data_type=DataType.INVIVO, force_processing=True)
    # try_arsvd()
    # mask_rank(data_type=DataType.MASK)
