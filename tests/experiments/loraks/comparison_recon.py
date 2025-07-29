import json
import logging
import pathlib as plib
import sys
from timeit import Timer

import torch

import plotly.graph_objects as go
import plotly.subplots as psub
import plotly.colors as plc

import polars as pl
import nibabel as nib
import tqdm

from pymritools.utils import torch_load, nifti_load, Phantom, fft_to_img, torch_save, root_sum_of_squares, calc_nmse, \
    calc_psnr, calc_ssim
from pymritools.recon.loraks.loraks import Loraks, OperatorType, RankReduction, RankReductionMethod
from pymritools.recon.loraks.ac_loraks import AcLoraksOptions, SolverType, NullspaceAlgorithm

p_tests = plib.Path(__file__).absolute().parent.parent.parent.parent
sys.path.append(p_tests.as_posix())

from tests.utils import get_test_result_output_dir, ResultMode
from tests.experiments.loraks.utils import TorchMemoryTracker, prep_k_space, unprep_k_space

logger = logging.getLogger(__name__)


def load_data():
    path_data = plib.Path(get_test_result_output_dir("data", mode=ResultMode.EXPERIMENT))
    k = torch_load(path_data.joinpath("01_k_space_rmos.pt"))
    bet = torch.from_numpy(nib.load(path_data.joinpath("01_bet.mgz")).get_fdata())
    return k, bet


def compare_batching():
    path = plib.Path(get_test_result_output_dir("reconstruction_comparisons", mode=ResultMode.EXPERIMENT)).absolute().joinpath("batching")
    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"set path: {path}")

    # load k space data
    logger.info(f"Load data")
    k_fs, bet = load_data()
    k_fs = k_fs.permute(1, 0, 2, 3, 4)

    # set device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"set device: {device}")
    # set subsampling - use phantom implementation to create the subsampling scheme
    nx, ny, ns, nc, ne = k_fs.shape
    phantom = Phantom.get_shepp_logan(shape=(nx, ny), num_coils=nc, num_echoes=ne)
    pk_us = phantom.sub_sample_ac_random_lines(acceleration=3.0, ac_lines=36)
    # extract the mask
    mask = pk_us.abs() > 1e-10
    mask = mask.unsqueeze(2).expand_as(k_fs)
    # subsample
    k_us = k_fs.clone()
    k_us[~mask] = 0

    # we now want to reconstruct once with random batching and once with the correlation batching idea
    # create ac loraks recon
    opts = AcLoraksOptions(
        loraks_neighborhood_size=5, loraks_matrix_type=OperatorType.S,
        rank=RankReduction(method=RankReductionMethod.HARD_CUTOFF, value=120), regularization_lambda=0.0,
        max_num_iter=30, device=device, solver_type=SolverType.LEASTSQUARES,
        nullspace_algorithm=NullspaceAlgorithm.EIGH
    )

    img = fft_to_img(k_fs, dims=(0, 1))
    img_us = fft_to_img(k_us, dims=(0, 1))
    for i, d in enumerate([img, img_us]):
        torch_save(d, path, f"recon_batching_comparison_{['gt', 'us'][i]}")

    for i, cb in enumerate([True, False]):
        k_recon = torch.zeros_like(k_fs)
        # set up loraks
        ac_loraks = Loraks.create(options=opts)
        for idx_z in tqdm.trange(ns):
            # prepare k-space
            k_in, in_shape, padding, batch_channel_idx = prep_k_space(
                k=k_us[:, :, idx_z, None], batch_size_channels=4, use_correlation_clustering=cb
            )
            tmp = ac_loraks.reconstruct(k_in)

            # unprep
            k_recon[:, :, idx_z] = unprep_k_space(k=tmp, padding=padding,batch_idx=batch_channel_idx, input_shape=in_shape).squeeze()

        # fft
        img_recon = fft_to_img(input_data=k_recon, dims=(0, 1))

        torch_save(img_recon, path, f"recon_batching_comparison_cb-{cb}")


def compare_variants():
    path = plib.Path(get_test_result_output_dir("reconstruction_comparisons", mode=ResultMode.EXPERIMENT)).absolute().joinpath("rsvds")
    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"set path: {path}")

    # load k space data
    logger.info(f"Load data")
    k_fs, bet = load_data()
    k_fs = k_fs.permute(1, 0, 2, 3, 4)

    # set device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"set device: {device}")
    # set subsampling - use phantom implementation to create the subsampling scheme
    nx, ny, ns, nc, ne = k_fs.shape
    phantom = Phantom.get_shepp_logan(shape=(nx, ny), num_coils=nc, num_echoes=ne)
    pk_us = phantom.sub_sample_ac_random_lines(acceleration=3, ac_lines=42)
    # extract the mask
    mask = pk_us.abs() > 1e-10
    mask = mask.unsqueeze(2).expand_as(k_fs)
    # subsample
    k_us = k_fs.clone()
    k_us[~mask] = 0

    img = fft_to_img(k_fs, dims=(0, 1))
    img_us = fft_to_img(k_us, dims=(0, 1))
    for i, d in enumerate([img, img_us]):
        torch_save(d, path, f"recon_comparison_{['gt', 'us'][i]}")

    times = []
    f_t = path.joinpath("times").with_suffix(".json")
    # prepare k-space
    k_in, in_shape, padding, batch_channel_idx = prep_k_space(
        k=k_us[:, :, k_us.shape[2] // 2, None].clone(), batch_size_channels=8
    )
    for i, nsa in enumerate(list(NullspaceAlgorithm)):
        logger.info(f"Set Algorithm: {nsa.name}")
        # set up loraks
        # create ac loraks recon
        opts = AcLoraksOptions(
            loraks_neighborhood_size=5, loraks_matrix_type=OperatorType.S,
            rank=RankReduction(method=RankReductionMethod.HARD_CUTOFF, value=200), regularization_lambda=0.0,
            max_num_iter=30, device=device, solver_type=SolverType.LEASTSQUARES,
            nullspace_algorithm=nsa
        )
        ac_loraks = Loraks.create(options=opts)
        # time - warmup and recon
        k_recon = ac_loraks.reconstruct(k_in)

        # create ac loraks recon - set convergence low to dont measure different convergence times
        opts = AcLoraksOptions(
            loraks_neighborhood_size=5, loraks_matrix_type=OperatorType.S,
            rank=RankReduction(method=RankReductionMethod.HARD_CUTOFF, value=100), regularization_lambda=0.0,
            max_num_iter=5, device=device, solver_type=SolverType.LEASTSQUARES,
            nullspace_algorithm=nsa
        )
        ac_loraks = Loraks.create(options=opts)

        timer = Timer(
            stmt="ac_loraks.reconstruct(k_in)",
            globals={"ac_loraks": ac_loraks, "k_in": k_in}
        )
        times.append({"alg": nsa.name.lower(), "time": timer.timeit(3) / 3})

        # unprep
        k_recon = unprep_k_space(k=k_recon, padding=padding,batch_idx=batch_channel_idx, input_shape=in_shape).squeeze()

        # fft
        img_recon = fft_to_img(input_data=k_recon, dims=(0, 1))

        torch_save(img_recon, path, f"recon_comparison_svd-{nsa.name}".lower())
        logger.info(f"update file: {f_t}")
        with open(f_t, "w") as json_file:
            json.dump(times, json_file, indent=2)


def plot_batching_comp():
    path = plib.Path(
        get_test_result_output_dir("reconstruction_comparisons", mode=ResultMode.EXPERIMENT)).absolute().joinpath(
        "batching")

    _, bet = load_data()
    bet =  bet.to(torch.bool).permute(1, 0, 2)

    imgs = []
    for i, t in enumerate(["gt", "us", "cb-True", "cb-False"]):
        d = torch_load(path.joinpath(f"recon_batching_comparison_{t}").with_suffix(".pt"))
        imgs.append({"name": t, "data": d})
    bet = bet.unsqueeze(-1).unsqueeze(-1).expand_as(imgs[-1]["data"])

    channel_idx = 20

    gt = imgs[0]["data"]
    gt_rsos = root_sum_of_squares(gt, dim_channel=-2) / 2
    gt_rsos = gt_rsos[:, :, gt_rsos.shape[2] // 2, 0].clone()
    gt_rsos[~bet[:, :, bet.shape[2] // 2, 0, 0]] = 0
    # gt_max = gt_rsos.abs().max().item() * 0.6

    c_max = gt[:, :, gt.shape[2] // 2, channel_idx, 0].abs().max().item() * 0.5

    # we want to build a list of lists for columns and rows
    # first row: channel and rsos data of each input img, second row, difference to ground truth channel and rsos
    names = []
    tmp_first = []
    tmp_second = []
    for i, d in enumerate(imgs):
        data = d["data"].abs()
        nx, ny, ns, nc, nt = data.shape
        name = d['name']
        logger.info(f"add: {name}")
        names.extend([name, f"{name} RSOS"])

        # channel data
        tmp_first.append(data[:, :, ns // 2, channel_idx, 0].abs())
        rsos = root_sum_of_squares(data, dim_channel=-2)[:, :, ns //2, 0] / 2
        tmp_first.append(rsos)

        diff = data.abs() - gt.abs()
        diff[~bet] = 0
        tmp_second.append(diff[:, :, ns // 2, channel_idx, 0])
        logger.info(f"Mean Error: {torch.mean(torch.abs(diff))}, l2: {torch.linalg.norm(diff)}")

        diff_rsos = rsos.abs() - gt_rsos.abs()
        diff_rsos[~bet[..., ns // 2, 0, 0]] = 0
        tmp_second.append(diff_rsos)
        logger.info(f"Mean Error: {torch.mean(torch.abs(diff_rsos))}, l2: {torch.linalg.norm(diff_rsos)}")
    plots  = [tmp_first, tmp_second]
    del imgs

    fig = psub.make_subplots(
        rows=len(plots), cols=len(plots[0]),
        column_titles=names,
        # row_titles=[f"Data", "Difference"],
        vertical_spacing=0.008, horizontal_spacing=0.005,
        shared_xaxes=True, shared_yaxes=True
    )

    for i, r in enumerate(plots):
        for ic, c in enumerate(r):
            fig.add_trace(
                go.Heatmap(
                    z=c, transpose=False,
                    showscale=False,
                    coloraxis="coloraxis" if i==0 else "coloraxis2",
                ),
                row=1+i, col=1 + ic
            )
            xaxis = fig.data[-1].xaxis
            fig.update_xaxes(visible=False, row=1+i, col=1 + ic)
            fig.update_yaxes(visible=False, row=1+i, col=1 + ic, scaleanchor=xaxis)

    fig.update_layout(
        width=1000,
        height=300,
        margin=dict(t=25, b=20, l=20, r=2),
        coloraxis=dict(
            colorscale="Inferno",
            cmin=0.0, cmax=c_max,
            colorbar=dict(
                x=1.01, y=0.5, len=0.5, thickness=14,
                title=dict(text=f"Signal [a.u.]", side="right"),
                xanchor="left", yanchor="bottom",
            )
        ),
        coloraxis2=dict(
            colorscale="Balance",
            cmin=-0.1*c_max, cmax=0.1*c_max,
            colorbar=dict(
                x=1.01, y=0.0, len=0.5, thickness=14,
                title=dict(text=f"Difference [a.u.]", side="right"),
                xanchor="left", yanchor="bottom",
            )
        )
    )
    fn = path.joinpath(f"recon_batching_comparison").with_suffix(".html")
    logger.info(f"write file: {fn}")
    fig.write_html(fn)
    for suff in [".pdf", ".png"]:
        fn = fn.with_suffix(suff)
        logger.info(f"write file: {fn}")
        fig.write_image(fn)


def plot_svd_comp():
    path = plib.Path(
        get_test_result_output_dir("reconstruction_comparisons", mode=ResultMode.EXPERIMENT)).absolute().joinpath(
        "rsvds")
    f_t = path.joinpath("times").with_suffix(".json")

    _, bet = load_data()
    bet =  bet.to(torch.bool).permute(1, 0, 2)

    gt = torch_load(path.joinpath(f"recon_comparison_gt").with_suffix(".pt"))
    gt = gt[:, :, gt.shape[2] // 2]
    gt_rsos = root_sum_of_squares(gt, dim_channel=-2)

    us = torch_load(path.joinpath(f"recon_comparison_us").with_suffix(".pt"))
    imgs = [
        {"name": "gt", "data": gt},
        {"name": "us", "data": us[:, :, us.shape[2] // 2]}
    ]
    for i, n in enumerate(list(NullspaceAlgorithm)):
        d = torch_load(path.joinpath(f"recon_comparison_svd-{n.name}".lower()).with_suffix(".pt"))
        imgs.append({"name": n.name, "data": d})
    bet = bet[:, :, bet.shape[2]//2].unsqueeze(-1).unsqueeze(-1).expand_as(imgs[-1]["data"])
    channel_idx = [20, 30]
    gt_max = gt_rsos[..., channel_idx, 0].abs().max().item() * 0.6
    gt_c_max = gt[..., channel_idx, 0].abs().max().item() * 0.6

    fig = psub.make_subplots(
        rows=len(channel_idx)+2, cols=len(imgs),
        column_titles=[n["name"] for n in imgs],
        row_titles=[f"Channel {c+1}" for c in channel_idx] + ["RSOS", "DIFF"],
        vertical_spacing=0.01, horizontal_spacing=0.01,
        shared_xaxes=True, shared_yaxes=True,
    )
    error = []
    names = []
    for i, img in enumerate(imgs):
        p = img["data"][..., 0]
        for ic, c in enumerate(channel_idx):
            fig.add_trace(
                go.Heatmap(
                    z=p[:, :, c].abs(), transpose=False,
                    showscale=False,
                    colorscale="Inferno",
                    zmin=0,
                    zmax=gt_c_max
                ),
                row=1+ic, col=1 + i
            )
        rsos = root_sum_of_squares(p, dim_channel=-1).abs()
        diff = rsos - gt_rsos[..., 0]
        logger.info(f"Mean Error {img['name']}: {torch.mean(torch.abs(diff))}")
        error.append(torch.mean(torch.abs(diff)))
        names.append(img["name"])
        for l, r in enumerate([rsos, diff]):
            fig.add_trace(
                go.Heatmap(
                    z=r, transpose=False,
                    showscale=False,
                    colorscale="Inferno" if l ==0 else "Balance",
                    zmin=0 if l == 0 else -0.05 * gt_max,
                    zmax=gt_max if l == 0 else 0.05 * gt_max,
                ),
                row=len(channel_idx) + 1 + l, col=1 + i
            )

    fig.update_yaxes(visible=False)
    fig.update_xaxes(visible=False)
    fig.update_layout(
        width=1000,
        height=650,
        margin=dict(t=25, b=55, l=65, r=5),
    )
    fn = path.joinpath(f"recon_svd_comparison").with_suffix(".html")
    logger.info(f"write file: {fn}")
    fig.write_html(fn)
    for suff in [".pdf", ".png"]:
        fn = fn.with_suffix(suff)
        logger.info(f"write file: {fn}")
        fig.write_image(fn)
    logger.info(f"load file: {f_t}")
    with open(f_t, "r") as json_file:
        times = json.load(json_file)
    t = [0, 0]
    for tt in times:
        t. append(tt["time"])

    cmap = plc.sample_colorscale("Inferno", 2, 0.2, 0.4)
    fig = psub.make_subplots(
        rows=2, cols=1,
        row_titles=["Mean Error", "Time [s]"],
        shared_xaxes=True,
    )
    for i, m in enumerate([error, t]):
        fig.add_trace(
            go.Bar(x=names, y=m, showlegend=False, marker=dict(color=cmap[i])), row=1+i, col=1
        )

    fig.update_layout(
        width=1000,
        height=300,
        margin=dict(t=25, b=55, l=65, r=5),
    )
    fn = path.joinpath(f"recon_svd_comparison_metrics").with_suffix(".html")
    logger.info(f"write file: {fn}")
    fig.write_html(fn)
    for suff in [".pdf", ".png"]:
        fn = fn.with_suffix(suff)
        logger.info(f"write file: {fn}")
        fig.write_image(fn)


def df_batching():
    path = plib.Path(
        get_test_result_output_dir("reconstruction_comparisons", mode=ResultMode.EXPERIMENT)).absolute().joinpath(
        "batching")

    _, bet = load_data()
    bet = bet.to(torch.bool).permute(1, 0, 2).unsqueeze(-1)

    gt = torch_load(path.joinpath(f"recon_batching_comparison_gt").with_suffix(".pt"))
    gt_rsos = root_sum_of_squares(gt, dim_channel=-2)
    gt_rsos[~bet.expand_as(gt_rsos)] = 0

    stats = []
    for i, t in enumerate(["us", "cb-True", "cb-False"]):
        d = torch_load(path.joinpath(f"recon_batching_comparison_{t}").with_suffix(".pt"))
        rsos = root_sum_of_squares(d, dim_channel=-2)
        rsos[~bet.expand_as(rsos)] = 0
        name = t
        nmse = calc_nmse(gt_rsos, rsos)
        psnr = calc_psnr(gt_rsos, rsos)
        ssim = calc_ssim(gt_rsos.permute(3, 2, 1, 0), rsos.permute(3, 2, 1, 0))
        stats.append({"name": name, "nmse": nmse, "psnr": psnr, "ssim": ssim})

    df =  pl.DataFrame(stats)
    logger.info(df)
    fn = path.joinpath("errors").with_suffix(".json")
    logger.info(f"write file: {fn}")
    df.write_ndjson(fn)


def df_svds():
    path = plib.Path(
        get_test_result_output_dir("reconstruction_comparisons", mode=ResultMode.EXPERIMENT)).absolute().joinpath(
        "rsvds")
    logger.info(f"set path: {path}")

    _, bet = load_data()
    bet = bet.to(torch.bool).permute(1, 0, 2)[:, :, bet.shape[2] //2, None]
    fn = path.joinpath(f"recon_comparison_gt").with_suffix(".pt")
    logger.info(f"Load file: {fn}")
    gt = torch_load(fn)
    gt_rsos = root_sum_of_squares(gt, dim_channel=-2)[:, :, gt.shape[2] // 2]
    gt_rsos[~bet.expand_as(gt_rsos)] = 0

    stats = []
    for i, nsa in enumerate(list(NullspaceAlgorithm)):
        logger.info(f"Process Algorithm: {nsa.name}")
        fn = path.joinpath(f"recon_comparison_svd-{nsa.name}".lower()).with_suffix(".pt")
        d = torch_load(fn)

        rsos = root_sum_of_squares(d, dim_channel=-2)
        rsos[~bet.expand_as(rsos)] = 0
        name = nsa.name.lower()
        nmse = calc_nmse(gt_rsos, rsos)
        psnr = calc_psnr(gt_rsos, rsos)
        ssim = calc_ssim(gt_rsos.permute(2, 0, 1), rsos.permute(2, 0, 1))
        stats.append({"name": name, "nmse": nmse, "psnr": psnr, "ssim": ssim})

    df =  pl.DataFrame(stats)
    logger.info(df)
    fn = path.joinpath(f"recon_comparison_svd-stats").with_suffix(".json")
    df.write_ndjson(fn)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    # compare_batching()
    # plot_batching_comp()
    df_batching()

    # compare_variants()
    # plot_svd_comp()
    df_svds()
