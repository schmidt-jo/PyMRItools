import pathlib as plib
import sys
import logging
from timeit import Timer
import polars as pl
import numpy as np

import torch
import plotly.graph_objects as go
import plotly.subplots as psub
import plotly.colors as plc

from pymritools.recon.loraks.loraks import Loraks
from pymritools.recon.loraks.ac_loraks import AcLoraksOptions, ComputationType, SolverType
from pymritools.recon.loraks.loraks import LoraksImplementation, OperatorType, RankReduction, RankReductionMethod
from scipy.io import loadmat, savemat

p_tests = plib.Path(__file__).absolute().parent.parent.parent.parent
sys.path.append(p_tests.as_posix())
from tests.utils import get_test_result_output_dir, ResultMode
from tests.experiments.loraks.utils import prep_k_space, unprep_k_space, create_phantom, run_ac_loraks_matlab_script

logger = logging.getLogger(__name__)


# --- COMPUTATIONS

### We need:
## 1) Provide the data a) virtual data of different sizes, b) real examples
## 2) Set up AC LORAKS leastsquares for GPU and CPU and matlab (same algorithm) for joint-echo -recon
## 3) Time computations

def compute():
    # get path
    path_out = plib.Path(get_test_result_output_dir("comparison_speed", mode=ResultMode.EXPERIMENT))
    # path_out.mkdir(parents=True, exist_ok=True)

    # do loops for different data sizes
    # ms_xy = torch.linspace(100*100, 280*280, 10)
    # ncs = torch.arange(4, 37, 4)
    # nes = torch.arange(2, 6, 1)
    ms_xy = torch.tensor([10000])
    ncs = torch.tensor([4])
    nes = torch.tensor([2])

    # some settings
    regularization_lambda = 0.0
    # use few passes only
    max_num_iter = 5
    # set number of benchmarking runs
    num_warmup_runs = 2
    num_timer_runs = 3

    meas = []

    for i, mxy in enumerate(ms_xy):
        logger.info(f"Processing Matrix Size XY : {i + 1} / {ms_xy.shape[0]}")
        nx = torch.sqrt(mxy).to(torch.int)
        ny = (mxy / nx).to(torch.int)
        for g, nc in enumerate(ncs):
            logger.info(f"__ nc: {g + 1} / {ncs.shape[0]}")
            for h, ne in enumerate(nes):
                logger.info(f"__ ne: {h + 1} / {nes.shape[0]}")
                mce = ne * nc * 5 ** 2
                rank = max(15, min(mce.item(), mxy.item()) // 10)

                # prepare the data
                _, k_us = create_phantom(
                    shape_xyct=(nx.item(), ny.item(), nc.item(), ne.item()), acc=3, ac_lines=ny.item() // 6.5
                )

                logger.info(f"__ Processing Torch GPU\n")
                if not torch.cuda.is_available():
                    msg = "No GPU device available, skipping GPU benchmark"
                    logger.warning(msg)
                else:
                    try:
                        time = recon_ac_loraks(
                            k=k_us.clone(), device=torch.device("cuda:0"),
                            rank=rank, regularization_lambda=regularization_lambda,
                            max_num_iter=max_num_iter, num_warmup_runs=num_warmup_runs, num_timer_runs=num_timer_runs
                        )
                        meas.append({
                            "Mode": "torch", "Device": "GPU", "Time": time,
                            "mxy": mxy, "mce": mce
                        })
                    except Exception as e:
                        logger.warning(e)
                        meas.append({
                            "Mode": "torch", "Device": "GPU", "Time": None,
                            "mxy": mxy, "mce": mce
                        })

                logger.info(f"__ Processing Torch CPU\n")
                try:
                    time = recon_ac_loraks(
                        k=k_us.clone(), device=torch.device("cpu"),
                        rank=rank, regularization_lambda=regularization_lambda,
                        max_num_iter=max_num_iter, num_warmup_runs=num_warmup_runs, num_timer_runs=num_timer_runs
                    )
                    meas.append({
                        "Mode": "torch", "Device": "CPU", "Time": time,
                        "mxy": mxy, "mce": mce
                    })
                except Exception as e:
                    logger.warning(e)
                    meas.append({
                        "Mode": "torch", "Device": "CPU", "Time": None,
                        "mxy": mxy, "mce": mce
                    })

                logger.info(f"__ Processing Matlab CPU\n")
                time = recon_ac_loraks_matlab(
                    k=k_us, rank=rank, regularization_lambda=regularization_lambda,
                    max_num_iter=max_num_iter, num_warmup_runs=num_warmup_runs, num_timer_runs=num_timer_runs
                )
                meas.append({
                    "Mode": "matlab", "Device": "CPU", "Time": time,
                    "mxy": mxy, "mce": mce
                })

        df = pl.DataFrame(meas)
        fn = path_out.joinpath("results_df_latest").with_suffix(".json")
        logger.info(f"Writing to {fn}")
        df.write_ndjson(fn)


def recon_ac_loraks_matlab(
        k: torch.Tensor,
        rank: int,
        regularization_lambda: float,
        max_num_iter: int = 30,
        num_warmup_runs: int = 2,
        num_timer_runs: int = 5):
    path = plib.Path(__file__).absolute().parent.joinpath("matlab")

    # build matlab data - for joint echo reconstruction, we just combine channel and echo data
    k = k.view(*k.shape[:2], -1)
    mask = (k.abs() > 1e-12)

    matlab_input_file = path.joinpath("input").with_suffix(".mat")
    logger.info(f"save matlab input data: {matlab_input_file}")
    matlab_input_data = {
        "k_data": k.numpy(), "mask": mask.numpy(),
        "rank": rank, "lambda": regularization_lambda, "max_num_iter": max_num_iter,
        "num_timer_runs": num_timer_runs, "num_warmup_runs": num_warmup_runs,
    }
    savemat(matlab_input_file, matlab_input_data)

    logger.info("Calling MATLAB routine")
    _ = run_ac_loraks_matlab_script(use_valgrind=False)

    logger.info("Fetching results")
    results = loadmat(path.joinpath("output.mat").as_posix())

    times = torch.from_numpy(results["t"][0])
    t = torch.sum(times) / times.shape[0]
    return t.item()


def loraks_run(
        k: torch.Tensor, device: torch.device,
        rank: int, regularization_lambda: float,
        max_num_iter: int = 30, ):
    # set up AC loraks
    opts = AcLoraksOptions(
        loraks_type=LoraksImplementation.AC_LORAKS, loraks_neighborhood_size=5,
        loraks_matrix_type=OperatorType.S, rank=RankReduction(method=RankReductionMethod.HARD_CUTOFF, value=rank),
        regularization_lambda=regularization_lambda, max_num_iter=max_num_iter, device=device,
        computation_type=ComputationType.FFT, solver_type=SolverType.LEASTSQUARES
    )
    ac_loraks = Loraks.create(options=opts)

    # prep data
    k_in, in_shape, padding, batch_channel_idx = prep_k_space(k=k.squeeze().unsqueeze_(2), batch_size_channels=-1)

    recon = ac_loraks.reconstruct(k_in)

    # unprep
    k_recon = unprep_k_space(recon, padding=padding, batch_idx=batch_channel_idx, input_shape=in_shape)
    return k_recon


def recon_ac_loraks(
        k: torch.Tensor, device: torch.device,
        rank: int, regularization_lambda: float,
        max_num_iter: int = 30,
        num_warmup_runs: int = 2, num_timer_runs: int = 5
):
    logger.info(f"Set device: {device}")

    # timing
    t = Timer(
        stmt="loraks_init_run(k, device, rank, regularization_lambda, max_num_iter)",
        setup="from __main__ import loraks_init_run",
        globals={"k": k, "device": device, "rank": rank, "regularization_lambda": regularization_lambda,
                 "max_num_iter": max_num_iter, }
    )

    # warmup & result
    _ = t.timeit(max(1, num_warmup_runs - 1))

    k_recon = loraks_run(
        k=k, device=device, rank=rank, regularization_lambda=regularization_lambda, max_num_iter=max_num_iter
    )

    # Time Measurement
    t_processing = t.timeit(num_timer_runs) / num_timer_runs

    return t_processing


# --- PLOTTING
def load_df(path: plib.Path, name: str = ""):
    if name and not name.startswith("_"):
        name = f"_{name}"
    fn = path.joinpath(f"results_df{name}").with_suffix(".json")
    logger.info(f"Loading from {fn}")

    df = pl.read_ndjson(fn)
    df = df.with_columns(pl.col('mxy').cast(pl.Int32))

    df_plot = df.with_columns(
        pl.concat_str(
            [pl.col("Mode"), pl.col("Device")],
            separator=" - "
        ).alias("plot_mode")
    )
    df_plot = df_plot.sort(by="mxy", descending=False)

    df_plot = df_plot.with_columns(
        pl.concat_str(
            [pl.col("mxy"), pl.col("mce")],
            separator=" x "
        ).alias("plot_size")
    )
    # average across same matrix sizes but different nc / ne parameters
    df_plot = df_plot.group_by(["Mode", "Device", "mxy", "mce", "plot_mode", "plot_size"]).agg([
        pl.col("Memory").mean().alias("Memory"),
        pl.col("Time").mean().alias("Time")
    ])
    return df_plot.sort(by=["mxy", "plot_mode"], descending=False)


def plot_per_mode(name: str = ""):
    path_out = plib.Path(get_test_result_output_dir("speed_comparison", mode=ResultMode.EXPERIMENT))
    df_plot = load_df(path=path_out, name=name)

    mces = df_plot["mce"].unique().sort(descending=False)
    mxys = df_plot["mxy"].unique().sort(descending=False)

    fig = psub.make_subplots(
        rows=2, cols=2,
        vertical_spacing=0.05, horizontal_spacing=0.12,
        x_title="Implementation",
        shared_xaxes=True,
    )

    # we want to plot time and memory against the implementation type
    # and color code either spatial matrix sizes for a fixed neighborhood matrix size or vice versa
    data = [
        {"fixed": "mxy", "colors": "mce", "data": "Time"}, {"fixed": "mce", "colors": "mxy", "data": "Time"}
    ]
    for i, d in enumerate(data):
        # fix one of the two matrix dims
        fixed_value = df_plot[d["fixed"]].unique().sort(descending=False)
        df_tmp = df_plot.filter(pl.col(d["fixed"]) == fixed_value[-2])
        # extract all matrix sizes for the unfixed dim
        unfixed_values = df_tmp[d["colors"]].unique()
        cmap = plc.sample_colorscale("Inferno", len(unfixed_values))

        # set coloraxis to either of the two depending on the input
        c_ax = "coloraxis" if i < 1 else f"coloraxis2"

        # iterate through unfixed matrix dim
        for g, m in enumerate(unfixed_values):
            df_tmptmp = df_tmp.filter((pl.col(d["colors"]) == m))

            # now we want to plot time and memory consumption in different columns
            units = ["[s]", "[MB]"]
            for it, t in enumerate(["Time", "Memory"]):
                fig.add_trace(
                    go.Bar(
                        x=[s[0].capitalize() + s[1:] for s in df_tmptmp["plot_mode"]], y=df_tmptmp[t],
                        marker=dict(
                            color=cmap[g],
                            coloraxis=c_ax,
                            showscale=True
                        ),
                        showlegend=False,
                        legendgroup=g, name=f"M. size - Nx Ny: {m}"
                    ),
                    row=1 + i, col=1 + it
                )
                fig.update_yaxes(title=f"{t} {units[it]}", row=1 + i, col=1 + it)
        fig.update_layout(
            {c_ax: dict(
                colorscale="Inferno",
                cmin=0.0, cmax=[mces, mxys][i].max(),
                colorbar=dict(
                    x=1.01, y=0.5 - i * 0.5, len=0.5, thickness=14,
                    title=dict(text=f"Matrix size - {['Nc Ne Nb', 'Nx Ny'][i]}", side="right"),
                    xanchor="left", yanchor="bottom",
                )
            )
            }
        )
    fig.update_layout(
        width=1000,
        height=500,
        margin=dict(t=25, b=55, l=65, r=5),
        # legend=dict(
        #     yanchor="bottom",
        #     y=0.0,
        #     xanchor="left",
        #     x=1.01
        # )
    )
    fn = path_out.joinpath(f"comparisons_per_mode").with_suffix(".html")
    logger.info(f"write file: {fn}")
    fig.write_html(fn)
    for suff in [".pdf", ".png"]:
        fn = fn.with_suffix(suff)
        logger.info(f"write file: {fn}")
        fig.write_image(fn)


def plot_per_size(name: str = ""):
    path_out = plib.Path(get_test_result_output_dir("speed_comparison", mode=ResultMode.EXPERIMENT))

    df_plot = load_df(path=path_out, name=name)

    mces = df_plot["mce"].unique().sort(descending=False)
    mxys = df_plot["mxy"].unique().sort(descending=False)
    modes = df_plot["plot_mode"].unique().sort(descending=False).to_list()
    mode_names = [s[0].capitalize() + s[1:] for s in modes]

    df_plot = df_plot.with_columns(
        pl.col('plot_mode').replace(
            dict(zip(
                sorted(df_plot["plot_mode"].unique()),
                range(0, len(df_plot["plot_mode"].unique()))
            ))
        ).cast(pl.Int32).alias("plot_mode_int")
    )

    fig = psub.make_subplots(
        rows=2, cols=2,
        vertical_spacing=0.05, horizontal_spacing=0.02,
        shared_xaxes=True, shared_yaxes=True
    )

    # we want to plot time and memory against the matrix size
    # and color code either spatial matrix sizes for a fixed neighborhood matrix size or vice versa
    data = [
        {"fixed": "mxy", "unfixed": "mce"}, {"fixed": "mce", "unfixed": "mxy"}
    ]

    cmap = plc.sample_colorscale("Inferno", len(modes) + 1, 0.15, 0.95)
    units = ["[s]", "[GB]"]

    for i, d in enumerate(data):
        # fix one of the two matrix dims
        fixed_value = df_plot[d["fixed"]].unique().sort(descending=False)
        df_tmp = df_plot.filter(pl.col(d["fixed"]) == fixed_value[-2])
        for im, m in enumerate(modes):
            for it, t in enumerate(["Time", "Memory"]):
                if it > 0:
                    if im < 2:
                        continue
                    c = 1
                    f = 1e-3
                else:
                    c = 0
                    f = 1
                df_tmptmp = df_tmp.filter(pl.col("plot_mode") == m)
                # plot for unfixed matrix dim, iterate through mode
                fig.add_trace(
                    go.Bar(
                        x=df_tmptmp[d["unfixed"]], y=df_tmptmp[t] * f,
                        marker=dict(
                            color=[cmap[l + c] for l in df_tmptmp["plot_mode_int"]],
                        ),
                        # offsetgroup=df_tmp["plot_mode_int"],
                        showlegend=(i == 0) & (it == 0),
                        legendgroup=im, name=mode_names[im]
                    ),
                    col=1 + i, row=1 + it
                )
                if i == 0:
                    fig.update_yaxes(title=f"{t} {units[it]}", col=1 + i, row=1 + it)
                if it == 1:
                    fig.update_xaxes(
                        title=f"Matrix size - {['N<sub>c</sub> x N<sub>e</sub> x N<sub>b</sub>', 'N<sub>x</sub> N<sub>y</sub>'][i]}",
                        col=1 + i, row=1 + it)

    fig.update_layout(
        width=1000,
        height=500,
        margin=dict(t=25, b=55, l=65, r=5),
        barmode="group"
        # legend=dict(
        #     yanchor="bottom",
        #     y=0.0,
        #     xanchor="left",
        #     x=1.01
        # )
    )
    fn = path_out.joinpath(f"comparisons_per_size").with_suffix(".html")
    logger.info(f"write file: {fn}")
    fig.write_html(fn)
    for suff in [".pdf", ".png"]:
        fn = fn.with_suffix(suff)
        logger.info(f"write file: {fn}")
        fig.write_image(fn)


def table_per_mode(name: str = ""):
    path_out = plib.Path(get_test_result_output_dir("speed_comparison", mode=ResultMode.EXPERIMENT))

    df = load_df(path=path_out, name=name)
    # get df time normalized by matlab time
    # create lookup reference
    reference_values = (
        df.filter(pl.col("Mode") == "matlab").group_by(
            ["mxy", "mce", "plot_size"]
        ).agg(pl.col("Time").first().alias("ref_time"))
    )
    df_norm = (
        df.join(reference_values, on=["mxy", "mce", "plot_size"], how="left").
        with_columns(
            (pl.col("ref_time") / pl.col("Time")).alias("Acceleration")
        )
        .drop("ref_time")
    )

    df_norm.write_ndjson(
        path_out.joinpath("df_performance_benchmark").with_suffix(".json")
    )

    # plot this quickly
    fig = go.Figure()
    fig2 = go.Figure()

    cm = "Inferno"
    text_cmap = plc.sample_colorscale(cm, [0.8, 0.7, 0.1])
    ann_a = []
    ann_m = []
    zmin = 0
    zmax = df_norm["Acceleration"].max() * 0.85
    mmax = df_norm["Memory"].max() * 0.85 * 1e-3
    mxys = df_norm["mxy"].unique().sort(descending=False).to_numpy()
    mces = df_norm["mce"].unique().sort(descending=False).to_numpy()
    for i, m in enumerate(df_norm["plot_mode"].unique().sort()):
        # build tensor to ensure order
        acc = torch.zeros((mxys.shape[0], mces.shape[0]))
        mem = torch.zeros((mxys.shape[0], mces.shape[0]))
        for g, mxy in enumerate(mxys):
            for h, mce in enumerate(mces):
                df_tmp = df_norm.filter(
                    (pl.col("plot_mode") == m) & (pl.col("mxy") == mxy) & (pl.col("mce") == mce)
                )
                za = df_tmp["Acceleration"]
                zm = df_tmp["Memory"]
                if len(za) > 1:
                    logger.warning("Found more than one acceleration value for mode, mxy, mce combination!")
                acc[g, h] = za[0]
                mem[g, h] = zm[0]
        fig.add_trace(
            go.Surface(
                x=mxys * 2, y=mces * 2,
                z=acc.mT.numpy(), opacity=0.82,
                cmin=zmin, cmax=zmax,
                colorscale=cm,
                showscale=i == 0,
                colorbar=dict(
                    titleside="right",
                    title="",
                    # title="Acceleration",
                    thickness=20,
                )
            )
        )
        ann_a.append(dict(
            text=m[0].capitalize() + m[1:], font=dict(color=text_cmap[i]),
            showarrow=False, textangle=-18,
            x=2 * mxys[0], y=2 * (mces[-3] + np.diff(mces)[-2] / 2), z=acc[0, -2] + [3, 5, 5][i]
        ))
        if i == 2:
            fig2.add_trace(
                go.Surface(
                    x=mxys * 2, y=mces * 2,
                    z=mem.mT.numpy() * 1e-3, opacity=0.82,
                    cmin=zmin, cmax=mmax,
                    colorscale=cm, showscale=True,
                    colorbar=dict(
                        titleside="right",
                        # title="Memory [GB]",
                        title="",
                        thickness=20,
                    )
                )
            )
    # add a point for a modern scan sizes
    df_pts = df_norm.filter(
        (pl.col("mxy") == 70800) & (pl.col("mce") == 1600)
    ).drop(["Mode", "plot_mode", "plot_size", "Time"])

    for i, t in enumerate(["Acceleration", "Memory"]):
        df_pt = df_pts.filter(pl.col("Device").is_in(["GPU"]))
        if i == 1:
            f = 1e-3
            ff = 0
        else:
            f = 1
            ff = 0
        fff = fig if i == 0 else fig2
        fff.add_trace(
            go.Scatter3d(
                x=df_pt["mxy"] * 2, y=df_pt["mce"] * 2, z=df_pt[t] * f, marker=dict(color="red", symbol="x", size=3),
                showlegend=False
            )
        )
        val = df_pt[t].unique().sort(descending=True)[ff] * f
        d = dict(
            text=f"{val:.1f} {['', 'GB'][i]}", x=70800 * 2, y=1400 * 2, z=val * 1.5, showarrow=False, yanchor="bottom",
            font=dict(color="red")
        )
        if i == 0:
            ann_a.append(d)
        else:
            ann_m.append(d)
        scene = dict(
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=-0.2),
                eye=dict(x=-2.1, y=1.05, z=0.8)
            ),
            annotations=ann_a if i == 0 else ann_m,
            xaxis=dict(
                # title="Matrix size N<sub>x</sub> x N<sub>y</sub>",
                title="",
                tickmode="array",
                tickvals=np.arange(40000, 121000, 40000),
                ticktext=[f"{val}k" for val in np.arange(40, 121, 40)]
            ),
            # yaxis=dict(title="Matrix size N<sub>c</sub> x N<sub>e</sub> x N<sub>b</sub>",),
            yaxis=dict(title="", ),
            # zaxis=dict(title="Acceleration     " if i==0 else "Memory [GB]     ")
            zaxis=dict(title="")
        )

        fff.update_scenes(
            scene
        )
        fff.update_layout(
            width=450, height=450,
            margin=dict(t=20, b=20, l=0, r=0)

        )
        fn = path_out.joinpath(f"performance_benchmark_{t}".lower()).with_suffix(".html")
        logger.info(f"write file: {fn}")
        fff.write_html(fn)
        for suff in [".pdf", ".png"]:
            fn = fn.with_suffix(suff)
            logger.info(f"write file: {fn}")
            fff.write_image(fn)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    compute()
    # table_per_mode("2025-07-16")
