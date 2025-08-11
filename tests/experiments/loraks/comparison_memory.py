import logging
import pathlib as plib
import sys
import torch
import polars as pl
import numpy as np
from scipy.io import savemat

from pymritools.recon.loraks.ac_loraks import AcLoraksOptions, SolverType, ComputationType
from pymritools.recon.loraks.loraks import LoraksImplementation, OperatorType, RankReduction, RankReductionMethod, \
    Loraks

import plotly.graph_objects as go
import plotly.colors as plc

p_tests = plib.Path(__file__).absolute().parent.parent.parent.parent
sys.path.append(p_tests.as_posix())
from tests.utils import get_test_result_output_dir, ResultMode
from tests.experiments.loraks.utils import (
    prep_k_space, unprep_k_space, create_phantom,
    run_ac_loraks_matlab_script, run_ac_loraks_torch_script
)

logger = logging.getLogger(__name__)


# --  Computations
def torch_loraks_run(
        k: torch.Tensor, device: torch.device, max_num_iter: int,
        regularization_lambda: float, rank: int):
    # set up AC loraks
    opts = AcLoraksOptions(
        loraks_type=LoraksImplementation.AC_LORAKS, loraks_neighborhood_size=5,
        loraks_matrix_type=OperatorType.S, rank=RankReduction(method=RankReductionMethod.HARD_CUTOFF, value=rank),
        regularization_lambda=regularization_lambda, max_num_iter=max_num_iter, device=device,
        computation_type=ComputationType.FFT, solver_type=SolverType.LEASTSQUARES
    )
    ac_loraks = Loraks.create(options=opts)

    # prep data
    k_in, in_shape, padding, batch_channel_idx = prep_k_space(k=k.squeeze().unsqueeze(2), batch_size_channels=-1)

    recon = ac_loraks.reconstruct(k_in)

    # unprep
    k_recon = unprep_k_space(recon, padding=padding, batch_idx=batch_channel_idx, input_shape=in_shape)
    return k_recon


def recon_ac_loraks_gpu(
        k: torch.Tensor, device: torch.device,
        rank: int, regularization_lambda: float,
        max_num_iter: int = 30,
):
    logger.info(f"Set device: {device}")

    if not device.type == "cuda":
        msg = f"called GPU run with wrong device: {device}"
        logger.error(msg)
        raise AttributeError(msg)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats(device)
    torch.cuda.synchronize(device)
    mem_start = torch.cuda.max_memory_allocated(device)

    k_recon = torch_loraks_run(
        k=k, device=device, rank=rank, regularization_lambda=regularization_lambda, max_num_iter=max_num_iter
    )

    mem_end = torch.cuda.max_memory_allocated(device)

    return (mem_end - mem_start) / 1024 / 1024


def recon_ac_loraks_cpu(
        k: torch.Tensor,
        rank: int, regularization_lambda: float,
        max_num_iter: int = 30, ):
    # we need to save the tensor and pass the settings
    device = torch.device("cpu")
    logger.info(f"Set device: {device}")

    # use a sub-path
    path = plib.Path(get_test_result_output_dir("comparison_memory", mode=ResultMode.EXPERIMENT)).joinpath("tmp")
    path.mkdir(parents=True, exist_ok=True)

    fn = path.joinpath("tmp_data").with_suffix(".pt")
    logger.info(f"Writing data {fn}")
    torch.save(k.unsqueeze(2), fn)

    # run torch script as cmd line script using valgrind
    logger.info(f"Calling Torch profile routine")
    result = run_ac_loraks_torch_script(
        profile_memory=True, data_dir=fn.as_posix(), script_args=(rank, regularization_lambda, max_num_iter)
    )
    return result


def recon_ac_loraks_matlab(
        k: torch.Tensor,
        rank: int,
        regularization_lambda: float,
        max_num_iter: int = 30):
    matlab_path = plib.Path(__file__).absolute().parent.joinpath("matlab")
    logger.info(f"Set MATLAB data path: {matlab_path}")

    # build matlab data - for joint echo reconstruction, we just combine channel and echo data
    k = k.view(*k.shape[:2], -1)
    mask = (k.abs() > 1e-12)

    matlab_input_file = matlab_path.joinpath("input").with_suffix(".mat")
    logger.info(f"Save MATLAB input data: {matlab_input_file}")
    mat_data = {
        "k_data": k.numpy(), "mask": mask.numpy(),
        "rank": rank, "lambda": regularization_lambda, "max_num_iter": max_num_iter,
        "num_timer_runs": 0, "num_warmup_runs": 1,
    }
    savemat(matlab_input_file, mat_data)

    logger.info("Calling MATLAB routine")
    result = run_ac_loraks_matlab_script(profile_memory=True, capture_output=True)

    return result


def write_df(meas, path):
    df = pl.DataFrame(meas)
    fn = path.joinpath("results_df_latest").with_suffix(".json")
    logger.info(f"Writing to {fn}")
    df.write_ndjson(fn)


def compute():
    # get path
    path_out = plib.Path(get_test_result_output_dir("comparison_memory", mode=ResultMode.EXPERIMENT))
    # do loops for different data sizes
    ms_xy = torch.linspace(100*100, 240*240, 5)
    ncs = torch.arange(4, 33, 8)
    nes = torch.arange(2, 6, 2)
    # ms_xy = torch.tensor([10000])
    # ncs = torch.tensor([4])
    # nes = torch.tensor([2])

    # set some params
    regularization_lambda = 0.0
    # need few runs, not interested in the result but just the memory allocation
    max_num_iter = 5

    # set up list for data-collection
    meas = []

    # cycle through different matrix sizes
    for i, mxy in enumerate(ms_xy):
        logger.info(f"Processing Matrix Size XY : {i + 1} / {ms_xy.shape[0]}")
        # extract k-space side from matrix
        nx = torch.sqrt(mxy).to(torch.int)
        ny = (mxy / nx).to(torch.int)
        # cycle through different number of channels and echoes

        for g, nc in enumerate(ncs):
            logger.info(f"__ nc: {g + 1} / {ncs.shape[0]}")

            for h, ne in enumerate(nes):
                logger.info(f"__ ne: {h + 1} / {nes.shape[0]}")
                mce = ne * nc * 5 ** 2

                rank = max(15, min(mce.item(), mxy.item()) // 10)

                # prepare the data
                _, k_us = create_phantom(shape_xyct=(nx.item(), ny.item(), nc.item(), ne.item()), acc=3,
                                         ac_lines=ny.item() // 6.5)

                # logger.info(f"__ Processing Torch GPU\n")
                # if not torch.cuda.is_available():
                #     msg = "No GPU device available, skipping GPU benchmark"
                #     logger.warning(msg)
                # else:
                #     try:
                #         mem_usage = recon_ac_loraks_gpu(
                #             k=k_us.clone(), device=torch.device("cuda:0"),
                #             rank=rank, regularization_lambda=regularization_lambda,
                #             max_num_iter=max_num_iter
                #         )
                #     except Exception as e:
                #         logger.warning(e)
                #         mem_usage = "Maxed Out"
                #     meas.append({
                #         "Mode": "torch", "Device": "GPU", "mxy": mxy, "mce": mce, "Memory": mem_usage
                #     })

                logger.info(f"__ Processing Torch CPU\n")
                write_df(meas, path_out)

                mem_usage = recon_ac_loraks_cpu(
                    k=k_us.clone(),
                    rank=rank, regularization_lambda=regularization_lambda,
                    max_num_iter=max_num_iter
                )

                meas.append({
                    "Mode": "torch", "Device": "CPU", "mxy": mxy, "mce": mce, "Memory": mem_usage
                })

                df = pl.DataFrame(meas)
                fn = path_out.joinpath("results_df_latest").with_suffix(".json")
                logger.info(f"Writing to {fn}")
                df.write_ndjson(fn)

                logger.info(f"__ Processing Matlab CPU\n")
                mem_usage = recon_ac_loraks_matlab(
                    k=k_us, rank=rank, regularization_lambda=regularization_lambda,
                    max_num_iter=max_num_iter
                )
                meas.append({
                    "Mode": "matlab", "Device": "CPU", "mxy": mxy, "mce": mce, "Memory": mem_usage
                })

                df = pl.DataFrame(meas)
                fn = path_out.joinpath("results_df_latest").with_suffix(".json")
                logger.info(f"Writing to {fn}")
                df.write_ndjson(fn)

# --- PLOTTING
def load_df(path: plib.Path, name: str = ""):
    if name and not name.startswith("_"):
        name = f"_{name}"
    fn = path.joinpath(f"results_df{name}").with_suffix(".json")
    logger.info(f"Loading from {fn}")

    df = pl.read_ndjson(fn)
    df = df.with_columns(
        pl.col('mxy').cast(pl.Int32),
        pl.col('mce').cast(pl.Int32)
    )

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
    ])
    return df_plot.sort(by=["mxy", "plot_mode"], descending=False)


def plot_table_per_mode():
    path_out = plib.Path(get_test_result_output_dir("comparison_memory", mode=ResultMode.EXPERIMENT))

    # df_gpu = load_df(path=path_out, name="latest")
    # df_cpu = load_df(path=path_out, name="latest_cpu")
    # df = pl.concat([df_gpu, df_cpu])
    df = load_df(path=path_out, name="latest")

    # get df time normalized by matlab time
    # create lookup reference
    reference_values = (
        df.filter(pl.col("Mode") == "matlab").group_by(
            ["mxy", "mce", "plot_size"]
        ).agg(pl.col("Memory").first().alias("ref_mem"))
    )
    df_norm = (
        df.join(reference_values, on=["mxy", "mce", "plot_size"], how="left").
        with_columns(
            (pl.col("Memory") / pl.col("ref_mem")).alias("Memory Savings")
        )
        .drop("ref_mem")
    )
    df_norm = df_norm.filter(
        (pl.col("Memory").is_not_null()) &
        (pl.col("mce") > 400) &
        (pl.col("mce") < 2600)
    )
    df_norm = df_norm.with_columns(
        pl.col("Memory") / 1000
    )

    df_norm.write_ndjson(
        path_out.joinpath("df_performance_benchmark").with_suffix(".json")
    )

    for l, t in enumerate(["Memory", "Memory Savings"]):
        fig = go.Figure()

        ann_m = []
        zmin = 0
        for i, m in enumerate(df_norm["plot_mode"].unique().sort(descending=True)):
            cm = "Inferno"
            # if i == 0:
            #     cm = "Purples"
            # if i == 0:
            #     cm = "deep_r"
            # else:
            #     # continue
            #     cm = "Greys"
            ddd = df_norm.filter(pl.col("plot_mode") == m)
            mxys = ddd["mxy"].unique().sort(descending=False).to_numpy()
            mces = ddd["mce"].unique().sort(descending=False).to_numpy()
            # mmax = ddd[t].max() * 0.85
            mmax = 0.5 if l == 1 else 46
            # build tensor to ensure order
            mem = torch.zeros((mxys.shape[0], mces.shape[0]))
            for g, mxy in enumerate(mxys):
                for h, mce in enumerate(mces):
                    df_tmp = df_norm.filter(
                        (pl.col("plot_mode") == m) & (pl.col("mxy") == mxy) & (pl.col("mce") == mce)
                    )
                    if len(df_tmp) < 1:
                        logger.warning(f"Found no entry for mode {m}, and {mxy} x {mce} combination!")
                        continue
                    zm = df_tmp[t]
                    if len(zm) > 1:
                        logger.warning("Found more than one acceleration value for mode, mxy, mce combination!")
                    if zm[0] is None:
                        logger.warning(f"Found null acceleration value for mode, {mxy}, {mce} combination!")
                    mem[g, h] = zm[0] if zm[0] > 1e-3 else torch.nan
            fig.add_trace(
                go.Surface(
                    x=mxys * 2, y=mces * 2,
                    z=mem.mT.numpy(), opacity=0.82,
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
            (pl.col("mxy") == 33800) & (pl.col("mce") == 1400)
        ).drop(["Mode", "plot_mode", "plot_size"])

        df_pt = df_pts.filter(pl.col("Device").is_in(["GPU"]))
        # fig.add_trace(
        #     go.Scatter3d(
        #         x=df_pt["mxy"] * 2, y=df_pt["mce"] * 2, z=df_pt[t] , marker=dict(color="red", symbol="x", size=3),
        #         showlegend=False
        #     )
        # )
        # val = df_pt[t].unique().sort(descending=True)[0]
        # d = dict(
        #     text=f"{val:.1f} {['GB', 'Saving'][l]}", x=70800 * 2, y=1400 * 2, z=val * 1.2,
        #     showarrow=False, yanchor="bottom", font=dict(color="red")
        # )
        # ann_m.append(d)
        scene = dict(
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=-0.2),
                eye=dict(x=-2.1, y=1.05, z=0.8)
            ),
            # annotations=ann_m,
            xaxis=dict(
                # title="Matrix size N<sub>x</sub> x N<sub>y</sub>",
                title="",
                tickmode="array",
                tickvals=np.arange(40000, 121000, 40000),
                ticktext=[f"{val}k" for val in np.arange(40, 121, 40)]
            ),
            # yaxis=dict(title="Matrix size N<sub>c</sub> x N<sub>e</sub> x N<sub>b</sub>",),
            yaxis=dict(title=""),
            # zaxis=dict(title="Acceleration     " if i==0 else "Memory [GB]     ")
            zaxis=dict(title="", range=(0, mmax))
        )

        fig.update_scenes(
            scene
        )
        fig.update_layout(
            width=450, height=450,
            margin=dict(t=20, b=20, l=0, r=0)

        )
        fn = path_out.joinpath(f"performance_benchmark_{t}".replace(" ", "-").lower()).with_suffix(".html")
        logger.info(f"write file: {fn}")
        fig.write_html(fn)
        for suff in [".pdf", ".png"]:
            fn = fn.with_suffix(suff)
            logger.info(f"write file: {fn}")
            fig.write_image(fn)


def print_table():
    path_out = plib.Path(get_test_result_output_dir("comparison_memory", mode=ResultMode.EXPERIMENT))
    df = pl.read_ndjson(path_out.joinpath("df_performance_benchmark").with_suffix(".json"))

    logger.info(df.filter(pl.col("Device") == "GPU")["Memory Savings"].mean())
    logger.info(df.filter((pl.col("Device") == "CPU") & (pl.col("Mode") == "torch"))["Memory Savings"].mean())

    logger.info(df["mxy"].unique())
    logger.info(df["mce"].unique())
    df = df.filter(
        (pl.col("mxy").is_in([48000, 78400])) &
        (pl.col("mce").is_in([800, 2500]))
    ).sort(by=["mxy", "mce", "Mode", "Device"], descending=False)
    logger.info(df)
    df.write_ndjson(path_out.joinpath("df_performance_benchmark_filtered").with_suffix(".json"))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    # compute()
    plot_table_per_mode()
    print_table()
