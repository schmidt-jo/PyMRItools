import logging
import pathlib as plib
from enum import Enum, auto
from timeit import Timer

import torch
import polars as pl
import tqdm

import plotly.graph_objects as go
import plotly.subplots as psub
import plotly.colors as plc

from pymritools.utils.algorithms import randomized_svd, subspace_orbit_randomized_svd
from tests.utils import get_test_result_output_dir, create_phantom, ResultMode
from tests.experiments.loraks.utils import prep_k_space, TorchMemoryTracker

logger = logging.getLogger(__name__)


class SVDType(Enum):
    SVD = auto()
    LRSVD = auto()
    RSVD = auto()
    SORSVD = auto()


def process_svd(matrix: torch.Tensor, svd_type: SVDType, rank: int, oversampling: int = 10, power_iterations: int = 2):
    match svd_type:
        case SVDType.SVD:
            _, _, _ = torch.linalg.svd(matrix, full_matrices=False)
        case SVDType.LRSVD:
            _, _, _ = torch.svd_lowrank(matrix, q=rank+oversampling, niter=power_iterations)
        case SVDType.SORSVD:
            _, _, _ = subspace_orbit_randomized_svd(matrix=matrix, q=rank+oversampling, power_projections=power_iterations)
        case SVDType.RSVD:
            _, _, _ = randomized_svd(matrix=matrix, q=rank+oversampling, power_projections=power_iterations)


def compute():
    path = plib.Path(get_test_result_output_dir("svd_performance", mode=ResultMode.EXPERIMENT))

    # do loops for different data sizes
    # ms_xy = torch.linspace(50 * 50, 100 * 100, 2).to(torch.int)
    nb = 5
    mxy = torch.tensor([300 * 240])
    logger.info(f"Set Spatial dimension size: {mxy[0]}")

    ms_ce = torch.linspace(100, 6000, 40).to(torch.int)
    oversampling = 10
    num_timer_runs = 3

    meas = []
    device = torch.device("cuda:0")
    logger.info(f"set device: {device}")
    # for i, mxy in enumerate(ms_xy):
    #     logger.info(f"Processing Matrix Size XY : {i+1} / {ms_xy.shape[0]}")

    for g in tqdm.trange(ms_ce.shape[0], desc="Processing Matrix sizes"):
        mce = ms_ce[g]
        # logger.info(f"\t\tProcessing Matrix Size CE : {g+1} / {ms_ce.shape[0]}")
        m = min(mxy.item(), mce)
        rank = max(m // 10, 10)

        matrix = torch.randn((mxy, mce), device=device)

        # SVDS
        for svd_type in list(SVDType):
            # init run and measure memory
            mem_track = TorchMemoryTracker(device=device)
            mem_track.start_tracking()
            process_svd(matrix=matrix, svd_type=svd_type, rank=rank, oversampling=oversampling, power_iterations=2)
            mem_track.end_tracking()
            # timing runs
            t = Timer(
                stmt="process_svd(matrix=matrix, svd_type=svd_type, rank=rank)",
                setup="from __main__ import process_svd",
                globals={"matrix":matrix, "svd_type":svd_type, "rank":rank}
            )
            t_processing = t.timeit(num_timer_runs) / num_timer_runs
            mem_processing, _, _ = mem_track.get_memory_usage()
            meas.append({
                "Mode": svd_type.name, "Time": t_processing, "Memory": mem_processing,
                "mxy": mxy.item(), "mce": mce
            })
    df = pl.DataFrame(meas)
    fn = path.joinpath("results_df").with_suffix(".json")
    logger.info(f"Writing to {fn}")
    df.write_ndjson(fn)


def plot():
    path = plib.Path(get_test_result_output_dir("svd_performance", mode=ResultMode.EXPERIMENT))
    fn = path.joinpath("results_df").with_suffix(".json")
    logger.info(f"Loading {fn}")
    df = pl.read_ndjson(fn)
    df = df.filter(pl.col("mce") <= 5500)

    fig = psub.make_subplots(
        rows=2, cols=2,
        shared_xaxes=True, shared_yaxes=False,
        x_title=f"Matrix Size (short axis)",
        vertical_spacing=0.03
    )

    cmap = plc.sample_colorscale("Inferno", len(list(SVDType)), 0.1, 0.9)
    units = ["[s]", "[MB]"]

    for si, svd_type in enumerate(list(SVDType)):
        for it, t in enumerate(["Time", "Memory"]):
            df_tmp = df.filter(pl.col("Mode") == svd_type.name)
            for c in range(2):
                if svd_type == SVDType.SVD and c > 0:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=df_tmp["mce"], y=df_tmp[t],
                        marker=dict(color=cmap[si]),
                        name=svd_type.name,
                        legendgroup=si,
                        showlegend=it == 0 and c == 0
                    ),
                    row=1 + it, col=1 + c
                )
                if si == 0:
                    fig.update_yaxes(row=1 + it, col=1 + c, title=f"{t} {units[it]}")
    fig.update_layout(
        width=1000,
        height=500,
        margin=dict(t=25, b=55, l=65, r=5),
        barmode="group"

    )
    fn = path.joinpath(f"comparison_memory_time").with_suffix(".html")
    logger.info(f"write file: {fn}")
    fig.write_html(fn)
    for suff in [".pdf", ".png"]:
        fn = fn.with_suffix(suff)
        logger.info(f"write file: {fn}")
        fig.write_image(fn)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    plot()
