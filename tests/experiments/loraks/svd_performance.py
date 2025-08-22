import logging
import pathlib as plib
from enum import Enum, auto
from timeit import Timer
import sys

import torch
import polars as pl
import tqdm

import plotly.graph_objects as go
import plotly.subplots as psub
import plotly.colors as plc

from pymritools.utils.algorithms import randomized_svd, subspace_orbit_randomized_svd, rand_qlp, randomized_nullspace

p_tests = plib.Path(__file__).absolute().parent.parent.parent.parent
sys.path.append(p_tests.as_posix())

from tests.utils import get_test_result_output_dir, ResultMode

logger = logging.getLogger(__name__)


class SVDType(Enum):
    EIGH = auto()
    SVD = auto()
    LRSVD = auto()
    RSVD = auto()
    SORSVD = auto()
    RANDQLP = auto()
    RANDNS = auto()


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
        case SVDType.EIGH:
            _, _ = torch.linalg.eigh(matrix.mH @ matrix)
        case SVDType.RANDQLP:
            _, _, _ = rand_qlp(matrix=matrix)
        case SVDType.RANDNS:
            m_dim = min(matrix.shape[-2:])
            _ = randomized_nullspace(matrix=matrix, nullity=m_dim - rank, oversample=oversampling)


def compute():
    path = plib.Path(get_test_result_output_dir("svd_performance", mode=ResultMode.EXPERIMENT))

    # do loops for different data sizes
    # ms_xy = torch.linspace(50 * 50, 100 * 100, 2).to(torch.int)
    nb = 5**2
    nc = 32
    ne = 4
    mxy = torch.tensor([224 * 192]) * 2
    logger.info(f"Set Spatial dimension size: {mxy[0]}")

    ms_ce = torch.linspace(200, 550, 20).to(torch.int)
    oversampling = 10
    num_timer_runs = 3

    meas = []
    device = torch.device("cuda:0")
    logger.info(f"set device: {device}")
    # for i, mxy in enumerate(ms_xy):
    #     logger.info(f"Processing Matrix Size XY : {i+1} / {ms_xy.shape[0]}")

    for g in tqdm.trange(ms_ce.shape[0], desc="Processing Matrix sizes"):
        mce = ms_ce[g] * 2
        # logger.info(f"\t\tProcessing Matrix Size CE : {g+1} / {ms_ce.shape[0]}")
        m = min(mxy.item(), mce)
        rank = max(m // 10, 10)

        from tests.utils import create_random_matrix
        matrix = create_random_matrix(mxy.item(), mce.item(), dtype=torch.float32, device=device)

        # matrix = torch.randn((mxy, mce), device=device)
        # matrix += torch.full_like(matrix, 1)
        # matrix += torch.randn_like(matrix)**2

        # SVDS
        for svd_type in list(SVDType):
            if g % 2 == 0:
                if svd_type == SVDType.SVD:
                    continue
            if g > 10:
                if svd_type == SVDType.SVD :
                    continue
            if g > 15:
                if svd_type == SVDType.RANDQLP or svd_type == SVDType.RANDNS:
                    continue
            # init run and measure memory
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated(device)
            torch.cuda.synchronize(device)
            mem_start = torch.cuda.max_memory_allocated(device)

            process_svd(matrix=matrix, svd_type=svd_type, rank=rank, oversampling=oversampling, power_iterations=2)

            mem_end = torch.cuda.max_memory_allocated(device)

            # timing runs
            t = Timer(
                stmt="process_svd(matrix=matrix, svd_type=svd_type, rank=rank)",
                setup="from __main__ import process_svd",
                globals={"matrix":matrix, "svd_type":svd_type, "rank":rank}
            )
            t_processing = t.timeit(num_timer_runs) / num_timer_runs
            mem_processing = (mem_end - mem_start) / 1024 / 1024
            meas.append({
                "Mode": svd_type.name, "Time": t_processing, "Memory": mem_processing,
                "mxy": mxy.item(), "mce": mce
            })
            torch.cuda.empty_cache()
        df = pl.DataFrame(meas)
        fn = path.joinpath("results_df").with_suffix(".json")
        logger.info(f"Writing to {fn}")
        df.write_ndjson(fn)


def plot(colorscale: str = "Inferno", cmin: float = 0.1, cmax: float = 0.9):
    path = plib.Path(get_test_result_output_dir("svd_performance", mode=ResultMode.EXPERIMENT))
    fn = path.joinpath("results_df").with_suffix(".json")
    logger.info(f"Loading {fn}")
    df = pl.read_ndjson(fn)
    # df = df.filter((pl.col("mce") != 7969) & (pl.col("mce") != 9238) & (pl.col("mce") != 7715) & (pl.col("mce") != 9492))

    fig = psub.make_subplots(
        rows=2, cols=2,
        shared_xaxes=True, shared_yaxes=False,
        x_title=f"Matrix Size (short axis)",
        vertical_spacing=0.03, horizontal_spacing=0.05
    )

    cmap = plc.sample_colorscale(colorscale, len(list(SVDType)), cmin, cmax)
    units = ["[s]", "[MB]"]

    for si, svd_type in enumerate(list(SVDType)):
        for it, t in enumerate(["Time", "Memory"]):
            df_tmp = df.filter(pl.col("Mode") == svd_type.name)
            for c in range(2):
                if (svd_type == SVDType.SVD or svd_type == SVDType.RANDQLP or svd_type == SVDType.RANDNS) and c > 0:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=df_tmp["mce"], y=df_tmp[t],
                        marker=dict(color=cmap[si]),
                        name=svd_type.name if svd_type != SVDType.SORSVD else "SOR-SVD",
                        legendgroup=si,
                        showlegend=it == 0 and c == 0,
                        mode="markers+lines"
                    ),
                    row=1 + it, col=1 + c
                )
                if si == 0:
                    fig.update_yaxes(row=1 + it, col=1 + c, title=f"{t} {units[it]}")
    fig.update_layout(
        width=900,
        height=400,
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
    compute()
    plot(colorscale="Turbo", cmin=0.0, cmax=1.0)
