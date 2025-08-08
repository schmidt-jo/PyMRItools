import logging
import pathlib as plib
import sys
import torch
import polars as pl

from scipy.io import savemat
from torch.profiler import profile, record_function, ProfilerActivity

from pymritools.recon.loraks.ac_loraks import AcLoraksOptions, SolverType, ComputationType
from pymritools.recon.loraks.loraks import LoraksImplementation, OperatorType, RankReduction, RankReductionMethod, \
    Loraks

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


def recon_ac_loraks_cpu_profile(
        k: torch.Tensor,
        rank: int,
        regularization_lambda: float,
        max_num_iter: int = 30):
    """
    This measurement of the peak memory is so bad, I wouldn't even call it an estimate.
    I try to sum allocated and freed memory over all profile events.
    If these events actually contain all allocations in an order sorted by time,
    this could be correct. However, the results just appear to be wildly off.
    No clue what to do.
    """

    # it seems we need that because the profiler still tries to access the GPU, although
    # we say during profiling that we only want to trace the CPU
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.enabled = False

    device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    path = plib.Path(get_test_result_output_dir("comparison_memory", mode=ResultMode.EXPERIMENT)).joinpath("tmp")
    path.mkdir(parents=True, exist_ok=True)

    with profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    ) as prof:
        k_recon = torch_loraks_run(
            k=k, device=device, rank=rank, regularization_lambda=regularization_lambda, max_num_iter=max_num_iter
        )

    events = prof.key_averages()
    accumulated_memory = 0
    max_memory = 0
    for e in events:
        accumulated_memory += e.self_cpu_memory_usage
        max_memory = max(max_memory, accumulated_memory)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.enabled = True
    return max_memory / 1024 ** 2


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


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    compute()
