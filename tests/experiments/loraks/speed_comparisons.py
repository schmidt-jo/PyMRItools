import pathlib as plib
import sys
import logging
from timeit import Timer

import torch
from pymritools.recon.loraks.loraks import Loraks
from pymritools.recon.loraks.ac_loraks import AcLoraksOptions, ComputationType, SolverType
from pymritools.recon.loraks.loraks import LoraksImplementation, OperatorType, RankReduction, RankReductionMethod
from scipy.io import loadmat, savemat

p_tests = plib.Path(__file__).absolute().parent.parent.parent.parent
sys.path.append(p_tests.as_posix())
from tests.utils import get_test_result_output_dir, ResultMode
from tests.experiments.loraks.utils import prep_k_space, unprep_k_space, DataType, create_phantom, run_matlab_script

logger = logging.getLogger(__name__)

### We need:
## 1) Provide the data a) virtual data of different sizes, b) real examples
## 2) Set up AC LORAKS leastsquares for GPU and CPU and matlab (same algorithm) for joint-echo -recon
## 3) Time computations

## 4) Set up P-LORAKS for GPU and CPU and matlab (same algorithm) for joint-echo-recon
## 5) Time computations


def main():
    # get path
    path_out = get_test_result_output_dir("speed_comparison", mode=ResultMode.EXPERIMENT)

    # get the data
    k, k_us = create_phantom(shape_xyct=(160, 140, 4, 2), acc=4, ac_lines=24)
    k_recon = []

    # logger.info(f"__ Processing Matlab CPU\n")
    recon_ac_loraks_matlab(
        k=k_us, rank=50, regularization_lambda=0.0, max_num_iter=30, num_warmup_runs=2, num_timer_runs=5
    )
    # k_recon.append({"mode": "torch", "device": "CPU", "time": time_cpu, "recon_data": k_re_cpu})
    # del k_re_cpu, time_cpu

    logger.info(f"__ Processing Torch GPU\n")
    if not torch.cuda.is_available():
        msg = "No GPU device available, skipping GPU benchmark"
        logger.warning(msg)
    else:
        k_re_gpu, time_gpu, mem_gpu = recon_ac_loraks(
            k=k_us.clone(), device=torch.device("cuda:0"),
            rank=50, regularization_lambda=0.0, max_num_iter=30, num_warmup_runs=2, num_timer_runs=5
        )
        k_recon.append({"Mode": "torch", "Device": "GPU", "Time": time_gpu, "recon_data": k_re_gpu, "Memory": mem_gpu})
        del k_re_gpu, time_gpu, mem_gpu

    logger.info(f"__ Processing Torch CPU\n")
    k_re_cpu, time_cpu, mem_cpu = recon_ac_loraks(
        k=k_us.clone(), device=torch.device("cpu"),
        rank=50, regularization_lambda=0.0, max_num_iter=30, num_warmup_runs=2, num_timer_runs=5
    )
    k_recon.append({"Mode": "torch", "Device": "GPU", "Time": time_cpu, "recon_data": k_re_cpu, "Memory": mem_cpu})
    del k_re_cpu, time_cpu

    logger.info(f"__ Results")
    for r in k_recon:
        logger.info(f"Mode: {r['mode']}: Device: {r['device']}: Time: {r['time']:.3f} s")


def recon_ac_loraks_matlab(
        k: torch.Tensor, rank: int, regularization_lambda: float, max_num_iter: int = 30,
        num_warmup_runs: int = 2, num_timer_runs: int = 5

):
    # set data path
    path = plib.Path(__file__).absolute().parent.joinpath("data")
    logger.info(f"set matlab data path: {path}")
    path.mkdir(exist_ok=True, parents=True)

    # build matlab data - for joint echo reconstruction, we just combine channel and echo data
    k = k.view(*k.shape[:2], -1)
    mask = (k.abs() > 1e-12)

    fn = path.joinpath("input").with_suffix(".mat")
    logger.info(f"save matlab input data: {fn}")
    mat_data = {
        "k_data": k.numpy(), "mask": mask.numpy(),
        "rank": rank, "lambda": regularization_lambda, "max_num_iter": max_num_iter,
        "num_timer_runs": num_timer_runs, "num_warmup_runs": num_warmup_runs,
    }
    # save as .mat file
    savemat(fn, mat_data)

    logger.info("Call matlab routine")
    # we could provide the data path as script parameters or just stick to always using "data.mat"
    memory = run_matlab_script("ac_loraks")

    # load in results
    logger.info("Fetch results")
    results = loadmat(path.joinpath("output.mat"))

    times = torch.from_numpy(results["t"][0])
    k_recon = torch.from_numpy(results["k_recon"][0])
    t = torch.sum(times) / times.shape[0]
    return k_recon, t_processing, memory


def recon_ac_loraks(
        k: torch.Tensor, device: torch.device,
        rank: int, regularization_lambda: float,
        max_num_iter: int = 30,
        num_warmup_runs: int = 2, num_timer_runs: int = 5
):
    logger.info(f"Set device: {device}")
    # set up AC loraks
    opts = AcLoraksOptions(
        loraks_type=LoraksImplementation.AC_LORAKS, loraks_neighborhood_size=5,
        loraks_matrix_type=OperatorType.S, rank=RankReduction(method=RankReductionMethod.HARD_CUTOFF, value=rank),
        regularization_lambda=regularization_lambda, max_num_iter=max_num_iter, device=device,
        computation_type=ComputationType.FFT, solver_type=SolverType.LEASTSQUARES
    )
    ac_loraks = Loraks.create(options=opts)

    # prep data
    k_in, in_shape, padding, batch_channel_idx = prep_k_space(k=k.unsqueeze_(2), batch_size_channels=-1)


    # timing
    t = Timer(
        stmt="ac_loraks.reconstruct(k_in)",
        # setup="from __main__ import wfn",
        globals={"ac_loraks": ac_loraks, "k_in": k_in}
    )
    # warmup & result
    recon = ac_loraks.reconstruct(k_in)
    _ = t.timeit(max(1, num_warmup_runs - 1))

    # measurement
    t_processing = t.timeit(num_timer_runs) / num_timer_runs

    # unprep
    k_recon = unprep_k_space(recon, padding=padding, batch_idx=batch_channel_idx, input_shape=in_shape)

    return k_recon, t_processing, None


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    main()
