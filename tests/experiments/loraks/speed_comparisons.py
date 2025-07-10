import pathlib as plib
import sys
import logging
from timeit import Timer

import torch
from pymritools.recon.loraks.loraks import Loraks
from pymritools.recon.loraks.ac_loraks import AcLoraksOptions, ComputationType, SolverType
from pymritools.recon.loraks.loraks import LoraksImplementation, OperatorType, RankReduction, RankReductionMethod

p_tests = plib.Path(__file__).absolute().parent.parent.parent.parent
sys.path.append(p_tests.as_posix())
from tests.utils import get_test_result_output_dir, ResultMode
from tests.experiments.loraks.utils import prep_k_space, unprep_k_space, DataType, create_phantom

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

    logger.info(f"__ Processing Torch GPU\n")
    if not torch.cuda.is_available():
        msg = "No GPU device available, skipping GPU benchmark"
        logger.warning(msg)
    else:
        k_re_gpu, time_gpu = recon_ac_loraks(k=k_us.clone(), device=torch.device("cuda:0"))
        k_recon.append({"mode": "torch", "device": "GPU", "time": time_gpu, "recon_data": k_re_gpu})
        del k_re_gpu, time_gpu

    logger.info(f"__ Processing Torch CPU\n")
    k_re_cpu, time_cpu = recon_ac_loraks(k=k_us.clone(), device=torch.device("cpu"))
    k_recon.append({"mode": "torch", "device": "CPU", "time": time_cpu, "recon_data": k_re_cpu})
    del k_re_cpu, time_cpu

    # logger.info(f"__ Processing Matlab CPU\n")
    # k_re_cpu, time_cpu = recon_ac_loraks_matlab(k=k_us)
    # k_recon.append({"mode": "torch", "device": "CPU", "time": time_cpu, "recon_data": k_re_cpu})
    # del k_re_cpu, time_cpu

    logger.info(f"__ Results")
    for r in k_recon:
        logger.info(f"Mode: {r['mode']}: Device: {r['device']}: Time: {r['time']:.3f} s")


def recon_ac_loraks(k: torch.Tensor, device: torch.device):
    logger.info(f"Set device: {device}")
    # set up AC loraks
    opts = AcLoraksOptions(
        loraks_type=LoraksImplementation.AC_LORAKS, loraks_neighborhood_size=5,
        loraks_matrix_type=OperatorType.S, rank=RankReduction(method=RankReductionMethod.HARD_CUTOFF, value=150),
        regularization_lambda=0.0, max_num_iter=30, device=device,
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
    _ = t.timeit(2)

    # measurement
    t_processing = t.timeit(5) / 5

    # unprep
    k_recon = unprep_k_space(recon, padding=padding, batch_idx=batch_channel_idx, input_shape=in_shape)

    return k_recon, t_processing


def recon_ac_loraks_matlab(k: torch.Tensor):
    pass


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    main()
