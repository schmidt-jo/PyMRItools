import pathlib as plib
import sys
import logging
from timeit import Timer
import polars as pl

import torch
from pymritools.recon.loraks.loraks import Loraks
from pymritools.recon.loraks.ac_loraks import AcLoraksOptions, ComputationType, SolverType
from pymritools.recon.loraks.loraks import LoraksImplementation, OperatorType, RankReduction, RankReductionMethod
from scipy.io import loadmat, savemat

p_tests = plib.Path(__file__).absolute().parent.parent.parent.parent
sys.path.append(p_tests.as_posix())
from tests.utils import get_test_result_output_dir, ResultMode
from tests.experiments.loraks.utils import prep_k_space, unprep_k_space, create_phantom, run_matlab_script, TorchMemoryTracker

logger = logging.getLogger(__name__)

### We need:
## 1) Provide the data a) virtual data of different sizes, b) real examples
## 2) Set up AC LORAKS leastsquares for GPU and CPU and matlab (same algorithm) for joint-echo -recon
## 3) Time computations

## 4) Set up P-LORAKS for GPU and CPU and matlab (same algorithm) for joint-echo-recon
## 5) Time computations


def main():
    # get path
    path_out = plib.Path(get_test_result_output_dir("speed_comparison", mode=ResultMode.EXPERIMENT))

    # do loops for different data sizes
    ms_xy = torch.linspace(100*100, 280*280, 10)
    ncs = torch.arange(4, 24, 4)
    nes = torch.arange(2, 6, 1)
    # ms_xy = torch.tensor([10000])
    # ncs = torch.tensor([4])
    # nes = torch.tensor([2])

    meas = []

    for i, mxy in enumerate(ms_xy):
        logger.info(f"Processing Matrix Size XY : {i+1} / {ms_xy.shape[0]}")
        nx = torch.sqrt(mxy).to(torch.int)
        ny = (mxy / nx).to(torch.int)
        for g, nc in enumerate(ncs):
            logger.info(f"__ nc: {g+1} / {ncs.shape[0]}")
            for h, ne in enumerate(nes):
                logger.info(f"__ ne: {h+1} / {nes.shape[0]}")
                mce = ne * nc * 5 ** 2

                mem_track_load = TorchMemoryTracker(torch.device("cpu"))
                mem_track_load.start_tracking()
                # get the data
                _, k_us = create_phantom(shape_xyct=(nx.item(), ny.item(), nc.item(), ne.item()), acc=3, ac_lines=ny.item() // 6.5)
                mem_track_load.end_tracking()
                mem_load = mem_track_load.get_memory_usage()
                tmp = loop(
                    k_us=k_us, mxy=mxy.item(), mce=mce.item(), nc=nc.item(), ne=ne.item(), batch_size_channels=-1,
                    rank=max(15, min(mce.item(), mxy.item()) // 10), regularization_lambda=0.0, max_num_iter=20, num_warmup_runs=2,
                    num_timer_runs=3, mem_load=mem_load
                )
                meas.extend(tmp)
        df = pl.DataFrame(meas)
        fn = path_out.joinpath("results_df").with_suffix(".json")
        logger.info(f"Writing to {fn}")
        df.write_ndjson(fn)


def loop(k_us: torch.Tensor, mxy, mce, nc, ne, batch_size_channels: int = -1,
         rank=50, regularization_lambda=0.0, max_num_iter=30, num_warmup_runs=2, num_timer_runs=3, mem_load=0):
    tmp = []
    logger.info(f"__ Processing Torch GPU\n")
    if not torch.cuda.is_available():
        msg = "No GPU device available, skipping GPU benchmark"
        logger.warning(msg)
    else:
        try:
            k_re_gpu, time_gpu, mem_gpu = recon_ac_loraks(
                k=k_us.clone(), device=torch.device("cuda:0"),
                rank=rank, regularization_lambda=regularization_lambda,
                max_num_iter=max_num_iter, num_warmup_runs=num_warmup_runs, num_timer_runs=num_timer_runs
            )
            tmp.append({
                "Mode": "torch", "Device": "GPU", "Time": time_gpu, "Memory": mem_gpu,
                "mxy": mxy, "mce": mce, "nc": nc, "ne": ne
            })
            del k_re_gpu, time_gpu, mem_gpu
        except Exception as e:
            tmp.append({
                "Mode": "torch", "Device": "GPU", "Time": None, "Memory": "Maxed Out",
                "mxy": mxy, "mce": mce, "nc": nc, "ne": ne
            })


    logger.info(f"__ Processing Torch CPU\n")
    try:
        k_re_cpu, time_cpu, mem_cpu = recon_ac_loraks(
            k=k_us.clone(), device=torch.device("cpu"),
            rank=rank, regularization_lambda=regularization_lambda,
            max_num_iter=max_num_iter, num_warmup_runs=num_warmup_runs, num_timer_runs=num_timer_runs
        )
        # for all but torch cpu the tensor is in ram (GPU) or the loading is tracked in memory (matlab),
        # hence we need to add this here
        mem_cpu += mem_load
        tmp.append({
            "Mode": "torch", "Device": "CPU", "Time": time_cpu, "Memory": mem_cpu,
            "mxy": mxy, "mce": mce, "nc": nc, "ne": ne
        })
        del k_re_cpu, time_cpu, mem_cpu
    except Exception as e:
        tmp.append({
            "Mode": "torch", "Device": "CPU", "Time": None, "Memory": "Maxed Out",
            "mxy": mxy, "mce": mce, "nc": nc, "ne": ne
        })

    logger.info(f"__ Processing Matlab CPU\n")
    k_re_mat, time_mat, mem_mat = recon_ac_loraks_matlab(
        k=k_us, rank=rank, regularization_lambda=regularization_lambda,
        max_num_iter=max_num_iter, num_warmup_runs=num_warmup_runs, num_timer_runs=num_timer_runs
    )
    tmp.append({
        "Mode": "matlab", "Device": "CPU", "Time": time_mat, "Memory": mem_mat,
        "mxy": mxy, "mce": mce, "nc": nc, "ne": ne
    })
    del k_re_mat, time_mat, mem_mat

    return tmp


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
    results = loadmat(path.joinpath("output.mat").as_posix())

    times = torch.from_numpy(results["t"][0])
    k_recon = torch.from_numpy(results["k_recon"][0])
    t = torch.sum(times) / times.shape[0]
    return k_recon, t.item(), memory


def loraks_init_run(
        k: torch.Tensor, device: torch.device,
        rank: int, regularization_lambda: float,
        max_num_iter: int = 30,):
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

    # memory
    mem_track = TorchMemoryTracker(device=device)

    # timing
    t = Timer(
        stmt="loraks_init_run(k, device, rank, regularization_lambda, max_num_iter)",
        setup="from __main__ import loraks_init_run",
        globals={"k": k, "device": device, "rank": rank, "regularization_lambda": regularization_lambda, "max_num_iter": max_num_iter,}
    )

    # warmup & result
    _ = t.timeit(max(1, num_warmup_runs - 1))

    # Mem Measurement
    mem_track.start_tracking()

    k_recon = loraks_init_run(k=k, device=device, rank=rank, regularization_lambda=regularization_lambda, max_num_iter=max_num_iter)
    mem_track.end_tracking()

    # Time Measurement
    t_processing = t.timeit(num_timer_runs) / num_timer_runs

    memory = mem_track.get_memory_usage()
    return k_recon, t_processing, memory


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    main()
