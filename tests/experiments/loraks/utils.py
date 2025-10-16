import torch
import logging
import pathlib as plib
import sys

from enum import Enum, auto
from pymritools.utils import torch_load
from pymritools.recon.loraks.utils import (
    prepare_k_space_to_batches, pad_input,
    check_channel_batch_size_and_batch_channels, unpad_output, unprepare_batches_to_k_space
)
from typing import Tuple, Union
import subprocess
from os import path
import msparser
import tempfile

p_tests = plib.Path(__file__).absolute().parent.parent.parent.parent
sys.path.append(p_tests.as_posix())
from tests.utils import get_test_result_output_dir, create_phantom, ResultMode

logger = logging.getLogger(__name__)


class DataType(Enum):
    SHEPPLOGAN = auto()
    PHANTOM = auto()
    INVIVO = auto()
    MASK = auto()


# __ misc functionalities
def load_data(data_type: DataType):
    match data_type:
        case DataType.SHEPPLOGAN:
            # set some shapes
            nx, ny, nc, ne = (286, 256, 48, 8)
            k, k_us = create_phantom(shape_xyct=(nx, ny, nc, ne), acc=1)
            bet = None
        case DataType.PHANTOM:
            raise NotImplementedError("Phantom data not yet provided")
        case DataType.INVIVO:
            # load input data fully sampled
            path = plib.Path(
                get_test_result_output_dir(f"data", mode=ResultMode.EXPERIMENT)
            )
            k = torch_load(path.joinpath("fs_data_slice.pt"))
            # ensure read dimension is correct
            k = torch.swapdims(k, 0, 1)
            k_us = None
            bet = torch_load(path.joinpath("bet.pt"))
        case _:
            raise ValueError(f"Data Type {data_type.name} not supported")
    return k, k_us, bet


# __ methods for autograd LORAKS iteration & prep
def prep_k_space(k: torch.Tensor, batch_size_channels: int = -1, use_correlation_clustering: bool = True):
    # we need to prepare the k-space
    batch_channel_idx = check_channel_batch_size_and_batch_channels(
        k_space_rpsct=k, batch_size_channels=batch_size_channels, use_correlation_clustering=use_correlation_clustering
    )
    prep_k, in_shape = prepare_k_space_to_batches(
        k_space_rpsct=k, batch_channel_indices=batch_channel_idx
    )
    prep_k, padding = pad_input(prep_k, sampling_dims=(-2, -1))
    return prep_k, in_shape, padding, batch_channel_idx


def unprep_k_space(k: torch.Tensor, padding: Tuple[int, int], batch_idx: torch.tensor, input_shape: Tuple):
    k = unpad_output(k_space=k, padding=padding)

    return unprepare_batches_to_k_space(
        k_batched=k, batch_channel_indices=batch_idx, original_shape=input_shape
    )


def run_command(function_call: str,
                profile_memory: bool = True,
                capture_output: bool = True):
    import subprocess, shlex
    if not profile_memory:
        command_to_run = shlex.split(function_call)
    else:
        # run the profile.sh found in the cpu_mem_profile folder
        profile_script_path = plib.Path(__file__).absolute().parent.joinpath("cpu_mem_profile").joinpath("profile").with_suffix(".sh").as_posix()

        command_to_run = [profile_script_path] + shlex.split(function_call)

    command_return = subprocess.run(command_to_run, capture_output=capture_output, text=True)

    if capture_output:
        logger.warning(f"Log OUT:\n{command_return.stderr}")
        logger.info(f"___________________________")
        logger.info(f"OUT:\n{command_return.stdout}")
        logger.info(f"___________________________")
    return command_return


def read_profile_peak_memory(command_stdout):
    """
    Read the peak memory from the profile log file

    Args:
        command_stdout: profile.sh command std out

    Returns:
        int: Peak memory usage in MB

    Raises:
        ValueError: If the specified file does not exist or is not a file.
    """
    try:
        # Read the last line which contains the peak memory
        if command_stdout.startswith("#"):
            # Look for peak memory in comment line
            peak_memory = int(command_stdout.split("=")[-1].strip().split()[0])
            return peak_memory
        # if no comment line was found parse last data line and find last "#"
        parts = command_stdout.split("#")
        if len(parts) < 1:
            raise ValueError
        else:
            # take last part
            peak_memory = int(parts[-1].split("=")[-1].strip().split()[0])
            return peak_memory
    except (FileNotFoundError, ValueError, IndexError):
        logger.error(f"Error reading Profile output: {command_stdout}")
        return 0
    return 0


def run_ac_loraks_matlab_script(profile_memory: bool = True,
                                script_args=None,
                                script_dir=None,
                                capture_output=True) -> dict:
    """
    Run AC-Loraks script with the given arguments.

    Parameters:
        profile_memory:
        script_args: Arguments to pass to the MATLAB script
        script_dir: Directory containing the MATLAB script
                   (defaults to 'matlab' subdirectory of the current file's directory)
        capture_output: Whether to capture and return the output

    Returns:
        subprocess.CompletedProcess: Result of the subprocess run

    Raises:
        RuntimeError: If the MATLAB script execution fails
    """

    # Default script directory is 'matlab' subdirectory of current file's directory
    if script_dir is None:
        script_dir = plib.Path(__file__).absolute().parent.joinpath("matlab")
    else:
        script_dir = plib.Path(script_dir)

    # Since Loraks is not open-source, we need developers to copy the script to the right location
    # because it will not be in the repository.
    if not script_dir.joinpath("AC_LORAKS.m").is_file():
        raise FileNotFoundError(f"Matlab script 'AC_LORAKS.m' not found in directory '{script_dir}'")

    # TODO: if possible, we should replace "matlab" with the real binary e.g. MATLAB/R2022a/bin/glnxa64/MATLAB
    # This would prevent valgrind to trace all commands that are done in the matlab shell script
    # However, it would be possible that library paths need to be set, etc.
    # Also check "matlab -n" which gives the overview of all variables and settings and which maybe will allow
    # us to extract the real binary automatically
    matlab_cmd = f"matlab -nodisplay -nosplash -nodesktop -nojvm -r \"addpath('{script_dir}'); "
    script_func = "profile_ac_loraks"

    if script_args:
        matlab_cmd += f"run('{script_func}({script_args})'); "
    else:
        matlab_cmd += f"run('{script_func}'); "

    matlab_cmd += "exit;\""
    logger.debug(f"CMD:: {matlab_cmd}")

    # For the valgrind massif output file, a temporary file is enough.
    # We return the file name together with the other return values.
    command_return = run_command(
        function_call=matlab_cmd,
        profile_memory=profile_memory,
        capture_output=capture_output
    )

    memory_usage_mb = read_profile_peak_memory(command_return.stdout) / 1024
    return memory_usage_mb


def run_ac_loraks_torch_script(profile_memory: bool = True,
                               script_args=None,
                               script_dir=None,
                               data_dir=None,
                               capture_output=True) -> dict:
    # Default script directory is 'torch' subdirectory of current file's directory
    if script_dir is None:
        script_dir = plib.Path(__file__).absolute().parent.joinpath("torch")
    else:
        script_dir = plib.Path(script_dir)
    script = script_dir.joinpath("run_torch_cpu").with_suffix(".py")
    if not script.is_file():
        raise FileNotFoundError(f"Python script 'run_torch_cpu.py' not found in directory '{script_dir}'")

    data_dir = plib.Path(data_dir)
    if not data_dir.is_file():
        raise FileNotFoundError(f"Data not found in file '{data_dir}'")

    # get script args
    rank, reg_lam, max_num_iter = script_args

    cmd = f"conda run --name mri_tools_env python {script.as_posix()} --file {data_dir.as_posix()} --rank {rank} --max_num_iter {max_num_iter} --regularization_lambda {reg_lam}"
    c_out = "CMD:\n"
    for i, p in enumerate(cmd.split("--")):
         c_out += f"\t--{p}\n" if i > 0 else f"{p}\n"
    logger.debug(c_out)

    # We return profiled memory output
    command_return = run_command(
        function_call=cmd,
        profile_memory=profile_memory,
        capture_output=capture_output
    )
    memory_usage_mb = read_profile_peak_memory(command_return.stdout) / 1024
    return memory_usage_mb
