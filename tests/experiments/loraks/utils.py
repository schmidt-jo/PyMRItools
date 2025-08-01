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
                use_valgrind: bool = False,
                massif_output_file: str = "massif.out",
                capture_output: bool = True):
    import subprocess, shlex
    if not use_valgrind:
        command_to_run = shlex.split(function_call)
    else:
        command_to_run = [
                             "valgrind",
                             "--tool=massif",
                             "--trace-children=yes",
                             f"--massif-out-file={massif_output_file}",
                         ] + shlex.split(function_call)
    return subprocess.run(command_to_run, capture_output=capture_output, text=True)


def read_massif_max_memory_used(file_path: Union[str, plib.Path]):
    """
    Reads the maximum memory usage details from a valgrind massif output file.

    Args:
        file_path: The file path to the massif output file as a string.

    Returns:
        A tuple containing two floats:
            - Maximum heap memory usage in megabytes.
            - Maximum extra heap memory usage in megabytes.

    Raises:
        ValueError: If the specified file does not exist or is not a file.
    """
    file_path = plib.Path(file_path)
    if not file_path.exists() or not file_path.is_file():
        raise ValueError(f"File '{file_path}' does not exist or is not a file.")
    data = msparser.parse_file(file_path)
    peak_idx = data['peak_snapshot_index']
    peak_snapshot = data['snapshots'][peak_idx]
    peak_heap_memory = peak_snapshot['mem_heap']
    peak_extra_memory = peak_snapshot['mem_heap_extra']
    return peak_heap_memory / (1024 ** 2), peak_extra_memory / (1024 ** 2)


def run_ac_loraks_matlab_script(use_valgrind: bool = True,
                                script_args=None,
                                script_dir=None,
                                capture_output=True) -> dict:
    """
    Run AC-Loraks script with the given arguments.

    Parameters:
        use_valgrind:
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
    import time
    tmp_massif_file = tempfile.mktemp(suffix=f".out.{int(time.time())}")
    command_return = run_command(
        function_call=matlab_cmd,
        use_valgrind=use_valgrind,
        massif_output_file=tmp_massif_file,
        capture_output=capture_output
    )

    memory_usage_mb, peak_extra_memory_mb = read_massif_max_memory_used(
        file_path=tmp_massif_file
    )
    return {
        "massif_file": tmp_massif_file,
        "peak_memory": memory_usage_mb,
        "peak_extra_memory": peak_extra_memory_mb,
        "command_output": None if not capture_output else command_return
    }


def run_ac_loraks_torch_script(use_valgrind: bool = True,
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
    logger.info(f"CMD: '{cmd}'")

    # For the valgrind massif output file, a temporary file is enough.
    # We return the file name together with the other return values.
    import time
    tmp_massif_file = tempfile.mktemp(suffix=f".out.{int(time.time())}")
    command_return = run_command(
        function_call=cmd,
        use_valgrind=use_valgrind,
        massif_output_file=tmp_massif_file,
        capture_output=capture_output
    )

    memory_usage_mb, peak_extra_memory_mb = read_massif_max_memory_used(
        file_path=tmp_massif_file
    )
    return {
        "massif_file": tmp_massif_file,
        "peak_memory": memory_usage_mb,
        "peak_extra_memory": peak_extra_memory_mb,
        "command_output": None if not capture_output else command_return
    }
