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
from typing import Tuple
import subprocess
from os import path
import msparser

p_tests = plib.Path(__file__).absolute().parent.parent.parent.parent
sys.path.append(p_tests.as_posix())
from tests.utils import get_test_result_output_dir, create_phantom, ResultMode

logger = logging.getLogger(__name__)


class DataType(Enum):
    SHEPPLOGAN = auto()
    PHANTOM = auto()
    INVIVO = auto()
    MASK = auto()



#__ misc functionalities
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


#__ methods for autograd LORAKS iteration & prep
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


def read_massif_max_memory_used(file_path):
    if not path.exists(file_path) or not path.isfile(file_path):
        raise ValueError(f"File '{file_path}' does not exist or is not a file.")
    data = msparser.parse_file(file_path)
    peak_idx = data['peak_snapshot_index']
    peak_snapshot = data['snapshots'][peak_idx]
    peak_heap_memory = peak_snapshot['mem_heap']
    peak_extra_memory = peak_snapshot['mem_heap_extra']
    return peak_heap_memory / (1024**2) , peak_extra_memory / (1024**2)


## For matlab comparisons we need the following logic:
# 1) Save the data as <data>.mat file
# 2) provide a <script>.m doing the task while interfacing with the <data>.mat
# 3) Save the results as <results>.mat inside the script (2)
# 4) load results via scipy.io into python as dictionary and do the rest

# For timing comparisons we could try to run the process without an actual function call in matlab,
# just the process calls and data loading, and then subtract this timing. Or we do the time keeping in matlab.

def run_matlab_script(script_name, use_valgrind: bool = True,
                      script_args=None, script_dir=None, capture_output=True):
    """
    Run a MATLAB script with the given arguments.

    Parameters:
        script_name (str): Name of the MATLAB script file (with or without .m extension)
        script_args (str, optional): Arguments to pass to the MATLAB script
        script_dir (str, optional): Directory containing the MATLAB script
                                   (defaults to 'matlab' subdirectory of current file's directory)
        capture_output (bool, optional): Whether to capture and return the output

    Returns:
        subprocess.CompletedProcess: Result of the subprocess run

    Raises:
        RuntimeError: If the MATLAB script execution fails
    """
    # Ensure script name has .m extension
    if not script_name.endswith('.m'):
        script_name += '.m'

    # Default script directory is 'matlab' subdirectory of current file's directory
    if script_dir is None:
        script_dir = plib.Path(__file__).absolute().parent.joinpath("matlab")
    # ensure Path object
    script_dir = plib.Path(script_dir)

    # Full path to the script
    script_path = script_dir.joinpath(script_name)

    # Construct the MATLAB command
    matlab_cmd = f"matlab -nodisplay -nosplash -nodesktop -r \"addpath('{script_dir}'); "

    # Extract script name without extension for the function call
    script_func = script_path.as_posix()

    # Add function call with arguments if provided
    if script_args:
        matlab_cmd += f"run('{script_func}({script_args})'); "
    else:
        matlab_cmd += f"run('{script_func}'); "

    # Add an exit command to close MATLAB after execution
    matlab_cmd += "exit;\""

    logger.debug(f"CMD:: {matlab_cmd}")
    # Run the MATLAB command
    # process = subprocess.run(matlab_cmd, shell=True, capture_output=capture_output, text=True)
    # process = subprocess.Popen(matlab_cmd, shell=True, text=True)
    file_name = "/data/pt_np-jschmidt/code/PyMRItools/test_output/EXPERIMENT/speed_comparison/massif.out.123"
    command_return = run_command(
        function_call=matlab_cmd, use_valgrind=use_valgrind,
        massif_output_file=file_name,
        capture_output=capture_output
    )

    memory_usage_mb, peak_extra_memory_mb = read_massif_max_memory_used(
        file_path=file_name
    )
    return memory_usage_mb, peak_extra_memory_mb
