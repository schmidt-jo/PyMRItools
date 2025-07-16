import torch
import logging
import pathlib as plib
import sys
import threading
import time
import psutil
from enum import Enum, auto
from pymritools.utils import torch_load
from pymritools.recon.loraks.utils import (
    prepare_k_space_to_batches, pad_input,
    check_channel_batch_size_and_batch_channels, unpad_output, unprepare_batches_to_k_space
)
from typing import Tuple
import subprocess
import os
import tracemalloc

p_tests = plib.Path(__file__).absolute().parent.parent.parent.parent
sys.path.append(p_tests.as_posix())
from tests.utils import get_test_result_output_dir, create_phantom, ResultMode

logger = logging.getLogger(__name__)


class DataType(Enum):
    SHEPPLOGAN = auto()
    PHANTOM = auto()
    INVIVO = auto()



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
def prep_k_space(k: torch.Tensor, batch_size_channels: int = -1):
    # we need to prepare the k-space
    batch_channel_idx = check_channel_batch_size_and_batch_channels(
        k_space_rpsct=k, batch_size_channels=batch_size_channels
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

## For matlab comparisons we need the following logic:
# 1) Save the data as <data>.mat file
# 2) provide a <script>.m doing the task while interfacing with the <data>.mat
# 3) Save the results as <results>.mat inside the script (2)
# 4) load results via scipy.io into python as dictionary and do the rest

# For timing comparisons we could try to run the process without an actual function call in matlab,
# just the process calls and data loading, and then subtract this timing. Or we do the time keeping in matlab.

def run_matlab_script(script_name, script_args=None, script_dir=None, capture_output=True):
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
    process = subprocess.Popen(matlab_cmd, shell=True, text=True)

    # Get the process object of the subprocess
    subprocess_process = psutil.Process(process.pid)

    memory_usage = []
    shared = []
    memory_usage.append(subprocess_process.memory_info().rss / (1024 * 1024))
    # Track the memory usage during the process
    # Function to track the memory usage of the subprocess
    def track_memory_usage():
        while subprocess_process.is_running():
            mfi = subprocess_process.memory_full_info()
            mem = mfi.rss + mfi.vms
            shared.append(mfi.shared / (1024 * 1024))
            memory_usage.append(mem / (1024 * 1024))
            time.sleep(0.5)

    # Create a thread to track the memory usage of the subprocess
    thread = threading.Thread(target=track_memory_usage)
    thread.start()

    # Wait for the subprocess to finish
    process.wait()
    #
    # # Check for errors
    # if result.returncode != 0:
    #     if capture_output:
    #         print(f"MATLAB error: {result.stderr}")
    #     raise RuntimeError(f"MATLAB script '{script_name}' execution failed")

    memory_usage = max(memory_usage)
    shared_usage = max(shared)
    return memory_usage, shared_usage


class TorchMemoryTracker:
    def __init__(self, device: torch.device):
        self.memory_allocated = 0
        self.memory_reserved = 0
        if device.type not in ['cpu', 'cuda']:
            raise ValueError(f"Device type {device.type} not supported")
        self.device = device
        self.memory_info = {}

        self.process = psutil.Process(os.getpid()) if self.device.type == 'cpu' else None


    def _get_cuda_mem_dict(self):
        cuda_max_alloc = torch.cuda.max_memory_allocated(self.device)
        cuda_res = torch.cuda.memory_reserved(self.device)
        d = {
            'memory_allocated': cuda_max_alloc,
            'memory_reserved': cuda_res,
            'memory_allocated_mb': cuda_max_alloc / (1024 * 1024),
            'memory_reserved_mb': cuda_res / (1024 * 1024)
        }
        return d

    def _get_cpu_mem_dict(self):
        # CPU memory tracking
        mem_info = self.process.memory_full_info()
        mem_used = mem_info.rss + mem_info.vms
        # Get tracemalloc stats
        current, peak = tracemalloc.get_traced_memory()
        d = {
            'memory_used': mem_used,  # Memory used during the operation
            'memory_used_mb': mem_used / (1024 * 1024),
            'tracemalloc_current': current,
            'tracemalloc_peak': peak,
            'tracemalloc_current_mb': current / (1024 * 1024),
            'tracemalloc_peak_mb': peak / (1024 * 1024),
            "mem_info": mem_info
        }
        return d

    def start_tracking(self):
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.reset_accumulated_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
            self.memory_info["start"] = self._get_cuda_mem_dict()
        else:
            tracemalloc.start()
            tracemalloc.reset_peak()
            self.memory_info["start"] = self._get_cpu_mem_dict()

    def end_tracking(self):
        if self.device.type == 'cuda':
            self.memory_info["end"] = self._get_cuda_mem_dict()
        else:
            self.memory_info["end"] = self._get_cpu_mem_dict()
            tracemalloc.stop()

    def get_memory_usage(self):
        if self.device.type == 'cuda':
            mem_after = self.memory_info["end"]["memory_allocated_mb"]
            mem_before = self.memory_info["start"]["memory_allocated_mb"]
            tm = None
            shared = None
        else:
            mem_after = self.memory_info["end"]["memory_used_mb"]
            mem_before = self.memory_info["start"]["memory_used_mb"]
            tm = self.memory_info["end"]["tracemalloc_peak_mb"] - self.memory_info["start"]["tracemalloc_peak_mb"]
            shared = (self.memory_info["end"]["mem_info"].shared - self.memory_info["start"]["mem_info"].shared) / 1024 / 1024
        return mem_after - mem_before, tm, shared

