from typing import Tuple
from enum import Enum, auto

import torch
import time
import os
import logging
import pickle

from pymritools.utils import Phantom


test_dir = os.path.dirname(__file__)
test_output_dir = os.path.join(os.path.dirname(test_dir), "test_output")


class ResultMode(Enum):
    """
    Specifies the type of test result to be generated.
    """
    TEST = auto()
    VISUAL = auto()
    EXPERIMENT = auto()


def get_test_result_output_dir(func, mode: ResultMode = ResultMode.TEST) -> str:
    """
    Returns a unique output directory for the given test function.

    For running profiling and timing tests, each function tested should have their own output directory under a common
    "test_output" directory.
    :param func: Function argument to get the name from
    :return: the absolute path of the output directory
    """
    if isinstance(func, str):
        func_name = func
    elif hasattr(func, "__name__"):
        func_name = func.__name__
    else:
        raise RuntimeError(f"Provided function argument should be a function or a string.")

    out_dir = os.path.join(test_output_dir, mode.name, func_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return os.path.abspath(out_dir)


def measure_function_call(func, *args, iterations: int = 10, disable_compilation: bool = False) -> tuple[float, float]:
    """
    Measures the execution time of a function call over multiple iterations.
    Can be called on any function but should not be used for torch compiled code.

    :param func: The function to measure the execution time of.
    :param args: The arguments to pass to the function during its call.
    :param iterations: The number of times to call the function. The default is 10.
    :param disable_compilation: Turns torch compilation on/off
    :return: The average execution time per call in seconds.
    """
    warmup_iterations = 3
    compiled_func = torch.compile(func, fullgraph=True, disable=disable_compilation)
    has_device_arg = "device" in func.__annotations__

    # Warmup phase
    # This should be necessary as I'm not sure if the compilation takes place on the call or during the first run.
    start_time = time.time()
    for _ in range(warmup_iterations):
        if has_device_arg:
            compiled_func(*args, device=torch.device("cpu"))
        else:
            compiled_func(*args)
    end_time = time.time()
    warmup_time = (end_time - start_time) / warmup_iterations

    start_time = time.time()
    for _ in range(iterations):
        if has_device_arg:
            compiled_func(*args, device=torch.device("cpu"))
        else:
            compiled_func(*args)
    end_time = time.time()
    elapsed_time = (end_time - start_time) / iterations
    return elapsed_time, warmup_time


def measure_cuda_function_call(
        func, *args, device: torch.device, iterations=10, disable_compilation: bool = False) -> tuple[float, float]:
    """
    Measures the execution time of a torch compiled function call over multiple iterations.
    Time measurement is tuned for CUDA devices.

    :param func: The function to be compiled and executed.
    :param args: Arguments to be passed to the function during execution.
    :param device: Is forwarded to the called function as keyword argument "device".
    :param iterations: The number of times to execute the function for performance measurement. Default is 10.
    :param disable_compilation: Turns torch compilation on/off.
    :return: The average execution time in seconds of the compiled function over the specified number of iterations.
    """
    warmup_iterations = 3
    compiled_func = torch.compile(func, fullgraph=True, disable=disable_compilation)

    has_device_arg = "device" in func.__annotations__

    # Warmup phase
    # This should be necessary as I'm not sure if the compilation takes place on the call or during the first run.
    start_time = time.time()
    for _ in range(warmup_iterations):
        if has_device_arg:
            compiled_func(*args, device=device)
        else:
            compiled_func(*args)
    end_time = time.time()
    warmup_time = (end_time - start_time) / warmup_iterations

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        if has_device_arg:
            compiled_func(*args, device=device)
        else:
            compiled_func(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations / 1000, warmup_time


def profile_torch_function(func, *args, device: torch.device, disable_compilation: bool = False) -> None:
    compiled_func = torch.compile(func, fullgraph=True, disable=disable_compilation, options={"trace.enabled": True})
    file_name = f"trace_{'compiled' if not disable_compilation else 'uncompiled'}.json"
    has_device_arg = "device" in func.__annotations__

    # Warmup phase
    for _ in range(3):
        if has_device_arg:
            compiled_func(*args, device=device)
        else:
            compiled_func(*args)

    with torch.profiler.profile() as prof:
        if has_device_arg:
            compiled_func(*args, device=device)
        else:
            compiled_func(*args)
        prof.step()
    prof.export_chrome_trace(os.path.join(get_test_result_output_dir(func), file_name))


def do_performance_test(func, *args, iterations=10, test_compilation: bool = True):
    has_cuda = torch.cuda.is_available()
    time_non_compiled, warmup_1 = measure_function_call(func, *args, iterations=iterations, disable_compilation=True)
    profile_torch_function(func, *args, device=torch.device("cpu"), disable_compilation=True)
    if test_compilation:
        time_compiled, warmup_2 = measure_function_call(func, *args, iterations=iterations)
        profile_torch_function(func, *args, device=torch.device("cpu"))

    cuda_time_non_compiled = None
    cuda_time_compiled = None
    warmup_3 = None
    warmup_4 = None
    if has_cuda:
        compiled_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                compiled_args.append(arg.cuda())
            else:
                compiled_args.append(arg)

        cuda_time_non_compiled, warmup_3 = measure_cuda_function_call(
            func,
            *compiled_args,
            device=torch.device("cuda"),
            iterations=iterations,
            disable_compilation=True)
        if test_compilation:
            cuda_time_compiled, warmup_4 = measure_cuda_function_call(
                func,
                *compiled_args,
                device=torch.device("cuda"),
                iterations=iterations)

    results_file = os.path.join(get_test_result_output_dir(func), "results.txt")
    with open(results_file, "w") as f:
        f.write("CPU Performance\n---------------\n\n")
        f.write(f"Non-compiled function took: {time_non_compiled:.6f} seconds per iteration\n")
        if test_compilation:
            f.write(f"Compiled function took: {time_compiled:.6f} seconds per iteration\n\n")
            f.write(f"First call timing: {warmup_1:.6f} seconds vs. {warmup_2}\n")
            f.write(f"Speed-up: {time_non_compiled / time_compiled:.2f}x\n\n")

        if has_cuda:
            f.write("CUDA Performance\n----------------\n\n")
            f.write(f"CUDA: Non-compiled function took: {cuda_time_non_compiled:.6f} seconds per iteration\n")
            if test_compilation:
                f.write(f"CUDA: Compiled function took: {cuda_time_compiled:.6f} seconds per iteration\n\n")
                f.write(f"First call timing: {warmup_3:.6f} seconds vs. {warmup_4}\n")
                f.write(f"Speed-up: {cuda_time_non_compiled / cuda_time_compiled:.2f}x\n\n")


def generate_trace_plot(pkl_file: str, output_file: str):
    """
    Generates a trace plot from a pickle file and writes it to the specified output HTML file.
    This is ripped out of torch.cuda._memory_viz
    """
    from torch.cuda._memory_viz import trace_plot
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    with open(output_file, 'w') as f:
        f.write(trace_plot(data))


class MemoryProfiler:
    """
    Utility class to profile and record CUDA memory usage during program execution.
    Example usage::

        with MemoryProfiler("indices_gpu_test"):
            # Any torch code here
    """

    def __init__(self, report_output_dir: str, max_entries=100000):
        """
        Initialize the memory profiler.

        Args:
            report_output_dir (str): The directory name to save memory snapshots under the test_output directory.
            max_entries (int): Maximum number of memory events to capture.
        """
        self.report_output_dir = get_test_result_output_dir(report_output_dir)
        self.snapshot_file = os.path.join(self.report_output_dir, "memory_snapshot.pkl")
        self.snapshot_file_html = os.path.join(self.report_output_dir, "memory_snapshot.html")
        self.max_entries = max_entries
        self.logger = logging.getLogger("MemoryProfiler")

    def __enter__(self):
        try:
            torch.cuda.memory._record_memory_history(max_entries=self.max_entries)
            self.logger.info(f"Memory profiling started with max_entries={self.max_entries}")
        except Exception as e:
            self.logger.error(f"Failed to start memory profiling: {e}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            torch.cuda.memory._dump_snapshot(self.snapshot_file)
            generate_trace_plot(self.snapshot_file, self.snapshot_file_html)
            self.logger.info(f"Memory snapshot saved to {self.snapshot_file}")
        except Exception as e:
            self.logger.error(f"Failed to save memory snapshot: {e}")

        try:
            torch.cuda.memory._record_memory_history(enabled=None)
            self.logger.info("Memory profiling stopped")
        except Exception as e:
            self.logger.error(f"Failed to stop memory profiling: {e}")


def create_phantom(shape_xyct: Tuple, acc: float = 3.0, ac_lines: int = 24):
    phantom = Phantom.get_shepp_logan(
        shape=shape_xyct[:2], num_coils=shape_xyct[-2], num_echoes=shape_xyct[-1]
    )
    return phantom.sub_sample_ac_random_lines(acceleration=acc, ac_lines=ac_lines)

