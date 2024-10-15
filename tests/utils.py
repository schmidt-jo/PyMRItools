import torch
import time
import os

test_dir = os.path.dirname(__file__)
test_output_dir = os.path.join(os.path.dirname(test_dir), "test_output")

def get_test_result_output_dir(func):
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

    out_dir = os.path.join(test_output_dir, func_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return os.path.abspath(out_dir)


def measure_function_call(func, *args, iterations=10):
    """
    Measures the execution time of a function call over multiple iterations.
    Can be called on any function but should not be used for torch compiled code.

    :param func: The function to measure the execution time of.
    :param args: The arguments to pass to the function during its call.
    :param iterations: The number of times to call the function. The default is 10.
    :return: The average execution time per call, in seconds.
    """
    start_time = time.time()
    for _ in range(iterations):
        func(*args)
    end_time = time.time()
    elapsed_time = (end_time - start_time) / iterations
    return elapsed_time

def measure_compiled_function_call(func, *args, iterations=10):
    """
    Measures the execution time of a torch compiled function call over multiple iterations.
    This only makes sense for torch functions specifically written to be compiled.

    :param func: The function to be compiled and executed.
    :param args: Arguments to be passed to the function during execution.
    :param iterations: The number of times to execute the function for performance measurement. Default is 10.
    :return: The average execution time in seconds of the compiled function over the specified number of iterations.
    """
    compiled_func = torch.compile(func)

    # Warmup phase
    # This should be necessary as I'm not sure if the compilation takes place on the call or during the first run.
    for _ in range(3):
        compiled_func(*args)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        compiled_func(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations / 1000

def profile_torch_function(func, *args):

    compiled_func = torch.compile(func)

    # Warmup phase
    for _ in range(3):
        compiled_func(*args)

    with torch.profiler.profile() as prof:
        compiled_func(*args)
        prof.step()
    prof.export_chrome_trace(os.path.join(get_test_result_output_dir(func), "trace.json"))


def do_performance_test(func, *args, iterations=10):
    has_cuda = torch.cuda.is_available()

    time_non_compiled = measure_function_call(func, *args, iterations=iterations)
    time_compiled = None
    if has_cuda:
        compiled_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                compiled_args.append(arg.cuda())
            else:
                compiled_args.append(arg)

        time_compiled = measure_compiled_function_call(func, *compiled_args, iterations=iterations)
        profile_torch_function(func, *compiled_args)

    results_file = os.path.join(get_test_result_output_dir(func), "results.txt")
    with open(results_file, "w") as f:
        f.write(f"Non-compiled function took: {time_non_compiled:.6f} seconds per iteration\n")
        if has_cuda:
            f.write(f"Compiled function took: {time_compiled:.6f} seconds per iteration\n")
            f.write(f"Speed-up: {time_non_compiled / time_compiled:.2f}x\n")