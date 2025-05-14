from typing import Tuple

import torch
import time
import pathlib as plib
import logging
import pickle

from pymritools.utils import Phantom

pe_dir = plib.Path(__file__).absolute().parent.parent
pe_output_dir = pe_dir.joinpath("pex_output")


def get_output_dir(func) -> plib.Path:
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

    out_dir = pe_output_dir.joinpath(func_name)
    if not out_dir.is_dir():
        out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def create_phantom_data(shape_xyct: Tuple, acc: float = 3.0, ac_lines: int = 24):
    phantom = Phantom.get_shepp_logan(
        shape=shape_xyct[:2], num_coils=shape_xyct[-2], num_echoes=shape_xyct[-1]
    )
    return phantom.sub_sample_ac_random_lines(acceleration=acc, ac_lines=ac_lines)
