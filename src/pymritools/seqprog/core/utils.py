import numpy as np
from pypulseq import Opts


def set_on_grad_raster_time(system: Opts, time: float, double: bool = False):
    grad_raster = system.grad_raster_time
    if double:
        grad_raster *= 2
    return np.ceil(time / grad_raster) * grad_raster

def set_on_rf_raster_time(system: Opts, time: float):
    return np.ceil(time / system.rf_raster_time) * system.rf_raster_time

def check_raster(value, raster):
    round_val = np.round(value / raster)
    rounded_val = round_val * raster
    return np.allclose(rounded_val, value)
