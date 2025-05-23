"""
Sketch for optimizing the sampling scheme from torch autograd gradient loraks reconstruction of fully sampled data

"""

import logging
import torch

from pymritools.utils import Phantom
from tests.utils import get_test_result_output_dir, create_phantom, ResultMode

logger = logging.getLogger(__name__)


def data_driven_subsampling_optimization():
    phantom = create_phantom(shape_xyct=(156, 140, 8, 2))
    path = get_test_result_output_dir(data_driven_subsampling_optimization, mode=ResultMode.EXPERIMENT)
