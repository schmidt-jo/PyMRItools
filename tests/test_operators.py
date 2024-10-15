from .utils import do_performance_test
from pymritools.recon.loraks.algorithm.opt_operators import get_C_k_space_pt_idxs


def test_c_k_space_pt_idxs_operator_performance():
    nx = 2048
    ny = 2048
    radius = 5
    do_performance_test(get_C_k_space_pt_idxs, nx, ny, radius)

