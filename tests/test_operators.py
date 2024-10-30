from .utils import do_performance_test
from pymritools.recon.loraks.operators import get_idx_2d_circular_neighborhood_patches_in_shape


def test_c_k_space_pt_idxs_operator_performance():
    nx = 2048
    ny = 2048
    radius = 5
    do_performance_test(get_idx_2d_circular_neighborhood_patches_in_shape, (nx, ny), radius)

