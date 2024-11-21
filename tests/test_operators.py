from .utils import do_performance_test
from pymritools.recon.loraks.operators import get_idx_2d_circular_neighborhood_patches_in_shape

# TODO: write test method that:
#       1. create kspace for phantom
#       2. create c-mapping and calculate c-matrix
#       3. uses the adjoint c op to backproject into original k-space

def test_c_k_space_pt_idxs_operator_performance():
    nx = 2048
    ny = 2048
    radius = 5
    do_performance_test(get_idx_2d_circular_neighborhood_patches_in_shape, (nx, ny), radius)

