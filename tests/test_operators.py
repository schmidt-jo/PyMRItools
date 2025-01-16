from tests.utils import do_performance_test
from pymritools.recon.loraks_dev.matrix_indexing import get_all_idx_nd_square_patches_in_nd_shape

def test_idxs_operator_performance():
    nx = 2048
    ny = 2048
    side_length = 5
    do_performance_test(
        get_all_idx_nd_square_patches_in_nd_shape,
        side_length, (1, 1, 0, 0, 0, 0), (nx, ny, 1, 1, 1, 1), (0, 0, 0, 1, 1, 1)
    )
