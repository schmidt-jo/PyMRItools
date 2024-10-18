from .utils import do_performance_test
import pymritools.utils.matrix_operators as op_indexing
import pymritools.recon.loraks.algorithm.opt_operators as loraks_operators


def test_c_k_space_pt_idxs_operator_performance():
    """
    Test for the C operator of LORAKS
    :return:
    """
    nx = 2048
    ny = 2048
    radius = 5
    do_performance_test(loraks_operators.get_C_k_space_pt_idxs, nx, ny, radius)

## Test performance of indexing

def test_get_idx_2d_square_grid_performance():
    nx = 2048
    ny = 2048
    do_performance_test(op_indexing.get_idx_2d_rectangular_grid, nx, ny)

def test_get_idx_2d_grid_circle_within_radius_performance():
    radius = 2048
    import torch._dynamo.config
    old_value = torch._dynamo.config.capture_dynamic_output_shape_ops
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    do_performance_test(op_indexing.get_idx_2d_grid_circle_within_radius, radius)
    torch._dynamo.config.capture_dynamic_output_shape_ops = old_value

def test_get_idx_2d_square_neighborhood_patches_in_shape_performance():
    shape_2d = (1024, 1024)
    nb_size = 5
    do_performance_test(op_indexing.get_idx_2d_square_neighborhood_patches_in_shape, shape_2d, nb_size)

def test_get_idx_2d_rectangular_neighborhood_patches_in_shape_performance():
    shape_2d = (1024, 1024)
    nb_size_x = 5
    nb_size_y = 5
    do_performance_test(op_indexing.get_idx_2d_rectangular_neighborhood_patches_in_shape, shape_2d, nb_size_x, nb_size_y)

def test_get_idx_2d_circular_neighborhood_patches_in_shape_performance():
    shape_2d = (1024, 1024)
    radius = 5
    import torch._dynamo.config
    old_value = torch._dynamo.config.capture_dynamic_output_shape_ops
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    do_performance_test(op_indexing.get_idx_2d_circular_neighborhood_patches_in_shape, shape_2d, radius)
    torch._dynamo.config.capture_dynamic_output_shape_ops = old_value


