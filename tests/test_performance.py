from .utils import do_performance_test
import torch
import pymritools.utils.matrix_indexing as op_indexing
import pymritools.recon.loraks.operators as op_operators

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

def test_c_operator():
    nx = 512
    ny = 512
    radius = 5
    k_space = torch.randn((nx, ny, 1, 1), dtype=torch.complex64)
    c_mapping = op_indexing.get_idx_2d_circular_neighborhood_patches_in_shape((nx, ny), radius)
    do_performance_test(op_operators.c_operator, k_space, c_mapping)

def test_c_adjoint_operator():
    nx = 512
    ny = 512
    radius = 5
    k_space = torch.randn((nx, ny, 1, 1), dtype=torch.complex64)
    c_mapping = op_indexing.get_idx_2d_circular_neighborhood_patches_in_shape((nx, ny), radius)
    c_matrix = op_operators.c_operator(k_space, c_mapping)
    do_performance_test(op_operators.c_adjoint_operator, c_matrix, c_mapping, (nx, ny))

