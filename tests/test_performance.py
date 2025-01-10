from pymritools.recon.loraks_dev.matrix_indexing import get_linear_indices
from pymritools.recon.loraks_dev.ps_loraks import create_loss_func, get_lowrank_algorithm_function, \
    LowRankAlgorithmType, get_sv_threshold_function, SVThresholdMethod, Loraks
from .utils import do_performance_test, measure_cuda_function_call
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
    do_performance_test(op_indexing.get_idx_2d_rectangular_neighborhood_patches_in_shape, shape_2d, nb_size_x,
                        nb_size_y)


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


def test_loraks_loss_function():
    k_space_shape = (256, 256, 8, 4)
    patch_shape = (5, 5, -1, -1)
    sample_directions = (1, 1, 0, 0)
    lambda_factor = 0.3
    device = torch.device("cuda")
    k_space = torch.randn(k_space_shape, dtype=torch.complex64, device=device)
    sampling_mask = torch.randn_like(k_space, dtype=torch.float, device=device) > 0.7
    k_sampled_points = sampling_mask * k_space
    indices, c_matrix_shape = get_linear_indices(k_space_shape, patch_shape, sample_directions)
    s_matrix_shape = tuple(2*entry for entry in c_matrix_shape)

    k_candidate = k_space.clone().requires_grad_()

    from pymritools.recon.loraks_dev.operators import s_operator_mem_opt

    operator_func = s_operator_mem_opt
    svd_func = get_lowrank_algorithm_function(LowRankAlgorithmType.TORCH_LOWRANK_SVD, (40, 2))
    sv_threshold_func = get_sv_threshold_function(SVThresholdMethod.HARD_CUTOFF, (40, 20, device))
    loss_func = create_loss_func(
        operator_func,
        svd_func,
        sv_threshold_func
    )
    print(
      measure_cuda_function_call(loss_func, k_candidate, indices, s_matrix_shape, k_sampled_points, sampling_mask, lambda_factor, device=torch.device("cpu"))
    )

def test_lowrank_function():
    k_space_shape = (256, 256, 8, 4)
    patch_shape = (5, 5, -1, -1)
    sample_directions = (1, 1, 0, 0)
    lambda_factor = 0.3
    k_space = torch.randn(k_space_shape, dtype=torch.complex64)
    sampling_mask = torch.randn_like(k_space, dtype=torch.float) > 0.7
    k_sampled_points = sampling_mask * k_space
    indices, c_matrix_shape = get_linear_indices(k_space_shape, patch_shape, sample_directions)

    final_matrix = k_space.view(-1)[indices].view(c_matrix_shape)

    svd_func = get_lowrank_algorithm_function(LowRankAlgorithmType.TORCH_LOWRANK_SVD, (40, 2))
    # We can't test performance on the SVThresholdMethod.HARD_CUTOFF because on creation of the threshold function,
    # we already inject the threshold tensor that we multiply with the singular values vector.
    # This tensor lives on a defined device, and our trickery of moving arguments to specified devices in the
    # do_performance_test function doesn't work anymore.
    do_performance_test(svd_func, final_matrix)


def test_reconstruction():
    k_space_shape = (1, 256, 256, 8, 4)
    patch_shape = (5, 5, -1, -1)
    sample_directions = (1, 1, 0, 0)
    k_space = torch.randn(k_space_shape, dtype=torch.complex64)
    sampling_mask = torch.randn_like(k_space, dtype=torch.float) > 0.7
    l = (Loraks()
         .with_patch_shape(patch_shape)
         .with_sample_directions(sample_directions)
         .with_torch_lowrank_algorithm(50, 2)
         .with_c_matrix()
         .with_sv_soft_cutoff(15.0)
         )
    l.reconstruct(k_space, sampling_mask)
