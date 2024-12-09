import torch

from tests.utils import do_performance_test

from pymritools.recon.loraks.operators import (
    get_idx_2d_circular_neighborhood_patches_in_shape, c_operator, c_adjoint_operator
)
from pymritools.utils.phantom import SheppLogan

import plotly.subplots as psub
import plotly.graph_objects as go


def test_k_space_to_c_matrix_and_back():
    # set shape
    shape = (256, 256)
    loraks_radius = 3
    # create c-mapping
    c_indices = get_idx_2d_circular_neighborhood_patches_in_shape(shape_2d=shape, nb_radius=loraks_radius)

    # get count matrix
    in_ones = torch.ones(shape, device=torch.device("cpu"), dtype=torch.complex64)
    forward_matrix = c_operator(in_ones[:, :, None, None], c_indices)
    count_matrix = c_adjoint_operator(
        c_matrix=forward_matrix, indices=c_indices, k_space_dims=shape
    ).real.to(torch.int)

    # create sub - sampled phantom
    k_space_us = SheppLogan().get_sub_sampled_k_space(shape=shape, acceleration=2, ac_lines=30)
    # create c_matrix
    c_matrix = c_operator(k_space=k_space_us[:, :, None, None], indices=c_indices)
    # get backward mapping
    c_recon_k_space = c_adjoint_operator(c_matrix=c_matrix, indices=c_indices, k_space_dims=shape)
    c_recon_k_space[count_matrix>0] /= count_matrix[count_matrix>0]
    # allclose should fail because the corners are not equal
    print(f"Allclose: {torch.allclose(k_space_us, c_recon_k_space)}")

    fig = psub.make_subplots(rows=1, cols=3)
    fig.add_trace(
        go.Heatmap(z=torch.log(torch.abs(k_space_us))), row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(z=torch.log(torch.abs(c_recon_k_space))), row=1, col=2
    )
    fig.add_trace(
        go.Heatmap(z=torch.log(torch.abs(k_space_us - c_recon_k_space)), showscale=False), row=1, col=3
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.show()

def test_c_k_space_pt_idxs_operator_performance():
    nx = 2048
    ny = 2048
    radius = 5
    do_performance_test(get_idx_2d_circular_neighborhood_patches_in_shape, (nx, ny), radius)


if __name__ == '__main__':
    test_k_space_to_c_matrix_and_back()
