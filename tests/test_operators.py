import torch

from tests.utils import do_performance_test

from pymritools.recon.loraks_dev.operators import c_operator, c_adjoint_operator, s_operator, s_adjoint_operator
from pymritools.recon.loraks_dev.matrix_indexing import get_all_idx_nd_square_patches_in_nd_shape
from pymritools.utils.phantom import SheppLogan

import plotly.subplots as psub
import plotly.graph_objects as go


def test_k_space_to_c_matrix_and_back():
    # set shape
    shape = (256, 256)
    loraks_nb_side_length = 5       # nb size = side_length**2

    # create sub - sampled phantom
    k_space_us = SheppLogan().get_sub_sampled_k_space(shape=shape, acceleration=2, ac_lines=30)
    # add dims to get input shape [nx, ny, nz, nc, ne, m]
    k_space_us = k_space_us[:, :, None, :, None, None]
    shape = k_space_us.shape
    # create c-mapping
    c_indices = get_all_idx_nd_square_patches_in_nd_shape(
        size=loraks_nb_side_length, k_space_shape=shape,
        patch_direction=(1,1,0,0,0,0), combination_direction=(0,0,0,1,1,1)
    )

    # get count matrix from indices, ensure nonzero (if using non rectangular patches)
    count_matrix = torch.bincount(c_indices.view(-1))
    count_matrix[count_matrix == 0] = 1

    # create c_matrix
    c_matrix = c_operator(k_space=k_space_us, indices=c_indices)

    # get backward mapping
    c_recon_k_space = c_adjoint_operator(c_matrix=c_matrix, indices=c_indices, k_space_dims=shape)

    c_recon_k_space /= count_matrix

    k_recon_c = c_adjoint_operator(c_matrix=c_matrix, indices=c_indices, k_space_dims=shape)
    k_recon_c /= count_matrix.view(k_recon_c.shape)
    # allclose will fail for circular patches because the corners are not equal
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


def test_k_space_to_s_matrix_and_back():
    # set shape
    shape = (256, 256)
    loraks_nb_side_length = 5       # nb size = side_length**2

    # create sub - sampled phantom
    k_space_us = SheppLogan().get_sub_sampled_k_space(shape=shape, acceleration=2, ac_lines=30)
    # add dims to get input shape [nx, ny, nz, nc, ne, m]
    k_space_us = k_space_us[:, :, None, :, None, None]
    shape = k_space_us.shape
    # create c-mapping
    s_indices = get_all_idx_nd_square_patches_in_nd_shape(
        size=loraks_nb_side_length, k_space_shape=shape,
        patch_direction=(1,1,0,0,0,0), combination_direction=(0,0,0,1,1,1)
    )

    # test s mapping
    s_matrix = s_operator(k_space=k_space_us, indices=s_indices)

    # Adjoint
    k_recon_s = s_adjoint_operator(s_matrix=s_matrix, indices=s_indices, k_space_dims=shape)

    # normalize
    # get count matrix from indices, ensure nonzero (if using non rectangular patches)
    count_matrix = torch.bincount(s_indices.view(-1))
    count_matrix[count_matrix == 0] = 1
    # s matrix uses indices twice, adjust count matrix
    count_matrix_s = 2 * count_matrix.view(shape)
    k_recon_s /= count_matrix_s

    # test
    # allclose will fail for circular patches because the corners are not equal
    print(f"Allclose: {torch.allclose(k_space_us, k_recon_s)}")

    fig = psub.make_subplots(rows=1, cols=3)
    fig.add_trace(
        go.Heatmap(z=torch.log(torch.abs(k_space_us))), row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(z=torch.log(torch.abs(k_recon_s))), row=1, col=2
    )
    fig.add_trace(
        go.Heatmap(z=torch.log(torch.abs(k_space_us - k_recon_s)), showscale=False), row=1, col=3
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.show()



def test_idxs_operator_performance():
    nx = 2048
    ny = 2048
    side_length = 5
    do_performance_test(
        get_all_idx_nd_square_patches_in_nd_shape,
        side_length, (1, 1, 0, 0, 0, 0), (nx, ny, 1, 1, 1, 1), (0, 0, 0, 1, 1, 1)
    )


if __name__ == '__main__':
    test_k_space_to_c_matrix_and_back()
