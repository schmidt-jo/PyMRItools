import os
import torch
import plotly.io as pio
import numpy as np

from tests.utils import get_test_result_output_dir
from pymritools.recon.loraks_dev.matrix_indexing import get_all_idx_nd_square_patches_in_nd_shape
from pymritools.recon.loraks_dev.operators import c_operator, c_adjoint_operator, s_operator, s_adjoint_operator
from pymritools.utils.phantom import SheppLogan

import plotly.subplots as psub
import plotly.graph_objects as go


def save_image(matrix: np.ndarray, output_file: str):
    """
    Saves a 2D NumPy array as an image file.

    :param matrix: A 2D NumPy array representing the pixel intensity values to be visualized as a heatmap.
    :param output_file: A string representing the file path where the output PNG image will be saved.
    :return: None
    """
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        colorscale="Jet",
        showscale=False,
        xgap = 1,
        ygap = 1
    ))

    fig.update_layout(
        width=800,
        height=800,
    )
    pio.write_image(fig, output_file)


def test_mapping_visually():
    """
    This needs to be generalized. We want a way to visually check if our mapping is correct.
    Here, we create a fake k-space where every read-line is just 0, 1, 2, ..., nxy-1.
    This pattern will be repeated for each phase encoding direction so that we get an image
    with gradually increasing color on each line. This slice will simply be copied for all slices.
    Note that if you look at the images, they will be transposed since we're now using indexing
    (nx, ny, nz, ...).
    Output files will be stored in the test_output directory.
    """
    nxy = 8
    nz = 3
    k_space = torch.zeros((nxy,))[None, :] + torch.arange(nxy)[:, None]
    k_space = k_space[:, :, None] + torch.zeros((nz,))[None, None, :]

    radius = 3
    indices = get_all_idx_nd_square_patches_in_nd_shape(
        size=radius,
        patch_direction=(1, 1, 0),
        k_space_shape=k_space.shape,
        combination_direction=(0, 0, 1)
    )
    count_matrix = torch.bincount(indices.view(-1))
    count_matrix[count_matrix == 0] = 1

    mapped_k_space = c_operator(k_space=k_space, indices=indices)
    recon_k_space = c_adjoint_operator(mapped_k_space, indices, k_space.shape)
    recon_k_space /= count_matrix.view(k_space.shape)

    output_dir = get_test_result_output_dir(test_mapping_visually)
    save_image(k_space[:,:,0].numpy(), os.path.join(output_dir, "kspace_slice.png"))
    save_image(mapped_k_space[0].numpy(), os.path.join(output_dir, "mapped_slice.png"))
    save_image(recon_k_space[:, :, 0].numpy(), os.path.join(output_dir, "recon_kspace_slice.png"))

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

    # create sub-sampled phantom
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
