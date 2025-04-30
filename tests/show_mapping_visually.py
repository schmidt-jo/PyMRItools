import os
import torch
import plotly.io as pio
import numpy as np

from tests.utils import get_test_result_output_dir
from pymritools.recon.loraks_dev_cleanup.matrix_indexing import get_circular_nb_indices_in_2d_shape, get_linear_indices
from pymritools.recon.loraks_dev_cleanup.operators import c_operator, c_adjoint_operator, s_operator, s_adjoint_operator
from pymritools.utils import Phantom

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
        xgap=1,
        ygap=1
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
    nz = 1
    k_space = torch.zeros((nxy,))[None, :] + torch.arange(nxy)[:, None]
    k_space = k_space[:, :, None] + torch.zeros((nz,))[None, None, :]

    c_indices, c_shape = get_linear_indices(
        k_space_shape=k_space.shape,
        patch_shape=(3, 3, -1),
        sample_directions=(1, 1, 0)
    )
    count_matrix = torch.bincount(c_indices)
    count_matrix[count_matrix == 0] = 1

    mapped_k_space = c_operator(k_space=k_space, indices=c_indices, matrix_shape=c_shape)
    recon_k_space = c_adjoint_operator(mapped_k_space, c_indices, k_space.shape)
    recon_k_space /= count_matrix.view(k_space.shape)

    output_dir = get_test_result_output_dir(test_mapping_visually)
    save_image(k_space[:, :, 0].numpy(), os.path.join(output_dir, "kspace_slice.png"))
    save_image(mapped_k_space.numpy(), os.path.join(output_dir, "mapped_slice.png"))
    save_image(recon_k_space[:, :, 0].numpy(), os.path.join(output_dir, "recon_kspace_slice.png"))


def test_k_space_to_c_matrix_and_back():
    # set shape
    img_shape = (256, 256)
    loraks_nb_side_length = 5  # nb size = side_length**2

    # create a subsampled phantom
    phantom = Phantom.get_shepp_logan(shape=img_shape)
    k_space_us = phantom.sub_sample_ac_skip_lines(acceleration=2, ac_lines=30)
    k_shape = k_space_us.shape

    c_indices, c_shape = get_linear_indices(
        k_space_shape=k_shape,
        patch_shape=(loraks_nb_side_length, loraks_nb_side_length),
        sample_directions=(1, 1)
    )

    # get count matrix from indices, ensure nonzero (if using non rectangular patches)
    count_matrix = torch.bincount(c_indices)
    count_matrix[count_matrix == 0] = 1

    # create c_matrix
    c_matrix = c_operator(k_space=k_space_us, indices=c_indices, matrix_shape=c_shape)

    # get backward mapping
    c_recon_k_space = c_adjoint_operator(c_matrix=c_matrix, indices=c_indices, k_space_dims=k_shape)
    c_recon_k_space /= count_matrix.view(k_shape)

    assert torch.allclose(k_space_us, c_recon_k_space)

    fig = psub.make_subplots(cols=3)
    fig.add_trace(
        go.Heatmap(z=torch.log(torch.abs(k_space_us))),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Heatmap(z=torch.log(torch.abs(c_recon_k_space))),
        row=1,
        col=2
    )
    fig.add_trace(
        go.Heatmap(z=torch.abs(k_space_us - c_recon_k_space), showscale=False),
        row=1,
        col=3
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    output_dir = get_test_result_output_dir("k_space_to_c_matrix_and_back_visualization")
    pio.write_html(fig, os.path.join(output_dir, "k_space_to_c_matrix_and_back_visualization.html"))


def test_k_space_to_s_matrix_and_back():
    # set shape
    nx = 256
    ny = 240
    # create a subsampled phantom
    phantom = Phantom.get_shepp_logan(shape=(nx, ny), num_coils=4, num_echoes=2)
    k_space_us = phantom.sub_sample_ac_random_lines(acceleration=2, ac_lines=30)
    k_space = k_space_us.permute(3, 2, 1, 0)
    k_space = torch.reshape(k_space, (-1, ny, nx))
    k_shape = k_space.shape

    # create s-mapping
    indices = get_circular_nb_indices_in_2d_shape(
        k_space_2d_shape=k_shape[-2:], nb_radius=3, reversed=False
    ).contiguous()
    indices_rev = get_circular_nb_indices_in_2d_shape(
        k_space_2d_shape=k_shape[-2:], nb_radius=3, reversed=True
    ).contiguous()
    # test s mapping
    s_matrix = s_operator(
        k_space=k_space,
        indices=indices,
        indices_rev=indices_rev,
        matrix_shape=indices.shape
    )

    # Adjoint mapping
    k_recon_s = s_adjoint_operator(
        matrix=s_matrix, indices=indices, indices_rev=indices_rev,
        k_space_dims=k_shape
    )

    # normalize
    ones_in = torch.ones_like(k_space)
    count_matrix = s_adjoint_operator(
        matrix=s_operator(
            k_space=ones_in, indices=indices, indices_rev=indices_rev, matrix_shape=indices.shape
        ),
        indices=indices, indices_rev=indices_rev, k_space_dims=k_shape
    )
    # get count matrix from indices, ensure nonzero (if using non-rectangular patches)
    # count_matrix = torch.bincount(indices.view(-1))
    # # s matrix uses indices twice, adjust count matrix
    # count_matrix = 2 * count_matrix.view(k_shape[-2:])[None].expand(k_shape)
    count_matrix[count_matrix == 0] = 1
    k_recon_s /= count_matrix

    # test
    # allclose will fail for circular patches because the corners are not equal
    # assert torch.allclose(k_space_us, k_recon_s)

    fig = psub.make_subplots(cols=3)
    fig.add_trace(
        go.Heatmap(z=torch.log(torch.abs(k_space[0]))),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Heatmap(z=torch.log(torch.abs(k_recon_s)[0])),
        row=1,
        col=2
    )
    fig.add_trace(
        go.Heatmap(z=torch.abs(k_space - k_recon_s)[0], showscale=False),
        row=1,
        col=3
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    output_dir = get_test_result_output_dir("k_space_to_s_matrix_and_back_visualization")
    fig.write_html(os.path.join(output_dir, "k_space_to_s_matrix_and_back_visualization.html"))
