import os
import torch
import plotly.io as pio
import plotly.graph_objects as go
import numpy as np

from tests.utils import get_test_result_output_dir
from pymritools.recon.loraks_dev.matrix_indexing import get_all_idx_nd_square_patches_in_nd_shape
from pymritools.recon.loraks_dev.operators import c_operator, c_adjoint_operator

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

