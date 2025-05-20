import os
import torch

from pymritools.recon.loraks_dev.operators import s_operator
from pymritools.recon.loraks_dev.matrix_indexing import get_linear_indices

from scipy.io import savemat, loadmat

import plotly.graph_objects as go
import plotly.subplots as psub

from tests.utils import get_test_result_output_dir


def new_s_operator(k_space: torch.Tensor, indices: torch.Tensor, matrix_shape: tuple):
    k_flip = torch.flip(k_space, dims=(0, 1))
    s_p = k_space.view(-1)[indices].view(matrix_shape)
    s_m = k_flip.view(-1)[indices].view(matrix_shape)

    matrix_shape_result = tuple([2*s for s in matrix_shape])
    # Todo: Check how to handle floating point size
    dtype = torch.float32 if k_space.dtype == torch.complex64 else torch.float64
    result = torch.zeros(matrix_shape_result, dtype=dtype, device=k_space.device)
    s_p_m = (s_p - s_m)
    result[:matrix_shape[0], :matrix_shape[1]] = s_p_m.real
    result[matrix_shape[0]:, :matrix_shape[1]] = -s_p_m.imag
    s_p_m = (s_p + s_m)
    result[:matrix_shape[0], matrix_shape[1]:] = s_p_m.imag
    result[matrix_shape[0]:, matrix_shape[1]:] = s_p_m.real
    return result

def new_matlike_s_operator(k_space: torch.Tensor, indices: torch.Tensor, matrix_shape: tuple):
    k_flip = torch.flip(k_space, dims=(0, 1))
    s_p = k_space.view(-1, k_space.shape[-1])[indices].view(*matrix_shape, -1)
    s_m = k_flip.view(-1, k_space.shape[-1])[indices].view(*matrix_shape, -1)

    s_p_m = (s_p - s_m)
    s_u = torch.concatenate([s_p_m.real, -s_p_m.imag], dim=1)
    s_p_m = (s_p + s_m)
    s_d = torch.concatenate([s_p_m.imag, s_p_m.real], dim=1)

    s = torch.concatenate([s_u, s_d], dim=0)
    s = torch.concatenate([s_sub for s_sub in s.permute(2, 0, 1)], dim=-1)
    return s


def test_s_matrix_build():
    nb_size = 6
    shape = (5, 30, 20)
    input_k = torch.arange(torch.prod(torch.tensor(shape)).item()).reshape(shape).to(torch.complex128)
    input_k = input_k.permute(1, 2, 0).contiguous()
    shape = input_k.shape
    savemat(
        "/data/pt_np-jschmidt/code/PyMRItools/resources/vs_matlab_tests/loraks_phantom/data_arange_30x20x5.mat",
        {"data": input_k},
    )

    indices, matrix_shape = get_linear_indices(
        k_space_shape=shape, patch_shape=(nb_size, nb_size, -1), sample_directions=(1, 1, 0)
    )
    mat_indices, mat_matrix_shape = get_linear_indices(
        k_space_shape=shape[:2], patch_shape=(nb_size, nb_size), sample_directions=(1, 1)
    )
    matrix_shape_s = tuple([2*s for s in matrix_shape])

    s_matrix = s_operator(
        k_space=input_k, indices=indices, matrix_shape=matrix_shape_s
    )
    s_matrix_new = new_s_operator(
        k_space=input_k, indices=indices, matrix_shape=matrix_shape
    )

    s_matrix_new_mat = new_matlike_s_operator(
        k_space=input_k, indices=mat_indices, matrix_shape=mat_matrix_shape
    )

    s_matrix_mat = loadmat(
        "/data/pt_np-jschmidt/code/PyMRItools/resources/vs_matlab_tests/loraks_test_data/s_matrix.mat"
    )
    s_matrix_mat = torch.from_numpy(s_matrix_mat["s_matrix"]).mT

    zmax = torch.abs(input_k).max().item()

    fig = psub.make_subplots(
        rows=5, cols=5,
        specs=[
            [{}, {}, {}, {}, {}],
            [{"colspan": 5}, None, None, None, None],
            [{"colspan": 5}, None, None, None, None],
            [{"colspan": 5}, None, None, None, None],
            [{"colspan": 5}, None, None, None, None]
        ]
    )
    for c in range(input_k.shape[-1]):
        fig.add_trace(
            go.Heatmap(z=torch.abs(input_k[..., c]), zmin=0, zmax=zmax),
            row=1, col=1+c
        )
    for i, d in enumerate([s_matrix, s_matrix_new, s_matrix_new_mat, s_matrix_mat]):
        fig.add_trace(
            go.Heatmap(z=torch.abs(d), zmin=0, zmax=1.5*zmax, transpose=False),
            row=2+i, col=1
        )
    output_dir = get_test_result_output_dir(test_s_matrix_build)

    fig.write_html(os.path.join(output_dir, "s_matrix_vis.html"))




