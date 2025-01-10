import os
import torch

from pymritools.recon.loraks_dev.matrix_indexing import get_linear_indices, get_linear_indices_mod
from pymritools.recon.loraks_dev.operators import s_operator_mem_opt, s_operator_mem_opt_b

import plotly.graph_objects as go
import polars as pl
from tests.utils import get_test_result_output_dir


def test_s_matrix_creation():
    # set device
    device = torch.device("cuda:0")
    # create k_space shape
    mem = []
    for nc in range(1, 9):
        shape = (256, 256, 1, 4*nc, 4, 1)
        k_space = torch.randn(shape, dtype=torch.complex64, device=device)

        gpu_mem_pre = torch.cuda.mem_get_info(device=device)[0] * 1e-6
        # current indices
        indices, matrix_shape = get_linear_indices(
            k_space_shape=shape, patch_shape=(5, 5, -1, -1, -1, -1), sample_directions=(1, 1, 0, 0, 0, 0)
        )
        indices = indices.to(device)

        # current s_matrix
        s_matrix = s_operator_mem_opt(k_space=k_space, indices=indices, matrix_shape=matrix_shape)
        gpu_mem_post = torch.cuda.mem_get_info(device=device)[0] * 1e-6
        mem.append({
            "nc": nc,
            "version": "current",
            "gpu_mem": gpu_mem_pre - gpu_mem_post,
        })

        del s_matrix
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()

        gpu_mem_pre = torch.cuda.mem_get_info(device=device)[0] * 1e-6
        # current indices
        indices, matrix_shape = get_linear_indices(
            k_space_shape=shape, patch_shape=(5, 5, -1, -1, -1, -1), sample_directions=(1, 1, 0, 0, 0, 0)
        )
        indices = indices.to(device)

        # current s_matrix
        s_matrix_b = s_operator_mem_opt_b(k_space=k_space, indices=indices, matrix_shape=matrix_shape)
        gpu_mem_post = torch.cuda.mem_get_info(device=device)[0] * 1e-6
        mem.append({
            "nc": nc,
            "version": "current b",
            "gpu_mem": gpu_mem_pre - gpu_mem_post,
        })

        del s_matrix_b
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()

        gpu_mem_pre = torch.cuda.mem_get_info(device=device)[0] * 1e-6
        # new version
        indices_mod_p, matrix_shape_mod_p = get_linear_indices_mod(
            k_space_shape=shape, patch_shape=(5, 5, -1, -1, -1, -1), sample_directions=(1, 1, 0, 0, 0, 0)
        )
        indices_mod_m, matrix_shape_mod_m = get_linear_indices_mod(
            k_space_shape=shape, patch_shape=(5, 5, -1, -1, -1, -1), sample_directions=(1, 1, 0, 0, 0, 0),
            flip_directions=(1, 1, 0, 0, 0, 0)
        )

        # new s_matrix
        s_p = k_space.view(-1)[indices_mod_p]
        s_m = k_space.view(-1)[indices_mod_m]
        result = torch.empty(2 * len(indices), 2, device=k_space.device)
        s_p_m = (s_p - s_m)
        result[:len(indices), 0] = s_p_m.real
        result[:len(indices), 1] = -s_p_m.imag
        s_p_m = (s_p + s_m)
        result[len(indices):, 0] = s_p_m.imag
        result[len(indices):, 1] = s_p_m.real

        shape = torch.tensor(matrix_shape_mod_p) + torch.tensor(matrix_shape_mod_m)
        s_matrix_mod = result.view(shape.tolist())

        gpu_mem_post = torch.cuda.mem_get_info(device=device)[0] * 1e-6

        mem.append({
            "nc": nc,
            "version": "new",
            "gpu_mem": gpu_mem_pre - gpu_mem_post,
        })

        del s_matrix_mod, s_p_m, s_m, s_p, result
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()

        assert torch.allclose(s_matrix, s_matrix_mod)
    mem = pl.DataFrame(mem)
    fig = go.Figure()

    m1 = mem.filter(pl.col("version") == "current")
    fig.add_trace(
        go.Scatter(
            x=m1["nc"], y=m1["gpu_mem"], name=f"current version"
        )
    )

    m1 = mem.filter(pl.col("version") == "current b")
    fig.add_trace(
        go.Scatter(
            x=m1["nc"], y=m1["gpu_mem"], name=f"current version b"
        )
    )

    m2 = mem.filter(pl.col("version") == "new")
    fig.add_trace(
        go.Scatter(
            x=m2["nc"], y=m2["gpu_mem"], name=f"new version"
        )
    )

    fig.update_layout(
        title=f"GPU Memory consumption for s_matrix creation of input 256 x 256 x 4*nc x 4",
        xaxis=dict(title="nc"),
        yaxis=dict(title="GPU Memory [MB]")
    )

    output_dir = get_test_result_output_dir(test_s_matrix_creation)
    fn = f"mem_requirement_s_matrix_creation"
    fig.write_html(os.path.join(output_dir, f"{fn}.html"))
