import os

import torch
import plotly.graph_objects as go
import plotly.colors as plc
import polars as pl
from mpmath.matrices.matrices import colsep

from pymritools.utils.phantom import SheppLogan
from pymritools.recon.loraks_dev.operators import c_operator, s_operator, c_adjoint_operator, s_adjoint_operator
from pymritools.recon.loraks_dev.matrix_indexing import get_all_idx_nd_square_patches_in_nd_shape
from pymritools.utils.algorithms import subspace_orbit_randomized_svd, randomized_svd

from tests.utils import get_test_result_output_dir


def iteration_per_size(nx, ny, nc, ne, ranks, device):

    # set svd iterations
    svd_names = ["svd", "svd_lowrank", "r_svd", "sor_svd"]
    # set loraks operations
    op_names = ["c", "s"]

    mem_track_matrix = []
    mem_track_size_svd = []

    # setup mappings
    indices, reshape = get_all_idx_nd_square_patches_in_nd_shape(
        size=5, k_space_shape=(nx, ny, nc, ne), patch_direction=(1, 1, 0, 0), combination_direction=(0, 0, 1, 1),
    )
    # track memory
    gpu_mem_pre = torch.cuda.mem_get_info(device=device)[0] * 1e-6
    indices = indices.to(device)[0]

    gpu_mem_post = torch.cuda.mem_get_info(device=device)[0] * 1e-6

    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    mem_track_matrix.append({
        "point": "matrix_indexing", "gpu_use": gpu_mem_pre - gpu_mem_post
    })

    # create shepp logan using its methods. The seed is set.
    # The impact of non-square slice dimensions should be neglicible
    sl_us_phantom = SheppLogan().get_sub_sampled_k_space(
        shape=(nx, ny), acceleration=3, ac_lines=20, mode="skip", as_torch_tensor=True,
        num_coils=nc, num_echoes=ne
    )
    k_space = torch.reshape(sl_us_phantom, reshape)

    # push to device
    # track memory
    gpu_mem_pre = torch.cuda.mem_get_info(device=device)[0] * 1e-6
    k_space = k_space.to(device)[0]
    gpu_mem_post = torch.cuda.mem_get_info(device=device)[0] * 1e-6
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    # track memory
    mem_track_matrix.append({
        "point": "k-space", "gpu_use": gpu_mem_pre - gpu_mem_post
    })

    # want to build c and s matrices
    for i, op in enumerate([c_operator, s_operator]):
        gpu_mem_pre = torch.cuda.mem_get_info(device=device)[0] * 1e-6
        # matrix
        matrix = op(k_space=k_space, indices=indices)

        gpu_mem_post = torch.cuda.mem_get_info(device=device)[0] * 1e-6
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()

        # track memory
        mem_track_matrix.append({
            "size": f"{nx}_{ny}_{nc}_{ne}",
            "point": f"{op_names[i]}-matrix",
            "gpu_use": gpu_mem_pre - gpu_mem_post
        })

        # iterate through svd variants
        for i_svd, svd in enumerate([
            torch.linalg.svd, torch.svd_lowrank, randomized_svd, subspace_orbit_randomized_svd
        ]):
            # track memory
            gpu_mem_pre = torch.cuda.mem_get_info(device=device)[0] * 1e-6
            # circle rank
            for i_r, rank in enumerate(ranks):
                if i_svd == 0:
                    if i_r > 0:
                        continue
                    u, s, vh = svd(matrix, full_matrices=False)
                elif i_svd == 1:
                    u, s, vh = svd(A=matrix, q=rank + 2, niter=2)
                else:
                    u, s, vh = svd(matrix=matrix, q=rank + 2, power_projections=2)
                gpu_mem_post = torch.cuda.mem_get_info(device=device)[0] * 1e-6
                # track memory
                del u, s, vh
                torch.cuda.empty_cache()
                torch.cuda.reset_max_memory_allocated()

                mem_track_size_svd.append({
                    "size": f"{nx}_{ny}_{nc}_{ne}",
                    "svd": svd_names[i_svd],
                    "rank": rank,
                    "op": op_names[i],
                    "gpu_use": gpu_mem_pre - gpu_mem_post
                })
    return mem_track_matrix, mem_track_size_svd

def test_memory_requirements():
    sizes = torch.arange(1,6)
    num_ops = 2
    num_svd = 4
    ranks = [20, 50, 100, 200, 300]
    num_ranks = len(ranks)

    channels = [4, 8, 16, 32]
    echoes = [1, 2, 4, 6]

    num_colors = num_ops * num_svd * num_ranks
    col_sep = 3
    cmap = plc.sample_colorscale("Turbo", torch.linspace(0.1, 0.9, col_sep*num_colors).tolist())

    # set svd iterations
    svd_names = ["svd", "svd_lowrank", "r_svd", "sor_svd"]
    # set loraks operations
    op_names = ["c", "s"]
    mem_track_sizes = []
    mem_track_svds = []

    for i_size, size in enumerate(sizes.tolist()):
        # set slice size
        nx, ny = 64 * size, 64 * size
        # set device
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        for i_c, nc in enumerate(channels):
            for i_e, ne in enumerate(echoes):
                mtps, mtss = iteration_per_size(
                    nx=nx, ny=ny, nc=nc, ne=ne, ranks=ranks, device=device
                )
                mem_track_sizes.extend(mtps)
                mem_track_svds.extend(mtss)

    mem = pl.DataFrame(mem_track_sizes)

    fig = go.Figure()
    for ni, n in enumerate(svd_names):
        for no, o in enumerate(op_names):
            for ri, r in enumerate(ranks):
                idx = colsep * (ri * 2 + no) * num_svd
                t = mem.filter(pl.col("svd") == n).filter(pl.col("op") == o).filter(pl.col("rank") == r)
                name = f"{n}-{o}"
                if ni > 0:
                    name += f"-rank-{r}"
                fig.add_trace(
                    go.Bar(
                        x=t["size"], y=t["gpu_use"], name=name,
                        marker=dict(color=cmap[idx])
                    )
                )
    fig.update_layout(
        xaxis=dict(title="Operation"),
        yaxis=dict(title="Memory (MB)"),
        title=dict(
            text="GPU Memory requirements for different operations",
            x=0.5,
            xanchor="center",
            yanchor="top"
        )
    )
    output_dir = get_test_result_output_dir(test_memory_requirements)
    fn = f"mem_requirement_test_sizes"
    fig.write_html(os.path.join(output_dir, f"{fn}.html"))
