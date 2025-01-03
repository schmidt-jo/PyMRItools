import os

import torch
import plotly.graph_objects as go
import polars as pl

from pymritools.utils.phantom import SheppLogan
from pymritools.recon.loraks_dev.operators import c_operator, s_operator, c_adjoint_operator, s_adjoint_operator
from pymritools.recon.loraks_dev.matrix_indexing import get_all_idx_nd_square_patches_in_nd_shape
from pymritools.utils.algorithms import subspace_orbit_randomized_svd, randomized_svd

from tests.utils import get_test_result_output_dir



def test_memory_requirements():
    mem_track_per_size = []
    for i_size, size in enumerate(torch.arange(1,6)):
        # set slice size
        nx, ny = 64*size, 64*size
        # set number of channels and echoes
        nc, ne = 2*size, size
        # set device
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        mem_track = []

        # set svd iteration
        svd_names = ["svd", "svd_lowrank", "r_svd", "sor_svd"]

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

        mem_track.append({
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
        mem_track.append({
            "point": "k-space", "gpu_use": gpu_mem_pre - gpu_mem_post
        })

        op_names = ["c", "s"]
        # want to build c and s matrices
        for i, op in enumerate([c_operator, s_operator]):
            gpu_mem_pre = torch.cuda.mem_get_info(device=device)[0] * 1e-6
            # matrix
            matrix = op(k_space=k_space, indices=indices)

            gpu_mem_post = torch.cuda.mem_get_info(device=device)[0] * 1e-6
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()

            # track memory
            mem_track.append({
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
                for i_r, rank in enumerate([20, 50, 100, 200]):
                    if i_svd == 0:
                        if i_r > 0:
                            continue
                        u, s, vh = svd(matrix, full_matrices=False)
                    elif i_svd == 1:
                        u, s, vh = svd(A=matrix, q=rank+2, niter=2)
                    else:
                        u, s, vh = svd(matrix=matrix, q=rank+2, power_projections=2)
                    gpu_mem_post = torch.cuda.mem_get_info(device=device)[0] * 1e-6
                    # track memory
                    mem_track.append({
                        "point": f"{op_names[i]}-{svd_names[i_svd]}",
                        "gpu_use": gpu_mem_pre - gpu_mem_post,
                        "rank": rank+2
                    })
                    del u, s, vh
                    torch.cuda.empty_cache()
                    torch.cuda.reset_max_memory_allocated()

                    mem_track_per_size.append({
                        "size": f"{nx.item()}_{ny.item()}_{nc.item()}-{ne.item()}",
                        "svd": svd_names[i_svd],
                        "rank": rank,
                        "op": op_names[i],
                        "gpu_use": gpu_mem_pre - gpu_mem_post
                    })

        # mem = pl.DataFrame(mem_track)

        # fig = go.Figure()
        # fig.add_trace(
        #     go.Bar(
        #         x=mem["point"], y=mem["gpu_use"],
        #
        #     )
        # )
        # fig.update_layout(
        #     xaxis=dict(title="Operation"),
        #     yaxis=dict(title="Memory (MB)"),
        #     title=dict(
        #         text="GPU Memory requirements for different operations",
        #         x=0.5,
        #         xanchor="center",
        #         yanchor="top"
        #     )
        # )
        # output_dir = get_test_result_output_dir(test_memory_requirements)
        # fn = f"mem_req_per_size-{nx.item()}_{ny.item()}_{nc.item()}_{ne.item()}"
        # fig.write_html(os.path.join(output_dir, f"{fn}.html"))

    mem = pl.DataFrame(mem_track_per_size)

    fig = go.Figure()
    for ni, n in enumerate(svd_names):
        for no, o in enumerate(op_names):
            for ri, r in enumerate([20, 50, 100, 200]):
                t = mem.filter(pl.col("svd") == n).filter(pl.col("op") == o).filter(pl.col("rank") == r)
                name = f"{n}-{o}"
                if ni > 0:
                    name += f"-rank-{r}"
                fig.add_trace(
                    go.Bar(
                        x=t["size"], y=t["gpu_use"], name=name
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
