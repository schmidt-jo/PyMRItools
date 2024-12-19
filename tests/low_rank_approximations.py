import json
import os

import plotly.graph_objects as go
import plotly.subplots as psub
import plotly.colors as plc
import pytest
import torch
import torch.linalg as LA
import numpy as np

from pymritools.utils.algorithms import randomized_svd
from pymritools.utils.algorithms import subspace_orbit_randomized_svd
from tests.utils import do_performance_test
from tests.utils import get_test_result_output_dir


def generate_noisy_low_rank_matrix(n: int, k: int, error_scale:float = 3.0) -> torch.Tensor:
    """
    Generate a matrix with controlled singular values and added noise
    like in the paper (DOI:10.1109/TSP.2018.2853137).
    """
    sigma_max = 1.0
    sigma_min = 1e-9
    # Note that I had to scale the random noise error way up to see a decrease in quality.
    s = torch.linspace(sigma_max, sigma_min, n)
    s[k:] = 0  # Set values after k to zero (note: Python uses 0-based indexing)
    G = torch.randn(n, n)
    E = G / LA.norm(G)
    Q1, _ = LA.qr(torch.randn(n, n))
    Q2, _ = LA.qr(torch.randn(n, n))
    return (Q1 * s) @ Q2.T + error_scale * s[k - 5] * E


def generate_low_rank_matrix(m: int, n: int, rank: int, dtype=torch.float32, device="cpu") -> torch.Tensor:
    """
    Generates a low-rank matrix A = B @ C where B and C are low-rank components.
    """
    B = torch.randn((m, rank), dtype=dtype, device=device)
    C = torch.randn((rank, n), dtype=dtype, device=device)
    return torch.matmul(B, C)


def create_singular_value_plot(
        sigmas: list, names: list,
        x_len: int,
        file_name: str):
    fig = go.Figure()
    for i, s in enumerate(sigmas):
        fig.add_trace(
            go.Scatter(y=s, name=names[i], mode="lines+markers")
        )
        fig.update_xaxes(range=(0, x_len))
        fig.update_layout(
            title="Singular values",
            xaxis_title="Index",
            yaxis_title="Singular value",
            legend_title="Method"
        )
    if file_name.endswith(".html"):
        fig.write_html(file_name)
    else:
        fig.write_image(file_name)


def test_show_singular_value_recovery():
    """
    Test visually how well singular values are recovered from big low-rank matrices by different methods.
    These plots are comparable to what can be found in figure 1 of the SOR-SVD paper (doi:10.1109/TSP.2018.2853137).
    """
    n = 1000  # size (n, n) of the example matrix
    k = 20  # low-rank of the created example matrix

    # Sampling size of the smaller matrix. If you want to have a rank k approximation of a big matrix
    # the size of q should be greater than k. Setting q = k + 0..2 should be enough.
    q = 30

    m = generate_noisy_low_rank_matrix(n, k)

    # Ground truth singular values for comparison
    _, s_svd, _ = torch.linalg.svd(m, full_matrices=False)

    # The gist is that all three methods seem to deliver equal quality when using 2 power-iterations which
    # is the default for torch.svd_lowrank. Also, it really seems that if we want to have rank k, we can get
    # away using k+0..2 as size. This is actually written in the docs to torch.svd_lowrank.
    fig = psub.make_subplots(
        rows=3, cols=1,
        row_titles=["Pow.-iter 0", "Pow.-iter 1", "Pow.-iter 2"],
        shared_xaxes=True
    )
    cmap = plc.sample_colorscale("Turbo", np.linspace(0.2, 0.9, 4))
    for pi in range(3):
        _, s_sor, _ = subspace_orbit_randomized_svd(m, q, power_projections=pi)
        _, s_rand, _ = randomized_svd(m, q, power_projections=pi)
        _, s_torch, _ = torch.svd_lowrank(m, q, niter=pi)
        names = ["SVD", "SOR-SVD", "RSVD", "TorchLR"]
        for i, s in enumerate([s_svd, s_sor, s_rand, s_torch]):
            showlegend = True if pi == 0 else False
            fig.add_trace(
                go.Scatter(
                    y=s, name=names[i], mode="lines+markers",
                    marker=dict(color=cmap[i]), legendgroup=i,
                    showlegend=showlegend
                ),
                row=pi+1, col=1
            )

    fig.update_xaxes(range=(0, q-1), title="Index")
    fig.update_yaxes(range=(0, 1.1), title="Singular value")
    fig.update_layout(legend_title="Method")
    output_dir = get_test_result_output_dir(test_show_singular_value_recovery)
    fig.write_html(os.path.join(output_dir, "singular_values_svds_power-iter.html"))
    # create_singular_value_plot(
    #     sigmas=[s_svd, s_sor, s_rand, s_torch], names=["SVD", "SOR-SVD", "RSVD", "TorchLR"], x_len=q,
    #     file_name=os.path.join(output_dir, f"singular_values_svds_power-iter-{pi}.html")
    # )
    # create_singular_value_plot(
    #     sigmas=[s_svd, s_rand], names=["SVD", "RSVD"], x_len=q,
    #     file_name=os.path.join(output_dir, "singular_values_rand.png")
    # )

def gold_standard_svd(matrix: torch.Tensor, rank: int) -> tuple[torch.Tensor, ...]:
    u, s, v = torch.linalg.svd(matrix, full_matrices=False)
    u_trunc = u[:, :rank].contiguous()
    s_trunc = s[:rank]
    v_trunc = v[:rank, :].contiguous()
    return u_trunc, s_trunc, v_trunc

def calculate_svd_error(original_matrix: torch.Tensor, u: torch.Tensor, sigma: torch.Tensor, vh: torch.Tensor) -> dict:
    approx_matrix = (u * sigma) @ vh
    frobenius_error = torch.norm(original_matrix - approx_matrix, p="fro")
    spectral_error = torch.linalg.norm(original_matrix - approx_matrix, ord=2)
    frobenius_relative_error = frobenius_error / torch.norm(original_matrix, p="fro")
    spectral_relative_error = spectral_error / torch.linalg.norm(original_matrix, ord=2)
    error_metrics = {
        "Frobenius Norm Error": frobenius_error.item(),
        "Spectral Norm Error": spectral_error.item(),
        "Relative Frobenius Norm Error": frobenius_relative_error.item(),
        "Relative Spectral Norm Error": spectral_relative_error.item(),
    }
    return error_metrics


@pytest.mark.parametrize("m, n, rank", [
    (4096, 512, 10),  # Test case 1: Tall matrix
    (4096, 4096, 10),  # Test case 2: Square matrix
    (8192, 64, 5),  # Test case 3: Tall skinny matrix
])
def test_low_rank_approximation_quality(m: int, n: int, rank: int):
    """Test the low-rank approximation quality of the randomized SVD."""
    torch.manual_seed(42)
    matrix = generate_low_rank_matrix(m, n, rank=rank)

    # Compute the gold standard SVD solution
    u_gold, s_gold, v_gold = gold_standard_svd(matrix, rank)

    # Compute the low-rank approximation using randomized SVD implementation
    u_rand, s_rand, v_rand = randomized_svd(matrix, rank)

    # Compute the low-rank approximation using Jochen's SOR_SVD implementation
    u_sor, s_sor, v_sor = subspace_orbit_randomized_svd(matrix, rank)

    # Compute the low-rank approximation using PyTorch's implementation
    u_torch, s_torch, v_torch = torch.svd_lowrank(matrix, rank)
    v_torch = v_torch.mH

    ouput_dir = get_test_result_output_dir("low_rank_approximation_errors")
    pretty_print_dict = lambda u, s, v: json.dumps(calculate_svd_error(matrix, u, s, v), indent=4, sort_keys=True)
    with open(os.path.join(ouput_dir, f"matrix_{m}_{n}_{rank}.txt"), "w") as f:
        f.write(f"Matrix Size: {m}x{n}, Rank: {rank}")
        f.write(f"\n\nGold Standard SVD Error:\n")
        f.write(pretty_print_dict(u_gold, s_gold, v_gold))
        f.write(f"\n\nSOR_SVD Error:\n")
        f.write(pretty_print_dict(u_rand, s_rand, v_rand))
        f.write(f"\n\nRSVD Error:\n")
        f.write(pretty_print_dict(u_sor, s_sor, v_sor))
        f.write(f"\n\nPyTorch low-rank SVD Error:\n")
        f.write(pretty_print_dict(u_torch, s_torch, v_torch))


def test_low_rank_approximation_performance():
    m = 256 * 256
    n = 25 * 64 * 4
    rank = 10
    matrix = generate_low_rank_matrix(m, n, rank=rank)
    do_performance_test(subspace_orbit_randomized_svd, matrix, rank)
    do_performance_test(randomized_svd, matrix, rank)
    # Unfortunately, we cannot test torch.svd_lowrank this way, because we're not allowed to
    # compile it.
    do_performance_test(torch.svd_lowrank, matrix, rank, test_compilation=False)
