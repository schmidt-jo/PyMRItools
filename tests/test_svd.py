import torch
import torch.linalg as LA

from pymritools.utils.algorithms import randomized_svd, subspace_orbit_randomized_svd_PS


def generate_noisy_low_rank_matrix(n: int, k: int) -> torch.Tensor:
    """
    Generate a matrix with controlled singular values and added noise.

    Args:
        n (int): Size of the square matrix
        k (int): Number of non-zero singular values

    Returns:
        torch.Tensor: Generated matrix A
    """
    smax = 1.0
    smin = 1e-9
    s = torch.linspace(smax, smin, n)
    s[k:] = 0  # Set values after k to zero (note: Python uses 0-based indexing)
    G = torch.randn(n, n)
    E = G / LA.norm(G)
    Q1, _ = LA.qr(torch.randn(n, n))
    Q2, _ = LA.qr(torch.randn(n, n))
    return (Q1 * s) @ Q2.T + 0.1 * s[k - 1] * E

def test_svd_variants():
    n = 1000
    k = 20
    l = 38

    m = generate_noisy_low_rank_matrix(n, k)

    _, s_svd, _ = torch.linalg.svd(m, full_matrices=False)
    _, s_sor, _ = subspace_orbit_randomized_svd_PS(m, k, l - k)
    _, s_rand, _ = randomized_svd(m, k, l - k)