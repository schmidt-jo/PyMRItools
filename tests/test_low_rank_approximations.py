import pytest
import torch

from tests.utils import do_performance_test

from pymritools.utils.algorithms import subspace_orbit_randomized_svd_PS, subspace_orbit_randomized_svd

def generate_low_rank_matrix(m: int, n: int, rank: int, dtype=torch.float32, device="cpu") -> torch.Tensor:
    """Generates a low-rank matrix A = B @ C where B and C are low-rank components."""
    B = torch.randn((m, rank), dtype=dtype, device=device)
    C = torch.randn((rank, n), dtype=dtype, device=device)
    return torch.matmul(B, C)

def gold_standard_svd(matrix: torch.Tensor, rank: int) -> tuple[torch.Tensor, ...]:
    u, s, v = torch.linalg.svd(matrix, full_matrices=False)
    u_trunc = u[:, :rank].contiguous()
    s_trunc = s[:rank]
    v_trunc = v[:rank, :].contiguous()
    return u_trunc, s_trunc, v_trunc

@pytest.mark.parametrize("m, n, rank", [
    (4096, 512, 10),  # Test case 1: Tall matrix
    (4096, 4096, 10),  # Test case 2: Square matrix
    (8192, 64, 5),   # Test case 3: Tall skinny matrix
])
def test_low_rank_approximation(m: int, n: int, rank: int):
    """Test the low-rank approximation quality of the randomized SVD."""
    torch.manual_seed(42)
    matrix = generate_low_rank_matrix(m, n, rank=rank)

    # Compute the gold standard SVD solution
    u_gold, s_gold, v_gold = gold_standard_svd(matrix, rank)
    approx_gold = (u_gold * s_gold) @ v_gold

    # Compute the low-rank approximation using Jochen's SOR_SVD implementation
    aj_1, sj_rand, aj_2 = subspace_orbit_randomized_svd(matrix, rank)
    approxj_rand = (aj_1 * sj_rand) @ aj_2

    # Compute the low-rank approximation using our SOR_SVD implementation
    ap_1, sp_rand, ap_2 = subspace_orbit_randomized_svd_PS(matrix, rank)
    approxp_rand = (ap_1 * sp_rand) @ ap_2

    # Charlie says the Frobenius norm is commonly used to compare matrices because
    # it is a natural and mathematically sound way to measure the "size" of the difference between two matrices.
    error_gold = torch.norm(matrix - approx_gold, p="fro")
    errorj_rand = torch.norm(approx_gold - approxj_rand, p="fro")
    errorp_rand = torch.norm(approx_gold - approxp_rand, p="fro")

    print(f"\nMatrix Size: {m}x{n}, Rank: {rank}")
    print(f"Gold Standard Error: {error_gold.item():.6f}")
    print(f"Jochen's randomized SVD Error: {errorj_rand.item():.6f}")
    print(f"Patrick's randomized SVD Error: {errorp_rand.item():.6f}")


def test_sor_svd_performance():
    m = 2048
    n = 128
    rank = 5
    matrix = generate_low_rank_matrix(m, n, rank=rank)
    do_performance_test(subspace_orbit_randomized_svd_PS, matrix, rank)
