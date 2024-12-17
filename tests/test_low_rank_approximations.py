import pytest
import torch
import json

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

def svd_error(original_matrix: torch.Tensor, u: torch.Tensor, sigma: torch.Tensor, vh: torch.Tensor) -> dict:
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
    (8192, 64, 5),   # Test case 3: Tall skinny matrix
])
def test_low_rank_approximation(m: int, n: int, rank: int):
    """Test the low-rank approximation quality of the randomized SVD."""
    torch.manual_seed(42)
    matrix = generate_low_rank_matrix(m, n, rank=rank)

    # Compute the gold standard SVD solution
    u_gold, s_gold, v_gold = gold_standard_svd(matrix, rank)

    # Compute the low-rank approximation using Jochen's SOR_SVD implementation
    aj_1, sj_rand, aj_2 = subspace_orbit_randomized_svd(matrix, rank)

    # Compute the low-rank approximation using our SOR_SVD implementation
    ap_1, sp_rand, ap_2 = subspace_orbit_randomized_svd_PS(matrix, rank)

    def pretty_print_dict(arg):
        print(json.dumps(arg, indent=4, sort_keys=True))

    print(f"\nMatrix Size: {m}x{n}, Rank: {rank}")
    print(f"Gold Standard SVD Error:")
    pretty_print_dict(svd_error(matrix, u_gold, s_gold, v_gold))
    print(f"Jochen's SOR_SVD Error:")
    pretty_print_dict(svd_error(matrix, aj_1, sj_rand, aj_2))
    print(f"Patrick's SOR_SVD Error:")
    pretty_print_dict(svd_error(matrix, ap_1, sp_rand, ap_2))



def test_sor_svd_performance():
    m = 2048
    n = 128
    rank = 5
    matrix = generate_low_rank_matrix(m, n, rank=rank)
    do_performance_test(subspace_orbit_randomized_svd_PS, matrix, rank)
