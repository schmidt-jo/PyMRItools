import torch


def check_matrices(matrix_1, matrix_2, rtol=1e-04, atol=1e-07, name=""):
    assert matrix_1.shape == matrix_2.shape, f"{name} shape mismatch"
    # give some distance
    print(f"{name} matrix distance: {torch.linalg.norm(matrix_1 - matrix_2):.5f}")
    assert torch.allclose(matrix_1, matrix_2, rtol=rtol, atol=atol), f"{name} matrix mismatch"
