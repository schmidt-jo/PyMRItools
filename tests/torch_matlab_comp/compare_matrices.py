import torch


def check_matrices(matrix_1, matrix_2, rtol=1e-04, atol=1e-07, name="", assertion: bool =True):
    matrix_1 = torch.squeeze(matrix_1)
    matrix_2 = torch.squeeze(matrix_2)

    # always error shape mismatches
    assert matrix_1.shape == matrix_2.shape, f"{name} shape mismatch"
    # give some distance
    print(f"{name} matrix distance: {torch.linalg.norm(matrix_1 - matrix_2):.5f}")
    # on some tests we can toggle assertion error vs reporting
    if assertion:
        assert torch.allclose(matrix_1, matrix_2, rtol=rtol, atol=atol), f"{name} matrix mismatch"
    else:
        print(f"WARN: Matrix mismatch: {name}")