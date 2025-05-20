import torch
from tests.torch_matlab_comp.compare_matrices import check_matrices


def test_matrix_order_eig():
    # 1. create a random complex matrix
    n1, n2 = (20, 10)
    matrix = torch.randn(n1, n2, dtype=torch.complex128)

    # 2. create a set of permuted column and row orders
    idx_r = torch.randperm(n1)
    idx_c = torch.randperm(n2)

    # 3. create column and row permuted matrices
    matrix_rp = matrix[idx_r]
    matrix_cp = matrix[:, idx_c]

    # 4. do eigenvalue decomposition
    eig_vals, eig_vecs = torch.linalg.eigh(matrix.mH @ matrix)
    eig_vals_rp, eig_vecs_rp = torch.linalg.eigh(matrix_rp.mH @ matrix_rp)
    eig_vals_cp, eig_vecs_cp = torch.linalg.eigh(matrix_cp.mH @ matrix_cp)

    # 5. test eigenvalues
    check_matrices(eig_vals, eig_vals_rp, name="Eigenvalues row permuted")
    check_matrices(eig_vals, eig_vals_cp, name="Eigenvalues col permuted")

    # 6. test eigenvectors
    check_matrices(torch.abs(eig_vecs), torch.abs(eig_vecs_rp), name="Eigenvectors row permuted")
    check_matrices(torch.abs(eig_vecs), torch.abs(eig_vecs_cp), name="Eigenvectors col permuted")
