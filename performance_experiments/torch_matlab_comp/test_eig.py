import os
import torch
import scipy.io as sio
from tests.utils import get_test_result_output_dir
from tests.torch_matlab_comp.matlab_utils import run_matlab_script
from tests.torch_matlab_comp.compare_matrices import check_matrices


def perform_eig_decomposition(input_matrix_path, output_dir):
    """
    Perform eigenvalue decomposition using MATLAB.

    Parameters:
        input_matrix_path (str): Path to the input .mat file containing the matrix
        output_dir (str): Directory where the output will be saved

    Returns:
        str: Path to the output .mat file containing eigenvalues and eigenvectors

    Raises:
        RuntimeError: If the MATLAB eigenvalue decomposition fails
    """
    # Construct arguments for the MATLAB script
    script_args = f"'{input_matrix_path}', '{output_dir}'"

    # Run the MATLAB script
    run_matlab_script("eig_decomposition", script_args)

    # Return the path to the output file
    return os.path.join(output_dir, "matlab_eig_result.mat")


def test_eig_comparison():
    """
    Test to compare eigenvalue decomposition between PyTorch and MATLAB.

    This test:
    1. Creates a random PyTorch complex 2D matrix
    2. Uses get_test_result_output_dir to create output directory
    3. Saves the matrix in MATLAB format
    4. Calls MATLAB to perform the same eigenvalue decomposition
    5. Performs eigenvalue decomposition with PyTorch
    6. Asserts that the comparison shows no difference
    """
    for d, dtype in enumerate([torch.float64, torch.complex128]):
        print(f"\nTesting eigenvalue decomposition for {dtype} matrix")
        # 1. Create a hermitian random PyTorch complex 2D matrix
        matrix_size = 10  # Size of the square matrix
        matrix = torch.randn((matrix_size, matrix_size), dtype=dtype)
        matrix = matrix @ matrix.mH

        # 2. Create the output directory
        output_dir = get_test_result_output_dir("test_eig_comparison")

        # MATLAB format (.mat)
        matlab_input_path = os.path.join(output_dir, "input_matrix.mat")
        # Convert to numpy for saving to .mat format
        matrix_np = matrix.numpy()
        sio.savemat(matlab_input_path, {'matrix': matrix_np})

        # 3. Call MATLAB to perform eigenvalue decomposition
        matlab_output_path = perform_eig_decomposition(matlab_input_path, output_dir)

        # 4. Load mat file
        mat = sio.loadmat(matlab_output_path)
        mat_eigenvalues = torch.from_numpy(mat['eigenvalues']).squeeze()
        mat_eigenvectors = torch.from_numpy(mat['eigenvectors']).squeeze()

        # 4. Perform eigenvalue decomposition with PyTorch
        # We need to flip the result to get the largest eigenvalues first
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        idx = torch.sort(torch.abs(eigenvalues), descending=True).indices
        eigenvalues = eigenvalues[..., idx]
        eigenvectors = eigenvectors[..., idx]

        # 5. Compare eigenvalues
        check_matrices(eigenvalues, mat_eigenvalues, name="Eigenvalues")

        # 6. Compare eigenvectors
        # eigenvectors are not unique, different software might compute different values up to a phase offset,
        # try comparing abs
        check_matrices(torch.abs(eigenvectors), torch.abs(mat_eigenvectors), name="Eigenvectors")
