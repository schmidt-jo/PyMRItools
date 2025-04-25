import os

import torch
from scipy.io import loadmat, savemat
from tests.utils import get_test_result_output_dir

from tests.torch_matlab_comp.matlab_utils import run_matlab_script
from tests.torch_matlab_comp.compare_matrices import check_matrices

def perform_nullspace_extraction(input_matrix_path, output_dir):
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
    run_matlab_script("ac_loraks", script_args)

    # Return the path to the output file
    return os.path.join(output_dir, "matlab_loraks_nmm.mat")


def get_indices(k_space_shape: tuple, nb_radius: int, reversed: bool = False):
    # want a circular neighborhood radius and convert to linear indices
    nb_x, nb_y = torch.meshgrid(
        torch.arange(-nb_radius, nb_radius + 1),
        torch.arange(-nb_radius, nb_radius + 1),
        indexing='ij'  # Use 'ij' indexing to match the shape correctly
    )

    # Create a mask for the circular neighborhood
    nb_r = nb_x ** 2 + nb_y ** 2 <= nb_radius ** 2

    # Get the indices of the circular neighborhood
    neighborhood_indices = torch.nonzero(nb_r).squeeze()

    # Calculate offsets relative to the center
    offsets = neighborhood_indices - nb_radius

    # Function to check if an index is within the k-space shape
    def is_valid_index(center, offset):
        new_idx = center + offset
        return torch.all(
            (new_idx >= torch.tensor([1 - s % 2 for s in k_space_shape[:2]])) &
            (new_idx < torch.tensor(k_space_shape[:2]))
        )

    # Prepare to collect valid linear indices
    linear_indices = []

    # Determine the mirroring center based on shape dimensions
    if reversed:
        # For odd dimensions, use the exact center
        # For even dimensions, use the floor of the dimension divided by 2
        mirror_center_y = (k_space_shape[0] - 1) // 2
        mirror_center_x = (k_space_shape[1] - 1) // 2

    # Iterate through the k-space to find valid neighborhoods
    for y in range(k_space_shape[0]):
        for x in range(k_space_shape[1]):
            # Check if all neighborhood indices are valid
            valid_neighborhood = [
                is_valid_index(torch.tensor([y, x]), offset)
                for offset in offsets
            ]

            if all(valid_neighborhood):
                # Convert 2D indices to linear indices
                neighborhood_linear_indices = [
                    (y + offset[0]) * k_space_shape[1] + (x + offset[1])
                    for offset in offsets
                ]

                if reversed:
                    # Calculate mirrored neighborhood indices
                    mirrored_indices = [
                        (mirror_center_y * 2 - (y + offset[0])) * k_space_shape[1] +
                        (mirror_center_x * 2 - (x + offset[1]))
                        for offset in offsets
                    ]
                    linear_indices.append(mirrored_indices)
                else:
                    linear_indices.append(neighborhood_linear_indices)

    return torch.tensor(linear_indices)


def new_matlike_s_operator(k_space: torch.Tensor, indices: torch.Tensor, matrix_shape: tuple):
    match k_space.dtype:
        case torch.float32:
            dtype = torch.complex64
        case torch.float64:
            dtype = torch.complex128
        case _:
            dtype = torch.complex64
    k_flip = torch.flip(k_space, dims=(0, 1))
    s_p = k_space.view(-1, k_space.shape[-1])[indices].view(*matrix_shape, -1)
    s_m = k_flip.view(-1, k_space.shape[-1])[indices].view(*matrix_shape, -1)

    s_p_m = (s_p - s_m).to(dtype)
    s_u = torch.concatenate([s_p_m.real, -s_p_m.imag], dim=1)
    s_p_m = (s_p + s_m).to(dtype)
    s_d = torch.concatenate([s_p_m.imag, s_p_m.real], dim=1)

    s = torch.concatenate([s_u, s_d], dim=0)
    s = torch.concatenate([s_sub for s_sub in s.permute(2, 0, 1)], dim=-1)
    return s


def test_nullspace_extraction():
    # create k-space
    k_data = torch.randn((40, 40, 1), dtype=torch.complex128)

    # 2. Create the output directory
    output_dir = get_test_result_output_dir("test_ac_loraks_nullspace_extraction")

    # MATLAB format (.mat)
    matlab_input_path = os.path.join(output_dir, "input_data.mat")
    # Convert to numpy for saving to .mat format
    k_data = k_data.numpy()
    # savemat(matlab_input_path, {'k_data': k_data})

    # 3. Call MATLAB to perform eigenvalue decomposition
    # matlab_output_path = perform_nullspace_extraction(matlab_input_path, output_dir)

    # 4. Load mat file
    # mat = loadmat(matlab_output_path)

    # do python version
    # build indices
    indices = get_indices(k_space_shape=k_data.shape, nb_radius=3)
    nb_size = indices.shape[-1]

    # build s matrix
    s_matrix = k_data.flatten()[indices]

    # eigenvalue decomposition
    e_vals, e_vecs = torch.linalg.eigh(s_matrix.mH @ s_matrix)
    idx = torch.argsort(torch.abs(e_vals), descending=True)

    um = e_vecs[:, idx]
    nmm = um.mH

    nfilt, filt_size = nmm.shape
    nss_c = torch.reshape(nmm, (nfilt, nb_size, -1))
    nss_c = nss_c[:, ::2] + 1j * nss_c[:, 1::2]

    print("")




