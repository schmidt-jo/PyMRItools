import os

import torch
from scipy.io import loadmat, savemat
from tests.utils import get_test_result_output_dir

from tests.torch_matlab_comp.matlab_utils import run_matlab_script
from tests.torch_matlab_comp.compare_matrices import check_matrices

from pymritools.recon.loraks_dev.matrix_indexing import get_linear_indices

import plotly.graph_objects as go
import plotly.subplots as psub


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
        indexing='ij'
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
    # Iterate through the k-space to find valid neighborhoods
    for y in range(k_space_shape[0]):
        for x in range(k_space_shape[1]):
            # Check if all neighborhood indices are valid
            valid_neighborhood = [
                is_valid_index(torch.tensor([y, x]), offset)
                for offset in offsets
            ]
            if all(valid_neighborhood):
                if reversed:
                    neighborhood_linear_indices = [
                        (
                            k_space_shape[0] - y + offset[0]
                        ) * k_space_shape[1] + (
                            k_space_shape[1] - x + offset[1]
                        )
                        for offset in offsets
                    ]
                else:
                    # Convert 2D indices to linear indices
                    neighborhood_linear_indices = [
                        (y + offset[0]) * k_space_shape[1] + (x + offset[1])
                        for offset in offsets
                    ]
                linear_indices.append(neighborhood_linear_indices)
    return torch.tensor(linear_indices).mT


def new_matlike_s_operator(k_space: torch.Tensor, indices: torch.Tensor, matrix_shape: tuple):
    match k_space.dtype:
        case torch.float32:
            dtype = torch.complex64
        case torch.float64:
            dtype = torch.complex128
        case _:
            dtype = torch.complex128
    k_flip = torch.flip(k_space, dims=(-2, -1))
    k_flip = k_flip.view(*k_flip.shape[:-2], -1)
    k_space = k_space.view(*k_space.shape[:-2], -1)
    # effectively c - matrix in each channel
    s_p = k_space[..., indices].view(-1, *matrix_shape)
    s_m = k_flip[..., indices].view(-1, *matrix_shape).flip(-2)

    s_p_m = (s_p - s_m).to(dtype)
    s_u = torch.concatenate([s_p_m.real, -s_p_m.imag], dim=1)

    s_p_m = (s_p + s_m).to(dtype)
    s_d = torch.concatenate([s_p_m.imag, s_p_m.real], dim=1)

    s = torch.concatenate([s_u, s_d], dim=-1).contiguous()
    return s.view(-1, s.shape[-1])


def new_matlike_s_operator_rev(k_space: torch.Tensor, indices: torch.Tensor, indices_rev: torch.Tensor, matrix_shape: tuple):
    match k_space.dtype:
        case torch.float32:
            dtype = torch.complex64
        case torch.float64:
            dtype = torch.complex128
        case _:
            dtype = torch.complex128
    k_space = k_space.view(*k_space.shape[:-2], -1)
    # effectively c - matrix in each channel
    s_p = k_space[..., indices].view(-1, *matrix_shape)
    s_m = k_space[..., indices_rev].view(-1, *matrix_shape)

    s_p_m = (s_p - s_m).to(dtype)
    s_u = torch.concatenate([s_p_m.real, -s_p_m.imag], dim=1)

    s_p_m = (s_p + s_m).to(dtype)
    s_d = torch.concatenate([s_p_m.imag, s_p_m.real], dim=1)

    s = torch.concatenate([s_u, s_d], dim=-1).contiguous()
    return s.view(-1, s.shape[-1])


def plot_matrices(matrices: list | torch.Tensor, name: str):
    if isinstance(matrices, torch.Tensor):
        matrices = [matrices]
    if matrices[0].shape.__len__() < 3:
        cols = 1
    elif matrices[0].shape.__len__() > 3:
        cols = matrices[0].shape[-3]
    else:
        cols = matrices[0].shape[0]

    fig = psub.make_subplots(rows=len(matrices), cols=cols)
    for i, d in enumerate(matrices):
        while d.shape.__len__() < 3:
            d = d.unsqueeze(0)
        while d.shape.__len__() > 3:
            d = d[0]
        for c, m in enumerate(d):
            fig.add_trace(
                go.Heatmap(z=torch.abs(m), showscale=False),
                row=i + 1, col=c + 1
            )
    fname = os.path.join(get_test_result_output_dir(test_nullspace_extraction), name)
    print(f"Saving {fname}.html")
    fig.write_html(f"{fname}.html")


def test_nullspace_extraction():
    print("\nK Space creation")
    torch.manual_seed(10)
    nx = 40
    ny = 30

    rank = 20
    r = 3

    # create k-space
    k_data = (
            torch.randn((1, ny, nx), dtype=torch.complex128) +
            torch.linspace(1, 20, ny*nx).reshape(1, ny, nx).to(dtype=torch.complex128)
    )

    print("\nMatlab subpprocessing")

    # 2. Create the output directory
    output_dir = get_test_result_output_dir("test_nullspace_extraction")

    # MATLAB format (.mat)
    matlab_input_path = os.path.join(output_dir, "input_data.mat")
    # Convert to numpy for saving to .mat format
    savemat(
        matlab_input_path,
        {
            'r': r, 'rank': rank,
            'k_data': k_data.permute(1, 2, 0).numpy()
        }
    )

    # 3. Call MATLAB to perform eigenvalue decomposition
    # matlab_output_path = perform_nullspace_extraction(matlab_input_path, output_dir)
    matlab_output_path = "/data/pt_np-jschmidt/code/PyMRItools/test_output/test_nullspace_extraction/matlab_loraks_nmm.mat"

    # 4. Load mat file
    mat = loadmat(matlab_output_path)

    print("\nK Space Input")
    mat_k_data = torch.from_numpy(mat["kData"]).to(dtype=k_data.dtype)
    plot_matrices([torch.squeeze(k_data), mat_k_data], name="01_k-space")
    check_matrices(k_data, mat_k_data, name="Input K Space")

    print("\nBuild Indices")
    # do python version
    # build indices
    indices = get_indices(k_space_shape=k_data.shape[-2:], nb_radius=r)
    indices_rev = get_indices(k_space_shape=k_data.shape[-2:], nb_radius=r, reversed=True)

    nb_size = indices.shape[0]
    indices_sq, matrix_shape_sq = get_linear_indices(
        k_space_shape=k_data.shape, patch_shape=(-1, 2 * r - 1, 2 * r - 1), sample_directions=(0, 1, 1)
    )

    print("\nBuild C Matrix")
    # build c matrix
    c_matrix = k_data.view(*k_data.shape[:-2], -1)[:, indices].view(indices.shape)
    c_matrix_sq = k_data.view(-1)[indices_sq].view(matrix_shape_sq).mT
    # get matlab matrix
    mat_c_matrix = torch.from_numpy(mat["c_matrix"]).to(dtype=c_matrix.dtype)
    plot_matrices([c_matrix, c_matrix_sq, mat_c_matrix], "02_c-matrix")
    check_matrices(c_matrix, mat_c_matrix, name="C-Matrix")

    # test mirroring
    c_matrix_flipped_indexing = k_data.view(*k_data.shape[:-2], -1)[:, indices_rev].view(indices.shape)
    c_matrix_flipped_k = torch.flip(k_data, dims=(-2, -1)).view(*k_data.shape[:-2], -1)[:, indices].view(indices.shape)
    plot_matrices([c_matrix_flipped_indexing, c_matrix_flipped_k], "02_c-matrix_mirrored_idx")

    print("\nBuild S-Matrix")
    # build s matrix
    s_matrix = new_matlike_s_operator(
        k_space=k_data, indices=indices, matrix_shape=indices.shape
    )
    s_matrix_rev = new_matlike_s_operator_rev(
        k_space=k_data, indices=indices, indices_rev=indices_rev, matrix_shape=indices.shape
    )
    # get matlab matrix
    mat_s_matrix = torch.from_numpy(mat["s_matrix"]).to(dtype=s_matrix.dtype)
    plot_matrices([s_matrix, s_matrix_rev, mat_s_matrix], "03_s-matrix")
    check_matrices(s_matrix_rev, mat_s_matrix, name="S-Matrices")

    print("\nEigenvalue decomposition")
    e_vals, e_vecs = torch.linalg.eigh(
        s_matrix_rev @ s_matrix_rev.mH,
        UPLO="U"        # some signs are reversed compared to matlab if using default L
    )
    idx = torch.argsort(torch.abs(e_vals), descending=True)

    um = e_vecs[:, idx]
    mat_um = torch.from_numpy(mat["U"]).to(dtype=um.dtype)
    plot_matrices([um, mat_um], "04-um")
    check_matrices(torch.abs(um), torch.abs(mat_um), name="abs Eigenvectors")
    check_matrices(um, mat_um, name="Eigenvectors")

    print("\nBuild Nullspace")
    nmm = um[:, rank:].mH
    mat_nmm = torch.from_numpy(mat["nmm"]).to(dtype=um.dtype)
    plot_matrices([nmm, mat_nmm], "05-nmm")
    check_matrices(nmm, mat_nmm, name="Nullspace")

    print("\nComplexify Nullspace")
    nfilt, filt_size = nmm.shape
    nss_c = torch.reshape(nmm, (nfilt, -1, nb_size))
    nss_c = nss_c[:, ::2] + 1j * nss_c[:, 1::2]
    nss_c = torch.reshape(nss_c, (nfilt, -1))
    mat_nss_c = torch.from_numpy(mat["nss_c"]).to(dtype=nss_c.dtype)
    plot_matrices([nss_c, mat_nss_c], "06-nss_c")
    check_matrices(nss_c, mat_nss_c, name="Complex Nullspace")


