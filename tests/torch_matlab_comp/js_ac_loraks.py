import os

import torch
from torch.nn.functional import pad
from scipy.io import loadmat, savemat
from tests.utils import get_test_result_output_dir

from tests.torch_matlab_comp.matlab_utils import run_matlab_script
from tests.torch_matlab_comp.compare_matrices import check_matrices

from pymritools.recon.loraks_dev.matrix_indexing import get_linear_indices
from pymritools.utils import Phantom

import plotly.graph_objects as go
import plotly.subplots as psub


def perform_matlab_computations(input_matrix_path, output_dir):
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
    return os.path.join(output_dir, "matlab_ac_loraks.mat")


def get_circular_nb_indices(nb_radius):
    # want a circular neighborhood, i.e. find all indices within a radius
    nb_x, nb_y = torch.meshgrid(
        torch.arange(-nb_radius, nb_radius + 1),
        torch.arange(-nb_radius, nb_radius + 1),
        indexing='ij'
    )
    # Create a mask for the circular neighborhood
    nb_r = nb_x ** 2 + nb_y ** 2 <= nb_radius ** 2
    # Get the indices of the circular neighborhood
    return torch.nonzero(nb_r).squeeze()


def get_indices(k_space_shape: tuple, nb_radius: int, reversed: bool = False):

    # want a circular neighborhood radius and convert to linear indices
    neighborhood_indices = get_circular_nb_indices(nb_radius=nb_radius)

    # Calculate offsets relative to the center
    offsets = neighborhood_indices - nb_radius

    y, x = torch.meshgrid(torch.arange(k_space_shape[-2]), torch.arange(k_space_shape[-1]))

    yx = torch.concatenate((y.unsqueeze(-1), x.unsqueeze(-1)), dim=-1)
    yx = torch.reshape(yx, (-1, 2))

    yxnb = yx.unsqueeze(0) + offsets.unsqueeze(1)
    idx = torch.all(
        (yxnb >= torch.tensor([1 - s % 2 for s in k_space_shape[:2]])) &
        (yxnb < torch.tensor(k_space_shape[:2])),
        dim=(0, 2)
    )

    if reversed:
        yxnb = torch.tensor(k_space_shape[:2]).unsqueeze(0).unsqueeze(1) - yx.unsqueeze(0) + offsets.unsqueeze(1)
    else:
        yxnb = yx.unsqueeze(0) + offsets.unsqueeze(1)

    yxnb = yxnb[:, idx]
    # convert to linear indices
    neighborhood_linear_indices = yxnb[..., 0] * k_space_shape[1] + yxnb[..., 1]
    return neighborhood_linear_indices


def get_indices_arxv(k_space_shape: tuple, nb_radius: int, reversed: bool = False):

    # want a circular neighborhood radius and convert to linear indices
    neighborhood_indices = get_circular_nb_indices(nb_radius=nb_radius)

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


def get_ac_matrix(k_data):
    # build s-matrix
    mask_in = (torch.abs(k_data) > 1e-11).to(torch.int)

    indices = get_indices(k_space_shape=k_data.shape[-2:], nb_radius=3, reversed=False)
    indices_rev = get_indices(k_space_shape=k_data.shape[-2:], nb_radius=3, reversed=True)

    nb_size = indices.shape[0] * k_data.shape[0]
    ac_matrix = new_matlike_s_operator_rev(k_data, indices=indices, indices_rev=indices_rev, matrix_shape=indices.shape)

    mask_p = torch.reshape(mask_in.view(*k_data.shape[:-2], -1)[:, indices], (-1, indices.shape[-1]))
    mask_f = torch.reshape(mask_in.view(*k_data.shape[:-2], -1)[:, indices_rev], (-1, indices.shape[-1]))

    idx = (torch.sum(mask_p, dim=0) == nb_size) & (torch.sum(mask_f, dim=0) == nb_size)
    idx = torch.concatenate([idx, idx], dim=0)
    ac_matrix[:, ~idx] = 0.0
    return ac_matrix


def plot_matrices(matrices: list | torch.Tensor, name: str, data_names: list | str = ""):
    if isinstance(matrices, torch.Tensor):
        matrices = [matrices]
    use_names = True
    if isinstance(data_names, str):
        if data_names:
            data_names = [data_names]
        else:
            use_names = False
    if matrices[0].shape.__len__() < 3:
        cols = 1
    elif matrices[0].shape.__len__() > 3:
        cols = matrices[0].shape[-3]
    else:
        cols = matrices[0].shape[0]
    fig = psub.make_subplots(
        rows=len(matrices),
        cols=cols,
        row_titles=data_names if use_names else [""]*len(matrices),
    )
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
    fig.update_layout(title=name)
    fname = os.path.join(get_test_result_output_dir(test_ac_loraks_vs_matlab), name)
    print(f"Saving {fname}.html")
    fig.write_html(f"{fname}.html")


def complex_subspace_representation(v: torch.Tensor, nb_size: int):
    nfilt, filt_size = v.shape
    nss_c = torch.reshape(v, (nfilt, -1, nb_size))
    nss_c = nss_c[:, ::2] + 1j * nss_c[:, 1::2]
    return torch.reshape(nss_c, (nfilt, -1))


def embed_circular_patch(v: torch.Tensor, nb_radius: int):
    nfilt, filt_size = v.shape

    # get indices
    circular_nb_indices = get_circular_nb_indices(nb_radius=nb_radius)
    # find neighborhood size
    nb_size = circular_nb_indices.shape[0]

    # build squared patch
    v = torch.reshape(v, (nfilt, -1, nb_size))
    nc = v.shape[1]

    v_patch = torch.zeros((nfilt, nc, 2 * nb_radius + 1, 2 * nb_radius + 1), dtype=v.dtype)
    v_patch[:, :, circular_nb_indices[:, 0], circular_nb_indices[:, 1]] = v

    return v_patch


def zero_phase_filter(v: torch.Tensor, nb_patch_side_length: int, matrix_type: str = "S"):
    # Conjugate of filters
    cfilt = torch.conj(v)

    # Determine ffilt based on opt
    if matrix_type == 'S':  # for S matrix
        ffilt = torch.conj(v)
    else:  # for C matrix
        ffilt = torch.flip(v, dims=(-2, -1))

    # Perform 2D FFT
    ccfilt = torch.fft.fft2(
        cfilt,
        dim=(-2, -1),
        s=(
            2 * nb_patch_side_length - 1,
            2 * nb_patch_side_length - 1
        )
    )
    fffilt = torch.fft.fft2(
        ffilt,
        dim=(-2, -1),
        s=(
            2 * nb_patch_side_length - 1,
            2 * nb_patch_side_length - 1
        )
    )

    # Compute patch via inverse FFT of element-wise multiplication and sum
    patch = torch.fft.ifft2(
        torch.sum(ccfilt.unsqueeze(2) * fffilt.unsqueeze(1), dim=0),
        dim=(-2, -1)
    )
    return patch

def v_pad(v_patch: torch.Tensor, nx : int, ny: int, nb_patch_side_length: int):
    # assumed dims of v_patch [px, py, nce, nce]
    pad_x = nx - nb_patch_side_length
    pad_y = ny - nb_patch_side_length
    return pad(
        v_patch,
        (
            0, pad_x,
            0, pad_y
        ),
        mode='constant', value=0.0
    )

def v_shift(v_pad: torch.Tensor, nx : int, ny: int, nb_patch_side_length: int, matrix_type: str = "S"):
    if matrix_type == 'S':
        return torch.roll(
            v_pad,
            dims=(-2, -1),
            shifts=(
                -2 * nb_patch_side_length + 2 - ny % 2,
                -2 * nb_patch_side_length + 2 - nx % 2
            )
        )
    else:
        return torch.roll(
            v_pad,
            dims=(-2, -1),
            shifts=(
                - nb_patch_side_length + 1,
                - nb_patch_side_length + 1
            )
        )

def test_ac_loraks_vs_matlab():
    print("\nK Space creation")
    torch.manual_seed(10)
    nx = 100
    ny = 80
    nc = 3
    ne = 2

    rank = 20
    r = 3
    # side length for squared patch
    nb_patch_side_length = 2 * r + 1

    # create k-space
    # k_data = (
    #         torch.randn((ne, nc, ny, nx), dtype=torch.complex128) +
    #         torch.linspace(1, 20, ne*nc*ny*nx).reshape(ne, nc, ny, nx).to(dtype=torch.complex128)
    # )
    # k_data = torch.reshape(k_data, (-1, ny, nx))
    sl_phantom = Phantom.get_shepp_logan(shape=(nx, ny), num_coils=3, num_echoes=2)
    k_data = sl_phantom.sub_sample_ac_random_lines(ac_lines=20, acceleration=3)
    k_data = k_data.permute(3, 2, 1, 0)
    k_data = k_data.reshape(-1, ny, nx)

    print("\nMatlab subpprocessing")

    # 2. Create the output directory
    output_dir = get_test_result_output_dir(test_ac_loraks_vs_matlab)

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
    name_mat_out = "matlab_ac_loraks.mat"   # need this name in the mat file
    output_path = os.path.join(output_dir, name_mat_out)
    # ensure file not already there -> otherwise we might load an old file
    if os.path.exists(output_path):
        os.remove(output_path)
    matlab_output_path = perform_matlab_computations(matlab_input_path, output_dir)
    # matlab_output_path = "/data/pt_np-jschmidt/code/PyMRItools/test_output/test_ac_loraks_vs_matlab/matlab_ac_loraks.mat"

    # 4. Load mat file
    mat = loadmat(matlab_output_path)

    print("\nK Space Input")
    mat_k_data = torch.from_numpy(mat["kData"]).to(dtype=k_data.dtype).permute(2, 0, 1)
    plot_matrices([torch.squeeze(k_data), mat_k_data], data_names=["torch", "matlab"], name="01_k-space")
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
    c_matrix = torch.reshape(k_data.view(*k_data.shape[:-2], -1)[:, indices], (-1, indices.shape[-1]))
    c_matrix_sq = k_data.view(-1)[indices_sq].view(matrix_shape_sq).mT
    # get matlab matrix
    mat_c_matrix = torch.from_numpy(mat["c_matrix"]).to(dtype=c_matrix.dtype)
    plot_matrices(
        [c_matrix, c_matrix_sq, mat_c_matrix],
        data_names=["torch", "torch squared patches", "matlab"],
        name="02_c-matrix"
    )
    check_matrices(c_matrix, mat_c_matrix, name="C-Matrix")

    # test mirroring
    c_matrix_flipped_indexing = torch.reshape(
        k_data.view(*k_data.shape[:-2], -1)[:, indices_rev],
        (-1, indices.shape[-1])
    )
    c_matrix_flipped_k = torch.reshape(
        torch.flip(k_data, dims=(-2, -1)).view(*k_data.shape[:-2], -1)[:, indices],
        (-1, indices.shape[-1])
    )
    plot_matrices(
        [c_matrix_flipped_indexing, c_matrix_flipped_k],
        data_names=["flipped indices", "flipped k -space"],
        name="02_c-matrix_mirrored_idx"
    )

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
    plot_matrices(
        [s_matrix, s_matrix_rev, mat_s_matrix],
        data_names=["torch flipped k", "torch reversed indexing", "matlab"],
        name="03_s-matrix"
    )
    check_matrices(s_matrix_rev, mat_s_matrix, name="S-Matrices")

    print("\nBuild AC Matrix")
    m_ac = get_ac_matrix(k_data=k_data)
    mat_m_ac = torch.from_numpy(mat["mac"]).to(dtype=m_ac.dtype)
    plot_matrices(
        [m_ac, mat_m_ac],
        data_names=["torch", "matlab"],
        name="04-mac"
    )
    check_matrices(m_ac, mat_m_ac, name="AC Matrix")

    print("\nEigenvalue decomposition")
    e_vals, e_vecs = torch.linalg.eigh(
        m_ac @ m_ac.mH,
        UPLO="U"        # some signs are reversed compared to matlab if using default L
    )
    u, s, _ = torch.linalg.svd(m_ac, full_matrices=False)
    se_vals = s**2 / m_ac.shape[0]

    idx = torch.argsort(torch.abs(e_vals), descending=True)
    um = e_vecs[:, idx]
    # um = e_vecs.flip(-1)

    mat_um = torch.from_numpy(mat["U"]).to(dtype=um.dtype)
    plot_matrices(
        [um, u, mat_um, mat_um - um],
        data_names=["torch", "torch svd", "matlab", "difference"],
        name="04-um"
    )
    check_matrices(torch.abs(um), torch.abs(mat_um), name="abs Eigenvectors", assertion=False)
    check_matrices(um, mat_um, name="Eigenvectors", assertion=False)
    check_matrices(u, mat_um, name="Eigenvectors svd", assertion=False)

    # ToDo: Upon similar inputs we get differences in the eigenvalue decomposition.
    #   Eigenvectors usually are normalized (check again). Can we get away with filtering n last eigenvectors,
    #   corresponding to the lowest eigenvalues as the "leading" vectors will define the space anyway.
    #   This way we might rid the offsets compared to the matlab version.
    # E.g. we could throw away all vectors corresponding to eigenvalues below a low value threshold
    # (in the e-15 and below range)
    # ToDo: Is it input value range dependent?
    # e_vals = e_vals[idx]
    # e_v_thresh = 1e-12 * torch.abs(torch.max(e_vals))
    # idx_thresh = torch.where(torch.abs(e_vals) < e_v_thresh)[0][0].item()

    print("\nBuild Nullspace")
    nmm = um[:, rank:].mH
    # nmm = um[:, rank:idx_thresh].mH
    mat_nmm = torch.from_numpy(mat["nmm"]).to(dtype=um.dtype)
    # mat_nmm = torch.from_numpy(mat["nmm"]).to(dtype=um.dtype)[:idx_thresh-rank]
    plot_matrices(
        [nmm, mat_nmm, mat_nmm - nmm],
        data_names=["torch", "matlab", "difference"],
        name="05-nmm"
    )
    check_matrices(nmm, mat_nmm, name="Nullspace", assertion=False)

    print("\nComplexify Nullspace")
    nss_c = complex_subspace_representation(nmm, nb_size=nb_size)
    mat_nss_c = torch.from_numpy(mat["nss_c"]).to(dtype=nss_c.dtype)
    plot_matrices(
        [nss_c, mat_nss_c, mat_nss_c - nss_c],
        data_names=["torch", "matlab", "difference"],
        name="06-nss_c"
    )
    check_matrices(nss_c, mat_nss_c, name="Complex Nullspace", assertion=False)

    print("\nPrep 0 phase input")
    v_patch = embed_circular_patch(nss_c, nb_radius=r)
    mat_v_patch = torch.from_numpy(mat["v_patch"]).to(dtype=v_patch.dtype).permute(3, 2, 0, 1)
    plot_matrices(
        [v_patch, mat_v_patch],
        data_names=["torch", "matlab"],
        name="07-filtfilt"
    )
    plot_matrices(
        [v_patch[:10, 0], mat_v_patch[:10, 0]],
        data_names=["torch", "matlab"],
        name="07-filtfilt_b"
    )
    check_matrices(v_patch, mat_v_patch, name="Filtfilt")

    print("\nPrep 0 phase filter")
    vs_patch = zero_phase_filter(v_patch.clone(), nb_patch_side_length=nb_patch_side_length, matrix_type="S")
    mat_vs_patch = torch.from_numpy(mat["vs_patch"]).to(dtype=vs_patch.dtype).permute(3, 2, 0, 1)
    plot_matrices(
        [vs_patch, mat_vs_patch],
        data_names=["torch", "matlab"],
        name="08-vs patch"
    )
    check_matrices(vs_patch, mat_vs_patch, name="VS Patch")

    vc_patch = zero_phase_filter(v_patch.clone(), nb_patch_side_length=nb_patch_side_length, matrix_type="C")
    mat_vc_patch = torch.from_numpy(mat["vc_patch"]).to(dtype=vc_patch.dtype).permute(3, 2, 0, 1)
    plot_matrices(
        [vc_patch, mat_vc_patch],
        data_names=["torch", "matlab"],
        name=f"08-vc patch"
    )
    check_matrices(vc_patch, mat_vc_patch, name="VC Patch")

    print(f"\nPrep filter convolutions")
    vs_pad_shift = v_shift(
        v_pad(
            vs_patch, nx=nx, ny=ny, nb_patch_side_length=nb_patch_side_length
        ),
        nx=nx, ny=ny, nb_patch_side_length=nb_patch_side_length, matrix_type="S"
    )
    vs = torch.fft.fft2(vs_pad_shift, dim=(-2, -1))
    mat_vs = torch.from_numpy(mat["vs"]).to(dtype=vs.dtype).permute(3, 2, 0, 1)
    plot_matrices(
        [vs, mat_vs],
        data_names=["torch", "matlab"],
        name="09-vs"
    )
    check_matrices(vs, mat_vs, name="VS")

    vc_pad_shift = v_shift(
        v_pad(
            vc_patch, nx=nx, ny=ny, nb_patch_side_length=nb_patch_side_length
        ),
        nx=nx, ny=ny, nb_patch_side_length=nb_patch_side_length, matrix_type="C"
    )
    vc = torch.fft.fft2(vc_pad_shift)
    mat_vc = torch.from_numpy(mat["vc"]).to(dtype=vc.dtype).permute(3, 2, 0, 1)
    plot_matrices(
        [vc, mat_vc],
        data_names=["torch", "matlab"],
        name="09-vc"
    )
    check_matrices(vc, mat_vc, name="VC")

    print("\nPrep k-space")
    pad_k = pad(k_data, (0, nb_patch_side_length - 1, 0, nb_patch_side_length - 1), mode="constant", value=0.0)
    fft_k = torch.fft.fft2(pad_k, dim=(-2, -1))
    mat_fft_k = torch.from_numpy(mat["fft_k"]).permute(2, 0, 1).to(dtype=fft_k.dtype)
    plot_matrices(
        [fft_k, mat_fft_k],
        data_names=["torch", "matlab"],
        name="10-fft-k"
    )
    check_matrices(fft_k, mat_fft_k, name="FFT K")

    print("\nCompute convolutions")
    # dims [nx + nb - 1, ny + nb - 1, nce]
    mv_c = torch.sum(vc * fft_k.unsqueeze(0), dim=1)
    mat_mv_c = torch.squeeze(torch.from_numpy(mat["vc_k"])).to(dtype=mv_c.dtype).permute(2, 0, 1)
    plot_matrices(
        [mv_c, mat_mv_c],
        data_names=["torch", "matlab"],
        name="11-mvc-k"
    )

    mv_s = torch.sum(vs * torch.conj(fft_k).unsqueeze(0), dim=1)
    mat_mv_s = torch.squeeze(torch.from_numpy(mat["vs_k"])).to(dtype=mv_c.dtype).permute(2, 0, 1)
    plot_matrices(
        [mv_s, mat_mv_s],
        data_names=["torch", "matlab"],
        name="11-mvs-k"
    )

    imv_c = torch.fft.ifft2(mv_c, dim=(-2, -1))[..., :ny, :nx]
    mat_imv_c = torch.squeeze(torch.from_numpy(mat["i_vc_k"])).to(dtype=imv_c.dtype).permute(2, 0, 1)
    plot_matrices(
        [imv_c, mat_imv_c],
        data_names=["torch", "matlab"],
        name="12-imvc-k"
    )
    check_matrices(imv_c, mat_imv_c, name="IMVC K")

    imv_s = torch.fft.ifft2(mv_s, dim=(-2, -1))[..., :ny, :nx]
    mat_imv_s = torch.squeeze(torch.from_numpy(mat["i_vs_k"])).to(dtype=imv_c.dtype).permute(2, 0, 1)
    plot_matrices(
        [imv_s, mat_imv_s],
        data_names=["torch", "matlab"],
        name="12-imvs-k"
    )
    check_matrices(imv_s, mat_imv_s, name="IMVS K")

    print("\nResults")
    m = 2 * (imv_c - imv_s)
    mat_m = torch.squeeze(torch.from_numpy(mat["m"])).to(dtype=m.dtype).permute(2, 0, 1)
    plot_matrices(
        [m, mat_m],
        data_names=["torch", "matlab"],
        name="13-m"
    )
    check_matrices(m, mat_m, name="M")

    mat_z = torch.from_numpy(mat["z"]).to(dtype=k_data.dtype).permute(2, 0, 1)
    mat_img = torch.fft.fftshift(
        torch.fft.ifft2(
            torch.fft.ifftshift(mat_z, dim=(-2, -1)),
            dim=(-2, -1)
        ),
        dim=(-2, -1)
    )
    plot_matrices(
        [mat_img],
        data_names=["mat reco img"],
        name="14-img"
    )


