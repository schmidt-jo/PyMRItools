import logging
import pathlib as plib

import torch
from torch.nn.functional import pad, conv2d
import numpy as np
import tqdm
import plotly.graph_objects as go
import plotly.subplots as psub

from pymritools.recon.loraks_dev.matrix_indexing import get_linear_indices
from pymritools.recon.loraks_dev.operators import c_operator, s_operator
from pymritools.utils import Phantom, fft, root_sum_of_squares, torch_load, nifti_save

log_module = logging.getLogger(__name__)


def fourier_mv(k, v, ps):
    """
    Compute mv directly in Fourier space with additional conjugate filters and extra padding.
    Pads all inputs into the Fourier transform by patch size (ps) + extra padding, sums conjugate terms,
    and finally crops to avoid boundary issues.

    Parameters:
        k (torch.Tensor): Input tensor with shape [nx, ny, nc] (real or complex-valued).
        v (torch.Tensor): Compression matrix with shape [nc * ps**2, l] (real or complex-valued).
        ps (int): Patch size (side length).
        extra_pad_factor (int): Factor by which to increase padding beyond the patch size.

    Returns:
        torch.Tensor: Result tensor of shape [(nx - ps + 1) * (ny - ps + 1), l].
    """
    # Extract dimensions
    nx, ny, nc = k.shape  # Input tensor dimensions
    ps2, l = v.shape  # Compression matrix dimensions
    assert ps2 == nc * ps ** 2, f"Compression matrix v must have {nc * ps ** 2} rows, but got {ps2}."

    # Make inputs complex if not already
    if not torch.is_complex(k):
        k = k.to(dtype=torch.complex64)
    if not torch.is_complex(v):
        v = v.to(dtype=torch.complex64)

    # pad inputs
    k_padded = torch.zeros((nx + 2 * ps, ny + 2 * ps, nc), dtype=k.dtype, device=k.device)
    k_padded[ps:-ps, ps:-ps, :] = k

    # Reshape and pad compression matrix `v`
    v_reshaped = v.view(ps, ps, nc, l)  # Reshape `v` into blocks [nc, ps, ps, l]
    v_padded = torch.zeros((*k_padded.shape, l), dtype=v.dtype, device=v.device)
    pad_l = (k_padded.shape[0] - v_reshaped.shape[0]) // 2
    pad_d = (k_padded.shape[1] - v_reshaped.shape[1]) // 2
    v_padded[pad_l:pad_l + ps, pad_d:pad_d + ps, :, :] = v_reshaped

    # Perform Fourier transform on padded `k` and `v`
    k_fft = torch.conj(fft(k_padded, img_to_k=False, axes=(0, 1)))  # FFT of k: [nx+2*ps, ny+2*ps, nc]
    v_fft = fft(torch.conj(v_padded), img_to_k=False, axes=(0, 1))  # FFT of v: [nc, nx+2*ps, ny+2*ps, l]

    # Combine original and conjugated filters
    # Summation in Fourier space before inverse transformation, summation of coils
    result_fft = (k_fft[..., None] * v_fft).sum(dim=-2)
    # Inverse Fourier transform back to spatial domain
    # ToDo check (Parsevals theorem) if we could take the norm before FFT back (should be equivalent) and save one FFT step
    result_spatial = fft(result_fft, img_to_k=True, axes=(0, 1))  # Shape: [nx+2*ps, ny+2*ps, l]

    # Extract valid convolution region by cropping out the padded sections
    edge = int(ps * 1.5)
    result_cropped = result_spatial[edge:-edge, edge:-edge, :]  # Shape: [nx, ny, l]

    result_final = result_cropped.reshape((-1, l))
    return result_final

def fourier_mv_opt(k, v_fft, ps):
    # Extract dimensions
    # nx, ny, nc = k.shape  # Input tensor dimensions
    # # pad input
    # k_padded = torch.zeros((nx + 2 * ps, ny + 2 * ps, nc), dtype=k.dtype, device=k.device)
    # k_padded[ps:-ps, ps:-ps, :] = k
    # # Perform Fourier transform on padded `k`
    # k_fft = torch.conj(fft(k_padded, img_to_k=False, axes=(0, 1)))  # FFT of k: [nx+2*ps, ny+2*ps, nc]
    # # Combine original and conjugated filters
    # # Summation in Fourier space before inverse transformation, summation of coils
    # result_fft = torch.zeros_like(v_fft[..., 0, :], device=k.device)
    # for idx_c in range(result_fft.shape[-1]):
    #     result_fft += k_fft[..., idx_c, None] * v_fft[..., idx_c, :].to(k_fft.device)
    # # result_fft = (k_fft[..., None] * v_fft).sum(dim=-2)
    # result_spatial = fft(result_fft, img_to_k=True, axes=(0, 1))  # Shape: [nx+2*ps, ny+2*ps, l]
    #
    # # Extract valid convolution region by cropping out the padded sections
    # edge = int(ps * 1.5)
    # result_cropped = result_spatial[edge:-edge, edge:-edge, :]  # Shape: [nx, ny, l]
    # result_final = result_cropped.reshape((-1, v_fft.shape[-1]))

    result_final = conv2d(torch.movedim(k, -1, 0)[None], v_fft, padding="same")
    result_final = torch.squeeze(torch.movedim(result_final, 1, -1))
    result_final = result_final.reshape((-1, v_fft.shape[0]))
    return result_final


def fourier_mv_s_opt(k, v_fft, ps):
    mv_c = fourier_mv_opt(k=k, v_fft=v_fft, ps=ps)
    mv_s = torch.conj(fourier_mv_opt(k=torch.flip(k, dims=(0, 1)), v_fft=v_fft, ps=ps))
    return 2 * (mv_c - mv_s)

def fourier_mv_s(k, v, ps):
    """
    Compute mv directly in Fourier space with additional conjugate filters and extra padding.
    Pads all inputs into the Fourier transform by patch size (ps) + extra padding, sums conjugate terms,
    and finally crops to avoid boundary issues. For S matrix

    Parameters:
        k (torch.Tensor): Input tensor with shape [nx, ny, nc] (real or complex-valued).
        v (torch.Tensor): Compression matrix with shape [nc * ps**2, l] (real or complex-valued).
        ps (int): Patch size (side length).

    Returns:
        torch.Tensor: Result tensor of shape [(nx - ps + 1) * (ny - ps + 1), l].
    """
    v = v[::2] + 1j * v[1::2]

    mv_c = fourier_mv(k=k, v=v, ps=ps)
    mv_s = torch.conj(fourier_mv(k=torch.flip(k, dims=(0, 1)), v=v, ps=ps))

    return 2 * (mv_c - mv_s)


def find_ac_region_rxv(k_space: torch.Tensor, mask: torch.Tensor):
    nx, ny = mask.shape[:2]
    logging.info("Find AC Region")
    cx = int(nx / 2)
    cy = int(ny / 2)
    nce = mask.view((nx, ny, -1)).shape[-1]
    for ix in tqdm.trange(cx - 1, desc="AC Region search - x"):
        if torch.sum(mask.view((nx, ny, nce))[cx - ix - 1:cx + ix, cy]) < nce * (2 * ix + 1):
            break
    for iy in tqdm.trange(cy - 1, desc="AC Region search - y"):
        if torch.sum(mask.view((nx, ny, nce))[cx, cy - iy - 1:cy + iy]) < nce * (2 * iy + 1):
            break
    ac_data = k_space[cx - ix:cx + ix, cy - iy:cy + iy]
    return ac_data.contiguous(), nce


def find_ac_region(tensor):
    """
    Find the central rectangular region by expanding from the center,
    ensuring full sampling and consistency across dimensions.

    Args:
        tensor (torch.Tensor): Input tensor of 3 or more dimensions

    Returns:
        tuple: A slice tuple representing the AC region
    """
    # Ensure tensor has at least 3 dimensions
    if tensor.ndim < 3:
        raise ValueError("Input tensor must have at least 3 dimensions")

    # Create a mask of non-zero elements
    mask = torch.abs(tensor) > 1e-10

    # Reduced mask across all but first two dimensions to check consistency
    reduced_mask_2d = mask.all(dim=tuple(range(2, tensor.ndim)))

    # Function to find central fully-sampled region in a dimension
    def find_central_region(dim_mask):
        # Find indices of non-zero elements
        non_zero_indices = torch.where(dim_mask)[0]

        if len(non_zero_indices) == 0:
            raise ValueError("No non-zero elements in dimension")

        # Compute dimension size and center
        dim_size = len(dim_mask)
        center = dim_size // 2

        # Initialize region from center
        left = right = center

        # Expand region outwards, ensuring consistent sampling
        while True:
            # Check if we can expand left
            can_expand_left = left > 0 and dim_mask[left - 1]
            # Check if we can expand right
            can_expand_right = right < dim_size - 1 and dim_mask[right + 1]

            # If can't expand either direction, we're done
            if not (can_expand_left or can_expand_right):
                break

            # Prioritize symmetric expansion
            if can_expand_left and (not can_expand_right or
                                    (center - (left - 1)) <= (right + 1 - center)):
                left -= 1
            elif can_expand_right:
                right += 1

        return left, right + 1

    # Find central region for first two dimensions
    ac_slices = []
    for dim in range(2):
        # Reduce to current dimension
        dim_mask = reduced_mask_2d.any(dim=1 - dim)
        start, end = find_central_region(dim_mask)
        ac_slices.append(slice(start, end))

    # Add full slices for additional dimensions
    ac_slices.extend([slice(None) for _ in range(2, tensor.ndim)])

    # Create and verify the region
    ac_region_slices = tuple(ac_slices)
    ac_region = tensor[ac_region_slices]

    # Ensure fully non-zero
    if (ac_region == 0).any():
        raise ValueError("Found region contains zero values")

    return ac_region


def recon(
        k_space: torch.Tensor, sampling_mask: torch.Tensor,
        rank: int, loraks_neighborhood_size: int = 5, lambda_reg: float = 0.1, matrix_type: str = "S",
        max_num_iter: int = 200,
        device: torch.device = torch.get_default_device()
):
    if matrix_type.capitalize() == "S":
        use_s_matrix = True
    elif matrix_type.capitalize() == "C":
        use_s_matrix = False
    else:
        err = "Currently only S or C matrix LORAKS types are supported."
        log_module.error(err)
        raise ValueError(err)

    ac_data, nce = find_ac_region(k_space, sampling_mask)
    mask = sampling_mask.to(torch.bool)
    k_data_consistency = k_space[mask]

    log_module.info("Set Matrix Indices and AC Matrix")
    # ToDo: Calculate size of AC subregion if whole region is to big for CPU?

    sample_dir = torch.zeros(k_space.shape.__len__(), dtype=torch.int32)
    sample_dir[:2] = 1
    patch_shape = [loraks_neighborhood_size if s == 1 else -1 for s in sample_dir]
    ac_indices, ac_matrix_shape = get_linear_indices(
        k_space_shape=ac_data.shape,
        patch_shape=tuple(patch_shape),
        sample_directions=tuple(sample_dir),
    )

    if use_s_matrix:
        ac_matrix_shape = tuple((torch.tensor(ac_matrix_shape, dtype=torch.int) * torch.tensor([2, 2])).tolist())
        ac_matrix = s_operator(k_space=ac_data, indices=ac_indices, matrix_shape=ac_matrix_shape).to(device)
    else:
        ac_matrix = c_operator(k_space=ac_data, indices=ac_indices, matrix_shape=ac_matrix_shape).to(device)

    log_module.info("Calculate sub-matrix Eigenvalues")
    eig_vals, vecs = torch.linalg.eigh(ac_matrix.mH @ ac_matrix)
    # eigenvals and corresponding evs are in ascending order
    v = vecs[..., :-rank]

    del ac_matrix
    torch.cuda.empty_cache()

    log_module.info("Init optimization")
    k_init = torch.randn_like(k_space) * 1e-5
    k_init[mask] = k_data_consistency
    k = k_init.clone().to(device).requires_grad_()
    k_data_consistency = k_data_consistency.to(device)
    lr = torch.linspace(1e-3, 1e-4, max_num_iter, device=device)
    mask = mask.to(device)

    progress_bar = tqdm.trange(max_num_iter, desc="Optimization")
    for i in progress_bar:
        loss_1 = torch.linalg.norm(
            k[mask] - k_data_consistency
        )
        if use_s_matrix:
            mv = fourier_mv_s(k=k.view(*k.shape[:2], -1), v=v, ps=loraks_neighborhood_size)
        else:
            mv = fourier_mv(k=k.view(*k.shape[:2], -1), v=v, ps=loraks_neighborhood_size)

        loss_2 = torch.linalg.norm(mv, ord="fro")

        loss = loss_1 + lambda_reg * loss_2
        loss.backward()

        with torch.no_grad():
            k -= lr[i] * k.grad
        k.grad.zero_()

        progress_bar.postfix = (
            f"loss low rank: {1e3 * loss_2.item():.2f} -- loss data: {1e3 * loss_1.item():.2f} -- "
            f"total_loss: {1e3 * loss.item():.2f}"
        )
    k_recon = k.detach().cpu()
    return k_recon


def recon_batch(
        k_space: torch.Tensor, sampling_mask: torch.Tensor,
        ac_data: torch.Tensor, rank: int, use_s_matrix: bool = True,
        loraks_neighborhood_size: int = 5,
        conv_tol: float = 1e-6, max_num_iter: int = 200,
        device: torch.device = torch.get_default_device(),
        lr_low: float = 1e-2, lr_high: float = 1,
        reduced_num_nullspace_vec: int = 3000, sampled_reduced_nullspace_vec: int = 2000):

    mask = sampling_mask.to(torch.bool)
    mask_unsampled = ~mask
    ac_data = ac_data.contiguous()

    log_module.info("Set Matrix Indices and AC Matrix")
    # ToDo: Calculate size of AC subregion if whole region is to big for CPU?
    sample_dir = torch.zeros(len(k_space.shape), dtype=torch.int)
    sample_dir[:2] = 1
    patch_shape = torch.full(size=(len(k_space.shape),), fill_value=-1, dtype=torch.int)
    patch_shape[:2] = loraks_neighborhood_size

    ac_indices, ac_matrix_shape = get_linear_indices(
        k_space_shape=ac_data.shape,
        patch_shape=tuple(patch_shape.tolist()),
        sample_directions=tuple(sample_dir.tolist())
    )

    if use_s_matrix:
        ac_matrix_shape = tuple((torch.tensor(ac_matrix_shape, dtype=torch.int) * torch.tensor([2, 2])).tolist())
        ac_matrix = s_operator(k_space=ac_data, indices=ac_indices, matrix_shape=ac_matrix_shape).to(device)
    else:
        ac_matrix = c_operator(k_space=ac_data, indices=ac_indices, matrix_shape=ac_matrix_shape).to(device)

    eig_vals, vecs = torch.linalg.eigh(ac_matrix.mH @ ac_matrix)
    # eigenvals and corresponding evs are in ascending order

    start = max(0, vecs.shape[-1] - reduced_num_nullspace_vec - rank)
    indices = torch.arange(start, vecs.shape[-1] - rank)
    if indices.shape[0] < vecs.shape[-1]:
        num_reduced_samples = max(
            min(
                vecs.shape[-1] - indices.shape[0] - rank,
                sampled_reduced_nullspace_vec
            ),
            0
        )
        indices = torch.sort(
            torch.concatenate([indices, torch.randperm(start)[:num_reduced_samples]], dim=0)
        ).values

    # leading nullspace vecs
    v = vecs[..., indices]
    log_module.debug(f"Taking {v.shape[-1]} / {vecs.shape[-1] - rank} nullspace eigenvectors")
    del ac_matrix
    torch.cuda.empty_cache()
    if use_s_matrix:
        v = v[::2] + 1j * v[1::2]

    # Reshape and pad compression matrix `v`
    nce = torch.prod(torch.tensor(k_space.shape[2:])).item()
    v_reshaped = v.view(loraks_neighborhood_size, loraks_neighborhood_size, nce, v.shape[-1])
    # Reshape `v` into blocks [ps, ps, nc, l]
    # prep for conv2d
    v_reshaped = torch.movedim(v_reshaped, 2, 0)
    v_fft = torch.movedim(v_reshaped, -1, 0)
    # dims [l, nc, ps, ps]

    del vecs, eig_vals, v_reshaped
    torch.cuda.empty_cache()

    logging.info("Init optimization")
    # we optimize only unmeasured data
    # find read direction
    read_dir = -1
    for i in range(2):
        if ac_data.shape[i] >= k_space.shape[i] - 5:
            # tmp_mask = torch.movedim(mask[:, :, 0].to(torch.int), i, 0)
            # if torch.sum(tmp_mask[:, tmp_mask.shape[2] // 2]) == tmp_mask.shape[0]:
            read_dir = i
    if read_dir < 0:
        err = "Could not find read direction."
        log_module.error(err)
        raise ValueError(err)

    shape = torch.tensor(k_space.shape)
    if read_dir == 0:
        shape[1] = torch.count_nonzero(mask_unsampled) / (shape[0] * torch.prod(shape[2:]))
    else:
        shape[0] = torch.count_nonzero(mask_unsampled) / torch.prod(shape[1:])
    k = torch.zeros(
        tuple(shape.tolist()),
        dtype=k_space.dtype, device=device, requires_grad=True
    )

    lr = np.linspace(lr_high, lr_low, max_num_iter)
    # ToDo: Adaptive learning rate -> up lr when loss function change is low in the beginning,
    #  down lr when loss is oscillating
    mask_unsampled = mask_unsampled.to(device)

    # we take the 0 filled sampled data to the device
    k_sampled = k_space.to(device)

    progress_bar = tqdm.trange(max_num_iter, desc="Optimization")
    losses = []

    for i in progress_bar:
        # we need a zero filling operator to compute the losses
        tmp = torch.zeros_like(k_sampled)
        tmp[mask_unsampled] = k.flatten()
        if use_s_matrix:
            mv = fourier_mv_s_opt(k=torch.reshape(k_sampled + tmp, (*tmp.shape[:2], -1)), v_fft=v_fft,
                                  ps=loraks_neighborhood_size)
        else:
            mv = fourier_mv(k=(k_sampled + tmp).view(*tmp.shape[:2], -1), v=v, ps=loraks_neighborhood_size)

        # loss = loss_factor * torch.linalg.norm(mv, ord="fro") / torch.numel(mv)
        loss = torch.linalg.norm(mv, ord="fro")

        loss.backward()

        with torch.no_grad():
            k_last = k.clone().detach()
            k -= lr[i] * k.grad
            residual = torch.linalg.norm(k - k_last)

        norm_grad = torch.linalg.norm(k.grad).item()
        progress_bar.postfix = (
            f"loss: {loss.item():.2f}, norm k: {torch.linalg.norm(k).item():.3f}, "
            f"norm grad: {norm_grad:.3f}, norm residual: {residual:.6f}"
        )
        losses.append(loss.item())

        k.grad.zero_()
        if residual < conv_tol:
            msg = f"Optimization converged at step: {i + 1}"
            log_module.info(msg)
            break
    # build data
    tmp = torch.zeros_like(k_sampled)
    tmp[mask_unsampled] = k.flatten().detach()
    log_module.info(
        f"norm k sampled: {torch.linalg.norm(k_sampled).item():.3f}, "
        f"norm k reconned: {torch.linalg.norm(tmp).item():.3f}, "
        f"norm residual: {torch.linalg.norm(residual).item():.6f}"
    )
    return k_sampled + tmp, losses


def recon_data_consistency(
        k_space: torch.Tensor, sampling_mask: torch.Tensor,
        rank: int, loraks_neighborhood_size: int = 5, matrix_type: str = "S",
        conv_tol: float = 1e-6, max_num_iter: int = 200, batch_channels: int = -1,
        device: torch.device = torch.get_default_device(),
        lr_low: float = 1e-2, lr_high: float = 1,
        reduced_num_nullspace_vec: int = 3000, sampled_reduced_nullspace_vec: int = 2000):
    if matrix_type.capitalize() == "S":
        use_s_matrix = True
    elif matrix_type.capitalize() == "C":
        use_s_matrix = False
    else:
        err = "Currently only S or C matrix LORAKS types are supported."
        log_module.error(err)
        raise ValueError(err)
    k_space = k_space.to(dtype=torch.complex64)

    ac_data = find_ac_region(k_space)

    if 0 < batch_channels < k_space.shape[-2]:
        log_module.info(f"Batch channels - chunks of {batch_channels}")
        num_batches = int(np.ceil(k_space.shape[-2] / batch_channels))
    else:
        num_batches = 1
        batch_channels = k_space.shape[-2]

    nxy = torch.prod(torch.tensor(ac_data.shape[:2])).item()
    nce = batch_channels * ac_data.shape[-1] * loraks_neighborhood_size**2
    m_size = min(nxy, nce)
    if rank > m_size:
        msg = f"Available AC data matrix has smaller rank ({m_size}) than chosen parameter ({rank}). "
        log_module.error(msg)
        raise AttributeError(msg)
    log_module.info(f"Data matrix sizes ~ {nxy} x {nce}")
    start = max(0, m_size - reduced_num_nullspace_vec - rank)
    indices = torch.arange(start, m_size - rank)
    if indices.shape[0] < m_size:
        num_reduced_samples = max(
            min(m_size - indices.shape[0] - rank, sampled_reduced_nullspace_vec),
            0
        )
        indices = torch.sort(
            torch.concatenate([indices, torch.randperm(start)[:num_reduced_samples]], dim=0)
        ).values

    if indices.shape[0] < m_size - rank:
        log_module.info(f"Optimizing memory - reduction of AC data nullspace used.")
        log_module.info(f"Taking {indices.shape[0]} / {m_size - rank} nullspace eigenvectors")
    del indices
    torch.cuda.empty_cache()

    k_recon = torch.zeros_like(k_space)
    losses = []

    for idx_b in range(num_batches):
        log_module.info(f"Processing Batch {idx_b + 1}/{num_batches}")
        start = idx_b * batch_channels
        end = min((idx_b + 1) * batch_channels, m_size)

        k_batch = k_space[..., start:end, :]
        ac_batch = ac_data[..., start:end, :]
        sampling_mask_batch = sampling_mask[..., start:end, :]

        tmp, loss = recon_batch(
            k_space=k_batch, sampling_mask=sampling_mask_batch,
            ac_data=ac_batch,
            rank=rank, use_s_matrix=use_s_matrix,
            loraks_neighborhood_size=loraks_neighborhood_size,
            conv_tol=conv_tol, max_num_iter=max_num_iter, device=device,
            lr_low=lr_low, lr_high=lr_high, reduced_num_nullspace_vec=reduced_num_nullspace_vec,
            sampled_reduced_nullspace_vec=sampled_reduced_nullspace_vec
        )

        k_recon[..., start:end, :] = tmp
        losses.append(loss)

    return k_recon.cpu(), losses


def recon_tiago():
    logging.info("Set Device")
    device = torch.device("cuda")

    log_module.info(f"Set Paths")
    path = plib.Path(
        "/data/pt_np-jschmidt/data/12_tiago_repro/data/test_692_MSE_vFA_LORAKS/processed/").absolute()
    path_out = path.joinpath("recon")
    path_fig = path_out.joinpath("figures").absolute()
    path_fig.mkdir(exist_ok=True, parents=True)

    k_space = torch_load(path.joinpath("k_space.pt"))
    # k_space *= 1e3
    k_space = k_space.to(torch.complex64)
    sampling = torch_load(path.joinpath("sampling_mask.pt"))

    k_shape = k_space.shape

    img_fft = torch.zeros((*k_shape[:3], k_shape[-1]))
    log_module.info("FFT & RSOS")
    for idx_z in tqdm.trange(k_shape[2]):
        tmp = fft(k_space[:, :, idx_z].clone(), img_to_k=False, axes=(0, 1))
        img_fft[:, :, idx_z] = root_sum_of_squares(tmp, dim_channel=-2)

    nifti_save(img_fft, img_aff=torch.eye(4), path_to_dir=path_out, file_name="naive_fft_img")

    k_space_slice = k_space[:, :, k_shape[2] // 2].clone()
    k_slice_shape = k_space_slice.shape

    sampling = sampling[:, :, None, :].clone().expand(k_slice_shape).contiguous()

    batch_size = 1
    rank = 10

    k_recon_slice, losses = recon_data_consistency(
        k_space=k_space_slice, sampling_mask=sampling,
        rank=rank, loraks_neighborhood_size=5, matrix_type="S",
        batch_channels=batch_size,
        reduced_num_nullspace_vec=3000, sampled_reduced_nullspace_vec=2000,
        max_num_iter=500, device=device, lr_high=10, lr_low=0.1,
        conv_tol=1e-3
    )

    img_recon_slice = fft(k_recon_slice, img_to_k=False, axes=(0, 1))
    img_recon_slice_rsos = root_sum_of_squares(img_recon_slice, dim_channel=-2)

    nifti_save(img_recon_slice_rsos, img_aff=torch.eye(4), path_to_dir=path_out, file_name=f"recon_img_ac_cb{batch_size}_r{rank}")

    fig = go.Figure()
    for i, l in enumerate(losses):
        fig.add_trace(
            go.Scatter(y=l, name=f"Loss Batch {i+1}")
        )
    f_name = path_fig.joinpath("losses").with_suffix(".html")
    log_module.info(f"Saving figure {f_name}")
    fig.write_html(f_name)


def main_pbp():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"Set Device: {device}")

    logging.info(f"Set Paths")
    path = plib.Path(__name__).parent.absolute()
    path_fig = path.joinpath("figures").absolute()
    path_fig.mkdir(exist_ok=True, parents=True)

    logging.info("Set Phantom")
    nx, ny, nc, ne = (256, 256, 4, 1)
    phantom = Phantom.get_shepp_logan(shape=(nx, ny), num_coils=nc, num_echoes=ne)
    sl_us = phantom.sub_sample_ac_random_lines(acceleration=2, ac_lines=40).unsqueeze(-1)

    sl_us = sl_us.contiguous()
    img_us = torch.abs(fft(sl_us, axes=(0, 1)))
    # img_us = root_sum_of_squares(img_us, dim_channel=-2)
    mask = (torch.abs(sl_us) > 1e-9).to(torch.int)

    k_recon, losses = recon_data_consistency(
        k_space=sl_us, sampling_mask=mask, rank=50, max_num_iter=100,
        lr_low=1e-2, lr_high=1,
        loraks_neighborhood_size=5, matrix_type="S", device=device
    )
    img_recon = torch.abs(fft(k_recon, axes=(0, 1)))

    fig = psub.make_subplots(
        rows=3, cols=4
    )
    for i, d in enumerate([sl_us, img_us, k_recon, img_recon, img_recon - img_us]):
        if i % 2 == 0 and i < 4:
            d = torch.log(torch.abs(d))
        else:
            d = torch.abs(d)
        row = int(i / 2) + 1
        for c in range(2):
            col = int(i % 2) + 2 * c + 1
            fig.add_trace(
                go.Heatmap(z=d[:, :, c, 0], showscale=False),
                row=row, col=col
            )
            fig.update_xaxes(visible=False, row=row, col=col)
            xaxis = fig.data[-1].xaxis
            fig.update_yaxes(visible=False, scaleanchor=xaxis, row=row, col=col)
    fig_name = plib.Path("scratches/figures/js_ac_test.html").absolute()
    log_module.info(f"Saving figure: {fig_name}")
    fig.write_html(fig_name)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main_pbp()
