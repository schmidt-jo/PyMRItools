import logging
import pathlib as plib

import torch
import tqdm
import plotly.graph_objects as go
import plotly.subplots as psub

from pymritools.recon.loraks_dev.matrix_indexing import get_linear_indices
from pymritools.recon.loraks_dev.operators import c_operator, s_operator
from pymritools.utils import Phantom
from pymritools.utils import fft


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
    v_padded[pad_l:pad_l+ps, pad_d:pad_d+ps, :, :] = v_reshaped

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
    # Extract dimensions
    nx, ny, nc = k.shape  # Input tensor dimensions
    # complex rep
    h = v.shape[0] // 2
    v = v[::2] + 1j * v[1::2]

    mv_c = fourier_mv(k=k, v=v, ps=ps)
    mv_s = torch.conj(fourier_mv(k=torch.flip(k, dims=(0, 1)), v=v, ps=ps))

    return 2 * (mv_c - mv_s)


def main():
    logging.info("Set Device")
    device = torch.device("cuda")

    logging.info(f"Set Paths")
    path = plib.Path(__name__).parent.absolute()
    path_fig = path.joinpath("figures").absolute()
    path_fig.mkdir(exist_ok=True, parents=True)

    logging.info("Set Phantom")
    nx, ny, nc, ne = (256, 256, 3, 2)
    phantom = Phantom.get_shepp_logan(shape=(nx, ny), num_coils=nc, num_echoes=ne)
    sl_us = phantom.sub_sample_ac_weighted_lines(acceleration=3, ac_lines=30)

    sl_us = sl_us.contiguous()
    img_us = torch.abs(fft(sl_us, axes=(0, 1)))
    # img_us = root_sum_of_squares(img_us, dim_channel=-2)
    mask = (torch.abs(sl_us) > 1e-9).to(torch.int)

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
    sl_ac = sl_us[cx - ix:cx + ix, cy - iy:cy + iy, :, :]
    sl_ac = sl_ac.contiguous()
    mask = mask.to(torch.bool)
    k_data_consistency = sl_us[mask]

    logging.info("Set Parameters")
    rank_s = 100
    lambda_s = 0.1
    max_num_iter = 100

    logging.info("Set Matrix Indices and AC Matrix")
    # ToDo: Calculate size of AC subregion if whole region is to big for CPU?
    ac_indices, ac_matrix_shape = get_linear_indices(
        k_space_shape=sl_ac.shape, patch_shape=(5, 5, -1, -1), sample_directions=(1, 1, 0, 0)
    )
    ac_matrix_shape_s = tuple((torch.tensor(ac_matrix_shape, dtype=torch.int) * torch.tensor([2, 2])).tolist())
    # ac_matrix_c = c_operator(k_space=sl_ac, indices=ac_indices, matrix_shape=ac_matrix_shape).to(device)
    ac_matrix_s = s_operator(k_space=sl_ac, indices=ac_indices, matrix_shape=ac_matrix_shape_s).to(device)

    # ev_c, v_c = torch.linalg.eigh(ac_matrix_c.mH @ ac_matrix_c)
    ev_s, v_s = torch.linalg.eigh(ac_matrix_s.mH @ ac_matrix_s)
    # eigenvals and corresponding evs are in ascending order
    # v_c = v_c[:, :-rank_s]
    v_s = v_s[:, :-rank_s]
    # complex number representation
    # v_s = torch.reshape(v_s, (2 * nce, -1, v_s.shape[-1]))
    # v_s = v_s[::2] + v_s[1::2] * 1j
    # v_s = torch.reshape(v_s, (-1, v_s.shape[-1]))

    del ac_matrix_s
    torch.cuda.empty_cache()

    logging.info("Init optimization")
    k_init = torch.randn_like(sl_us) * 1e-5
    k_init[mask] = sl_us[mask]
    k = k_init.clone().to(device).requires_grad_()
    k_data_consistency = k_data_consistency.to(device)
    lr = torch.linspace(5e-3, 5e-4, max_num_iter, device=device)
    mask = mask.to(device)

    # do FFT speedups of haldar
    progress_bar = tqdm.trange(max_num_iter, desc="Optimization")
    for i in progress_bar:
        loss_1 = torch.linalg.norm(
            k[mask] - k_data_consistency
        )

        # matrix = s_operator(
        #         k_space=k, indices=indices, matrix_shape=matrix_shape
        # )
        # loss_2 = torch.linalg.norm(
        #     torch.matmul(matrix, v),
        #     ord="fro"
        # )
        # mv = fourier_mv(k=k.view(*k.shape[:2], -1), v=v_c, ps=5)
        mv = fourier_mv_s(k=k.view(*k.shape[:2], -1), v=v_s, ps=5)
        # loss_2 = fourier_mv_s(k=k.view(*k.shape[:2], -1), v=v_s, ps=5)
        loss_2 = torch.linalg.norm(mv, ord="fro")
        # loss_2 = torch.linalg.norm(mv_s, ord="fro")

        loss = loss_1 + lambda_s * loss_2
        loss.backward()

        with torch.no_grad():
            k -= lr[i] * k.grad
        k.grad.zero_()

        progress_bar.postfix = (
            f"loss low rank: {1e3 * loss_2.item():.2f} -- loss data: {1e3 * loss_1.item():.2f} -- "
            f"total_loss: {1e3 * loss.item():.2f}"
        )
    k_recon = k.detach().cpu()
    img_recon = torch.abs(fft(k_recon, axes=(0, 1)))

    fig = psub.make_subplots(
        rows=2, cols=4
    )
    for i, d in enumerate([sl_us, img_us, k_recon, img_recon]):
        if i % 2 == 0:
            d = torch.log(torch.abs(d))
        else:
            d = torch.abs(d)
        row = int(i % 2) + 1
        for c in range(2):
            col = int(i / 2) + 2 * c + 1
            fig.add_trace(
                go.Heatmap(z=d[:, :, c, 0], showscale=False),
                row=row, col=col
            )
            fig.update_xaxes(visible=False, row=row, col=col)
            xaxis = fig.data[-1].xaxis
            fig.update_yaxes(visible=False, scaleanchor=xaxis, row=row, col=col)
    fig.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
