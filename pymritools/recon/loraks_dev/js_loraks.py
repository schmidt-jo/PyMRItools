import logging
import pathlib as plib

import numpy as np
import torch

import plotly.graph_objects as go
import plotly.subplots as psub
import tqdm

from pymritools.recon.loraks_dev.matrix_indexing import get_linear_indices
from pymritools.recon.loraks_dev.operators import c_operator, s_operator
from pymritools.utils import Phantom, fft, root_sum_of_squares, torch_load, nifti_save

log_module = logging.getLogger(__name__)



def recon_data_consistency(
        k_space: torch.Tensor, sampling_mask: torch.Tensor,
        rank: int, loraks_neighborhood_size: int = 5, matrix_type: str = "S",
        max_num_iter: int = 200,
        device: torch.device = torch.get_default_device()):
    if matrix_type.capitalize() == "S":
        use_s_matrix = True
    elif matrix_type.capitalize() == "C":
        use_s_matrix = False
    else:
        err = "Currently only S or C matrix LORAKS types are supported."
        log_module.error(err)
        raise ValueError(err)
    k_space = k_space.to(dtype=torch.complex64)

    mask = sampling_mask.to(torch.bool)
    mask_unsampled = ~mask

    logging.info("Set Matrix Indices and AC Matrix")
    # ToDo: Calculate size of AC subregion if whole region is to big for CPU?
    sample_dir = torch.zeros(len(k_space.shape), dtype=torch.int)
    sample_dir[:2] = 1
    patch_shape = torch.full(size=(len(k_space.shape),), fill_value=-1, dtype=torch.int)
    patch_shape[:2] = loraks_neighborhood_size

    indices, matrix_shape = get_linear_indices(
        k_space_shape=k_space.shape,
        patch_shape=tuple(patch_shape.tolist()),
        sample_directions=tuple(sample_dir.tolist())
    )

    if use_s_matrix:
        matrix_shape = tuple((torch.tensor(matrix_shape, dtype=torch.int) * torch.tensor([2, 2])).tolist())

    logging.info("Init optimization")
    # we optimize only unmeasured data
    # find read direction
    read_dir = -1
    for i in range(2):
        tmp_mask = torch.movedim(mask[:, :, 0].to(torch.int), i, 0)
        shape = list(tmp_mask.shape)
        shape.pop(1)
        if torch.sum(tmp_mask[:, tmp_mask.shape[1] // 2]) == torch.prod(torch.tensor(shape, dtype=torch.int)):
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
    k = torch.zeros(tuple(shape), dtype=k_space.dtype, device=device, requires_grad=True)

    lr = torch.linspace(1e-3, 1e-4, max_num_iter, device=device)
    mask_unsampled = mask_unsampled.to(device)

    # we take the 0 filled sampled data to the device
    k_sampled = k_space.to(device)
    # build truncation vector
    oversampling = 10
    q = rank + oversampling
    s_tr = torch.zeros(q, device=device)
    s_tr[:-oversampling] = 1

    progress_bar = tqdm.trange(max_num_iter, desc="Optimization")
    for i in progress_bar:
        # we need a zero filling operator to compute the losses
        tmp = torch.zeros_like(k_sampled)
        tmp[mask_unsampled] = k.flatten()

        if use_s_matrix:
            matrix = s_operator(k_space=k_sampled+tmp, indices=indices, matrix_shape=matrix_shape)
        else:
            matrix = c_operator(k_space=k_sampled+tmp, indices=indices, matrix_shape=matrix_shape)

        # do svd
        u, s, v = torch.svd_lowrank(A=matrix, q=rank+10, niter=2)
        # truncate
        s = s_tr * s
        # build lr-matrix
        matrix_lr = torch.matmul(u * s.to(u.dtype), v.mH)

        loss = torch.linalg.norm(matrix - matrix_lr, ord="fro")

        loss.backward()

        with torch.no_grad():
            k -= lr[i] * k.grad

        norm_grad = torch.linalg.norm(k.grad).item()
        progress_bar.postfix = (
            f"loss: {loss.item():.2f}, norm k: {torch.linalg.norm(k).item():.3f}, norm grad: {norm_grad:.3f}"
        )
        k.grad.zero_()
        if norm_grad < 1e-3:
            msg = f"Optimization converged at step: {i+1}"
            log_module.info(msg)
            break
    # build data
    tmp = torch.zeros_like(k_sampled)
    tmp[mask_unsampled] = k.flatten().detach()
    k_recon = k_sampled + tmp
    return k_recon.cpu()


def recon_tiago():
    logging.info("Set Device")
    device = torch.device("cuda")

    log_module.info(f"Set Paths")
    path = plib.Path(
        "/data/pt_np-jschmidt/data/12_tiago_repro/data/test_692_MSE_vFA_LORAKS/processed/").absolute()
    path_out = path.joinpath("recon")
    path_fig = path_out.joinpath("figures").absolute()
    path_fig.mkdir(exist_ok=True, parents=True)

    k_space = torch_load(path.joinpath("k_space.pt")).to(torch.complex64)
    sampling = torch_load(path.joinpath("sampling_mask.pt"))

    k_shape = k_space.shape

    img_fft = torch.zeros((*k_shape[:3], k_shape[-1]))
    log_module.info("FFT & RSOS")
    for idx_z in tqdm.trange(k_shape[2]):
        tmp = fft(k_space[:, :, idx_z], img_to_k=False, axes=(0, 1))
        img_fft[:, :, idx_z] = root_sum_of_squares(tmp, dim_channel=-2)

    nifti_save(img_fft, img_aff=torch.eye(4), path_to_dir=path_out, file_name="naive_fft_img")

    k_space_slice = k_space[:, :, k_shape[2] // 2].clone()
    k_slice_shape = k_space_slice.shape

    sampling = sampling[:, :, None, :].expand(k_slice_shape).contiguous()

    log_module.info(f"Batching")
    batch_size = 6
    num_batches = int(np.ceil(k_space.shape[-2] / batch_size))
    k_recon_slice = torch.zeros_like(k_space_slice)

    for b in range(num_batches):
        log_module.info(f"Processing batch {b+1} / {num_batches}")
        start = b * batch_size
        end = min((b + 1) * batch_size, k_space.shape[3])
        k_recon_slice[..., start:end, :] = recon_data_consistency(
            k_space=k_space_slice[..., start:end, :], sampling_mask=sampling[..., start:end, :],
            rank=150, loraks_neighborhood_size=5, matrix_type="S",
            max_num_iter=200, device=device
        )

    img_recon_slice = fft(k_recon_slice, img_to_k=False, axes=(0, 1))
    img_recon_slice = root_sum_of_squares(img_recon_slice, dim_channel=-2)

    nifti_save(img_recon_slice, img_aff=torch.eye(4), path_to_dir=path_out, file_name="recon_img_plo_cb6_1250")


def main_pbp():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"Set Device: {device}")

    logging.info(f"Set Paths")
    path = plib.Path(__name__).parent.absolute()
    path_fig = path.joinpath("figures").absolute()
    path_fig.mkdir(exist_ok=True, parents=True)

    logging.info("Set Phantom")
    nx, ny, nc, ne = (256, 256, 32, 8)
    phantom = Phantom.get_shepp_logan(shape=(nx, ny), num_coils=nc, num_echoes=ne)
    sl_us = phantom.sub_sample_ac_random_lines(acceleration=4, ac_lines=40)

    sl_us = sl_us.contiguous()
    img_us = torch.abs(fft(sl_us, axes=(0, 1)))
    # img_us = root_sum_of_squares(img_us, dim_channel=-2)
    mask = (torch.abs(sl_us) > 1e-9).to(torch.int)

    # k_recon = recon(
    #     k_space=sl_us, sampling_mask=mask, rank=50,
    #     loraks_neighborhood_size=5, lambda_reg=0.1, matrix_type="S",
    #     max_num_iter=200, device=device
    # )
    k_recon = torch.zeros_like(sl_us)

    batch_size_channels = 16
    num_batches = int(np.ceil(k_recon.shape[-2] / batch_size_channels))

    for idx_b in range(num_batches):
        log_module.info(f"Processing batch: {idx_b + 1} / {num_batches}")
        start = idx_b * batch_size_channels
        end = min((idx_b + 1) * batch_size_channels, k_recon.shape[-2])
        k_recon[..., start:end, :] = recon_data_consistency(
            k_space=sl_us[..., start:end, :], sampling_mask=mask[..., start:end, :], rank=150, max_num_iter=150,
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
    fig_name = plib.Path("scratches/figures/js_loraks_test.html").absolute()
    log_module.info(f"Saving figure: {fig_name}")
    fig.write_html(fig_name)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    recon_tiago()
