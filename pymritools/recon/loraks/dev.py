import logging
import pathlib as plib
import tqdm

import torch
import plotly.graph_objects as go
import plotly.subplots as psub

from pymritools.utils.phantom import SheppLogan
from pymritools.utils import fft, root_sum_of_squares, randomized_svd
from pymritools.recon.loraks.algorithm import operators

log_module = logging.getLogger(__name__)


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    log_module.info(f"using device: {device}")

    # to check the figures - use accessible path
    # path_fig = plib.Path("/data/pt_np-jschmidt/data/05_code_dev/loraks")
    path_fig = plib.Path("/home/patrick/Workspace/PycharmProjects/PyMRItools/results")
    path_fig.mkdir(exist_ok=True, parents=True)

    # here we can adjust the phantom sizes
    size_x, size_y = 200, 200
    # import shepp logan
    max_val = 100
    sl_phantom = SheppLogan((size_x, size_y), as_torch_tensor=True) * max_val
    # convert to k-space
    sl_k = fft(sl_phantom, img_to_k=True, axes=(0, 1))[:, :, None, None]
    # set up sampling pattern - keep central phase encodes and skip some outer ones
    sampling_mask = torch.zeros_like(sl_k, dtype=torch.int)
    sampling_mask[:, torch.randint(low=0, high=size_y, size=(int(size_y/2),))] = 1
    sampling_mask[:, ::3] = 1
    sampling_mask[:, int(2/5 * size_y):int(3/5 * size_y)] = 1
    # sampling_mask = torch.load("/LOCAL/jschmidt/PyMRItools/examples/raw_data/results/k_sampling_mask.pt")

    sl_undersampled_k = sl_k * sampling_mask

    # sl_undersampled_k = torch.load("/LOCAL/jschmidt/PyMRItools/examples/raw_data/results/k_space.pt")
    # take only middle slice
    # sl_undersampled_k = sl_undersampled_k[:, :, int(sl_undersampled_k.shape[2] / 2)]
    sl_image_recon_us = torch.abs(fft(sl_undersampled_k, img_to_k=False, axes=(0, 1)))

    # setup operator
    loraks_operator = operators.S(k_space_dims_x_y_ch_t=sl_image_recon_us.shape, nb_radius=3)
    nb_size = loraks_operator.operator(torch.ones_like(sl_undersampled_k)).shape[-1]
    rank = 20

    svd_sampling_size = 5000
    matrix_size = loraks_operator.operator(torch.ones_like(sl_undersampled_k)).shape[0]
    # log_module.info(f"reducing matrix dimensions on long axes to "
    #                 f"{svd_sampling_size} (was ({matrix_size, nb_size}))")

    # Todo: Fix nans in a better way or check
    scaling_factor = torch.nan_to_num(
        1 / loraks_operator.p_star_p(torch.ones_like(sl_undersampled_k, device=device)),
        nan=0.0, posinf=0.0, neginf=0.0
    )

    s_threshold = torch.ones(nb_size, dtype=torch.float32, device=device)
    # s_threshold = torch.linspace(0, nb_size, nb_size, dtype=torch.float32, device=device)
    # s_threshold[rank:] = 0.0
    # k = nn.Parameter(sl_undersampled_k.clone().to(device), requires_grad=True)
    k = sl_undersampled_k.clone().to(device).requires_grad_(True)

    # optim = TorchOptim.SGD(params=[k], lr=0.1)

    max_iter = 50
    data_consistency = 0.95

    losses = []
    bar = tqdm.trange(max_iter, desc="Optimization")
    for _ in bar:
        # get operator matrix
        matrix = loraks_operator.operator(k)

        # do svd
        # we can use torch svd, or try the randomized version, see above

        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        # u, s, vh = randomized_svd(matrix, sampling_size=svd_sampling_size, oversampling_factor=5, power_projections=2)

        # first part of loss
        # to do, try: cutoff lower svals via sig function or another smooth function
        st = s_threshold.clone()
        st[rank:] = 0.0
        s_r = s * st

        # not using the sum anymore
        # loss_1 = torch.sum(torch.pow(s * s_threshold, 1.5))
        # instead reconstruct the low rank approximation
        matrix_recon_loraks = torch.matmul(
            torch.matmul(u, torch.diag(s_r).to(u.dtype)),
            vh
        )
        # first part of loss
        # calculate difference to low rank approx
        loss_1 = torch.linalg.norm(matrix - matrix_recon_loraks)

        # second part, calculate reconstructed k
        k_recon_loraks = torch.reshape(
            loraks_operator.operator_adjoint(matrix_recon_loraks) * scaling_factor, sl_image_recon_us.shape
        )
        # take difference to sampled k for samples
        loss_2 = torch.linalg.norm(k_recon_loraks * sampling_mask - sl_undersampled_k)

        loss = data_consistency * loss_2 + (1 - data_consistency) * loss_1

        loss.backward()

        with torch.no_grad():
            # gu, gs, gv = torch.linalg.svd(torch.squeeze(k.grad), full_matrices=False)
            # # want to step along svd axis
            # gs[10:] = 0.0
            # p_grad = torch.matmul(torch.matmul(gu, torch.diag(gs).to(gu.dtype)), gv)
            k -= k.grad * 0.4

        k.grad.zero_()
        # optim.step()
        # optim.zero_grad()
        losses.append(loss.item())

        bar.postfix = (
            f"loss 1: {loss_1.item():.2f} -- loss 2: {loss_2.item():.2f} -- total_loss: {loss.item():.2f} -- rank: {rank}"
        )

    matrix_recon_loraks = torch.reshape(k.detach(), sl_undersampled_k.shape).cpu()
    recon_image = fft(matrix_recon_loraks, img_to_k=False, axes=(0, 1))

    idx_c = int(recon_image.shape[-2] / 2)
    idx_t = int(recon_image.shape[-1] / 2)

    fig = psub.make_subplots(
        rows=2, cols=4,
        specs=[
            [{}, {}, {}, {}],
            [{"colspan": 4}, None, None, None]
        ]
    )
    for idx_d, d in enumerate([recon_image, sl_image_recon_us, matrix_recon_loraks, sl_undersampled_k]):
        row = 1
        col = 1 + idx_d
        d = torch.abs(d[:, :, idx_c, idx_t])
        if idx_d > 1:
            zmin = -14
            zmax = 0
            d = torch.log(d)
        else:
            zmin = 0
            zmax = max_val
        fig.add_trace(
            go.Heatmap(
                z=d.numpy(),
                zmin=zmin, zmax=zmax,
                colorscale="Magma", showscale=False),
            row=row, col=col
        )
        x = fig.data[-1].xaxis
        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, scaleanchor=x, row=row, col=col)

    fig.add_trace(go.Scattergl(y=losses, name="loss"), row=2, col=1)
    fig.update_xaxes(title="Iteration", row=2, col=1)
    fig.update_yaxes(title="Loss", row=2, col=1)

    fig.update_layout(
        title=f"Results Rank: {rank}, data consistency: {data_consistency:.2f}"
    )
    file_name = path_fig.joinpath(f"results_dc{data_consistency:.2f}_r{rank}".replace(".", "p")).with_suffix(".html")
    log_module.info(f"saving figure to {file_name}")
    fig.write_html(file_name.as_posix())


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s")
    main()


