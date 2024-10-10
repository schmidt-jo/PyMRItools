import logging
import pathlib as plib
import tqdm

import torch
from torch import nn
from torch import optim as TorchOptim
import plotly.graph_objects as go
import plotly.subplots as psub

from pymritools.recon.loraks.recon import recon
from pymritools.utils.phantom import SheppLogan
from pymritools.utils import fft, root_sum_of_squares
from pymritools.recon.loraks.algorithm import operators

log_module = logging.getLogger(__name__)


def nystrom_method(matrix: torch.Tensor, rank: int, num_samples: int = None, sample_axis: int = 0):
    # matrix is the input matrix (symmetric, positive semidefinite)
    # we take this to be 2D with the dims [k-space, neighborhoods / concats]
    # num_samples is the number of rows to sample, if None we sample a quadratic matrix, aka as many as the shorter axis
    # rank is the target rank for the approximation

    nk, ns = matrix.shape
    if nk < ns:
        err = f"implemented for the shorter axis to be in 2nd dim but found dims {matrix.shape}"
    if num_samples is None:
        num_samples = ns

    # Step 1: Randomly sample m rows from matrix
    idx = torch.randperm(nk)[:num_samples]
    C = matrix[idx]
    W = matrix[:, idx][idx]

    # Step 2: Compute low-rank approximation of W
    U, S, V = torch.svd(W)
    S_k = torch.diag(S[:rank])  # Keep top k singular values
    U_k = U[:, :rank]

    # Step 3: Compute pseudo-inverse of W_k
    W_k_inv = U_k @ torch.diag(1 / S[:rank]) @ U_k.t()

    # Step 4: Compute the Nyström approximation
    G_approx = C @ W_k_inv @ C.t()

    return G_approx


def boosting_nystrom(G, p, m, k):
    # G is the input matrix
    # p is the number of boosting iterations
    # m is the number of columns for each Nyström approximation
    # k is the target rank

    n = G.size(0)
    G_approx = torch.zeros_like(G)

    for i in range(p):
        # Compute weak Nyström approximation
        G_nys = nystrom_method(G, m, k)

        # Combine approximations (uniform weighting for simplicity)
        G_approx += G_nys / p

    return G_approx


def randomized_svd(matrix: torch.Tensor, sampling_size: int, power_projections: int = 1, oversampling_factor: int = 5):
    # take matrix size to be dim [k-space, neighborhoods / concats]
    # But want to sample across k-space dims, hence transpose matrix
    matrix = torch.movedim(matrix, 0, 1)

    nnb, nk = matrix.shape

    # Generate a random Gaussian matrix
    sample_projection = torch.randn((nk, sampling_size), dtype=matrix.dtype, device=matrix.device)

    # Form the random projection, dim: [nnb, sampling size]
    sample_matrix = torch.matmul(matrix, sample_projection)

    for _ in range(power_projections):
        sample_matrix = torch.matmul(matrix, torch.matmul(matrix.T, sample_matrix))

    # Orthonormalize basis using QR decomposition
    q, _ = torch.linalg.qr(sample_matrix)

    # Obtain the low-rank approximation of the original matrix - project original matrix onto that orthonormal basis
    lr = torch.matmul(q.T, matrix)

    # Perform SVD on the low-rank approximation
    u, s, vh = torch.linalg.svd(lr, full_matrices=False)

    # s, vh should be approximately the matrix s, vh of the svd from random matrix theory
    # we can get the left singular values by back projection
    u_matrix = torch.matmul(q, u)

    return u_matrix, s, vh


def main():
    # to check the figures - use accessible path
    path_fig = plib.Path("/data/pt_np-jschmidt/data/05_code_dev/loraks")
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

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    log_module.info(f"using device: {device}")

    # setup operator
    loraks_operator = operators.S(k_space_dims=sl_image_recon_us.shape, radius=3)
    nb_size = loraks_operator.operator(torch.ones_like(sl_undersampled_k)).shape[-1]
    rank = 20

    svd_sampling_size = 5000
    matrix_size = loraks_operator.operator(torch.ones_like(sl_undersampled_k)).shape[0]
    # log_module.info(f"reducing matrix dimensions on long axes to "
    #                 f"{svd_sampling_size} (was ({matrix_size, nb_size}))")

    scaling_factor = torch.nan_to_num(
        1 / loraks_operator.p_star_p(torch.ones_like(sl_undersampled_k)),
        nan=0.0, posinf=0.0, neginf=0.0
    )

    s_threshold = torch.ones(nb_size, dtype=torch.float32, device=device)
    # s_threshold = torch.linspace(0, nb_size, nb_size, dtype=torch.float32, device=device)
    # s_threshold[rank:] = 0.0
    k = nn.Parameter(sl_undersampled_k.clone().to(device), requires_grad=True)

    optim = TorchOptim.SGD(params=[k], lr=0.1)

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

        # instead reconstruct the low rank approximation - need to movedims if using randomized svd
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
        optim.step()
        optim.zero_grad()

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


