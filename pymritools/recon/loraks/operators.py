import logging
import pathlib as plib

import torch
import plotly.graph_objects as go
import plotly.subplots as psub

from pymritools.utils.phantom import SheppLogan
from pymritools.utils import (
    fft, MatrixOperatorLowRank2D,
    get_idx_2d_circular_neighborhood_patches_in_shape,
    get_idx_2d_rectangular_grid,
    get_idx_2d_grid_circle_within_radius
)

log_module = logging.getLogger(__name__)


def c_operator(k_space: torch.Tensor, c_mapping: torch.Tensor, device=None):
    """
    Maps from k-space with shape (nx, ny, nc, nt) into the neighborhood-representation
    :param k_space: k-space with shape (nx, ny, nc, nt)
    :param c_mapping: neighborhood mapping with shape (nb, 2)
    :return: neighborhood representation with shape (nt, nb)
    """
    k_space = torch.flatten(k_space, 2)
    return k_space[c_mapping[:, :, 0], c_mapping[:, :, 1]].permute(0,2,1).flatten(1)



def c_adjoint_operator(c_matrix: torch.Tensor, indices: torch.Tensor, k_space_dims: tuple, device=None):
    # Do we need to ensure dimensions? k-space in first, neighborhood / ch / t in second.
    # store shapes
    sm, sk = c_matrix.shape
    # get dims
    nb = indices.shape[1]
    n_tch = int(sk / nb)
    # extract sub-matrices - they are concatenated in neighborhood dimension for all t-ch images
    t_ch_idxs = torch.arange(nb)[:, None] + torch.arange(n_tch)[None, :] * nb
    c_matrix = c_matrix[:, t_ch_idxs]

    # build k_space
    k_space_recon = torch.zeros((*k_space_dims[:2], n_tch), dtype=torch.complex128, device=c_matrix.device)
    for idx_nb in range(nb):
        k_space_recon[
            indices[:, idx_nb, 0], indices[:, idx_nb, 1]
        ] += c_matrix[:, idx_nb]
    return torch.reshape(k_space_recon, k_space_dims)


def s_operator(k_space_x_y_ch_t: torch.Tensor, indices: torch.Tensor):
    shape = k_space_x_y_ch_t.shape
    # need shape into 2D if not given like this
    k_space = torch.reshape(k_space_x_y_ch_t, (shape[0], shape[1], -1))
    #  we want to build a matrix point symmetric to k_space around the center in the first two dimensions:

    # build S matrix
    log_module.debug(f"build s matrix")
    # we build the matrices per channel / time image
    s_p = k_space[indices[..., 0], indices[..., 1]]
    # flip - the indices aren't mirrored symmetrically but the neighborhoods
    # should be in the same order on the first axes.
    s_m = torch.flip(k_space, dims=(0, 1))[indices[..., 0], indices[..., 1]]
    # s_m = k_space[self.neighborhood_indices_pt_sym[:, :, 0], self.neighborhood_indices_pt_sym[:, :, 1]]
    # concatenate along respective dimensions
    s_matrix = torch.concatenate((
        torch.concatenate([(s_p - s_m).real, (-s_p + s_m).imag], dim=1),
        torch.concatenate([(s_p + s_m).imag, (s_p + s_m).real], dim=1)
    ), dim=0
    )
    # now we want the neighborhoods of individual t-ch info to be concatenated into the neighborhood axes.
    s_matrix = torch.reshape(torch.movedim(s_matrix, -1, 1), (s_matrix.shape[0], -1))
    return s_matrix


def s_adjoint_operator(s_matrix: torch.Tensor, indices: torch.Tensor, k_space_dims: tuple):
    # store shapes
    sm, sk = s_matrix.shape
    # get dims
    snb = 2 * indices.shape[1]
    n_tch = int(sk / snb)
    # extract sub-matrices - they are concatenated in neighborhood dimension for all t-ch images
    t_ch_idxs = torch.arange(snb)[:, None] + torch.arange(n_tch)[None, :] * snb
    s_matrix = s_matrix[:, t_ch_idxs]

    matrix_u, matrix_d = torch.tensor_split(s_matrix, 2, dim=0)
    srp_m_srm, msip_p_sim = torch.tensor_split(matrix_u, 2, dim=1)
    sip_p_sim, srp_p_srm = torch.tensor_split(matrix_d, 2, dim=1)
    # extract sub-sub
    srp = srp_m_srm + srp_p_srm
    srm = - srp_m_srm + srp_p_srm
    sip = sip_p_sim - msip_p_sim
    sim = msip_p_sim + sip_p_sim

    # build k_space
    dtype = torch.complex128 if s_matrix.dtype == torch.float64 else torch.complex64
    k_space_recon = torch.zeros((*k_space_dims[:2], n_tch), dtype=dtype).to(s_matrix.device)
    # # fill k_space
    log_module.debug(f"build k-space from s-matrix")
    nb = int(snb / 2)
    for idx_nb in range(nb):
        k_space_recon[
            indices[:, idx_nb, 0], indices[:, idx_nb, 1]
        ] += srp[:, idx_nb] + 1j * sip[:, idx_nb]
        torch.flip(k_space_recon, dims=(0, 1))[
            indices[:, idx_nb, 0], indices[:, idx_nb, 1]
        ] += srm[:, idx_nb] + 1j * sim[:, idx_nb]

    # mask = self.p_star_p > 0
    # k_space_recon[mask] /= self.p_star_p[mask]

    return torch.reshape(k_space_recon, k_space_dims)


if __name__ == '__main__':
    # set up logging
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s -- %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    logging.info("load phantom")
    # load phantom
    size_x, size_y = (256, 256)
    sl_phantom = SheppLogan()
    sl_fs_img = sl_phantom.get_2D_image(shape=(size_x, size_y), as_torch_tensor=True)
    sl_us_k = sl_phantom.get_sub_sampled_k_space(
        shape=(size_x, size_y), acceleration=3, ac_lines=32, mode="weighted",
        as_torch_tensor=True
    )
    # cast to dimensions [x, y, ch, t]
    sl_us_k = sl_us_k[:, :, None, None]
    img_recon_us = fft(sl_us_k, img_to_k=False, axes=(0, 1))

    # get indices
    nb_indices = get_idx_2d_circular_neighborhood_patches_in_shape(shape_2d=(size_x, size_y), nb_radius=3)

    log_module.info("C matrix")
    # construct c matrix
    c_count_matrix = torch.abs(
        c_adjoint_operator(
            c_matrix=c_operator(
                k_space_x_y_ch_t=torch.ones_like(sl_us_k), indices=nb_indices
            ),
            indices=nb_indices, k_space_dims=(size_x, size_y)
        )
    ).to(torch.int)

    m = c_operator(k_space_x_y_ch_t=sl_us_k, indices=nb_indices)

    target_rank = 20
    log_module.info(f"C: matrix size: {m.shape}, target rank: {target_rank}")

    # do svd
    u, s, v = torch.linalg.svd(m, full_matrices=False)
    s[target_rank:] = 0.0

    # recon
    s_lr_approx_c = torch.matmul(
        torch.matmul(
            u, torch.diag(s).to(dtype=u.dtype)
        ),
        v
    )

    # get back to k - space
    k_recon_c = c_adjoint_operator(c_matrix=s_lr_approx_c, indices=nb_indices, k_space_dims=(size_x, size_y))
    mask = c_count_matrix > 1e-9
    k_recon_c[mask] = k_recon_c[mask] / c_count_matrix[mask]
    # fft
    img_recon_c = fft(k_recon_c, img_to_k=False, axes=(0, 1))

    log_module.info("S matrix")
    # construct c matrix
    s_count_matrix = torch.abs(
        s_adjoint_operator(
            s_matrix=s_operator(
                k_space_x_y_ch_t=torch.ones_like(sl_us_k), indices=nb_indices
            ),
            indices=nb_indices, k_space_dims=(size_x, size_y)
        )
    ).to(torch.int)

    m = s_operator(k_space_x_y_ch_t=sl_us_k, indices=nb_indices)

    target_rank = 30
    log_module.info(f"S: matrix size: {m.shape}, target rank: {target_rank}")

    # do svd
    u, s, v = torch.linalg.svd(m, full_matrices=False)
    s[target_rank:] = 0.0

    # recon
    s_lr_approx_s = torch.matmul(
        torch.matmul(
            u, torch.diag(s).to(dtype=u.dtype)
        ),
        v
    )

    # get back to k - space
    k_recon_s = s_adjoint_operator(s_matrix=s_lr_approx_s, indices=nb_indices, k_space_dims=(size_x, size_y))
    mask = s_count_matrix > 1e-9
    k_recon_s[mask] = k_recon_s[mask] / s_count_matrix[mask]
    # fft
    img_recon_s = fft(k_recon_s, img_to_k=False, axes=(0, 1))

    plots = [sl_fs_img, img_recon_us, img_recon_c, img_recon_s, c_count_matrix, s_count_matrix]
    names = ["phantom", "us fft", "c recon", "s recon", "c counts", "s counts"]
    fig = psub.make_subplots(
        rows=1, cols=len(plots),
        column_titles=names
    )
    for idx_i, i in enumerate(plots):
        r = 1
        c = idx_i + 1
        if i is not None:
            fig.add_trace(
                go.Heatmap(z=torch.squeeze(torch.abs(i)).numpy(), colorscale='Magma', showscale=False),
                row=r, col=c
            )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig_path = plib.Path("./examples/recon/loraks/phantom_recon").absolute().with_suffix(".html")

    logging.info(f"write file: {fig_path}")
    fig.write_html(fig_path.as_posix())
