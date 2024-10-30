""" script functions and operators needed for loraks_arxv implementation - use torch"""
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


def c_operator(k_space_x_y_ch_t: torch.Tensor, indices: torch.Tensor):
    shape = k_space_x_y_ch_t.shape
    # build combined 3rd dim
    k_space_x_y_cht = torch.reshape(k_space_x_y_ch_t, (shape[0], shape[1], -1))
    # extract from matrix
    c_matrix = k_space_x_y_cht[indices[..., 0], indices[..., 1]]
    # now we want the neighborhoods of individual t-ch info to be concatenated into the neighborhood axes.
    c_matrix = torch.reshape(torch.movedim(c_matrix, -1, 1), (c_matrix.shape[0], -1))
    return c_matrix


def c_adjoint_operator(c_matrix: torch.Tensor, indices: torch.Tensor, k_space_dims: tuple):
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
    # flip - the indices arent mirrored symmetrically but the neighborhoods
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
    k_space_recon = torch.zeros((*k_space_dims[:2], n_tch), dtype=torch.complex128).to(s_matrix.device)
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

class C(MatrixOperatorLowRank2D):
    def __init__(self, k_space_dims_x_y_ch_t: tuple, nb_radius: int = 3,
                 device: torch.device = torch.get_default_device()):
        super().__init__(k_space_dims_x_y_ch_t=k_space_dims_x_y_ch_t, nb_radius=nb_radius, device=device)

    def _operator(self, k_space_x_y_ch_t: torch.tensor) -> torch.tensor:
        """
        operator to map k-space vector to loraks_arxv c matrix
        :param k_space_x_y_ch_t: k_space vector in 4d
        :return: C matrix, dims [(kx - 2R)*(ky - 2R)
        """
        # build combined 3rd dim
        k_space_x_y_cht = torch.reshape(k_space_x_y_ch_t, (self.k_space_dims[0], self.k_space_dims[1], -1))
        # extract from matrix
        c_matrix = k_space_x_y_cht[
            self.neighborhood_indices[:, :, 0],
            self.neighborhood_indices[:, :, 1]
        ]
        # now we want the neighborhoods of individual t-ch info to be concatenated into the neighborhood axes.
        c_matrix = torch.reshape(torch.movedim(c_matrix, -1, 1), (c_matrix.shape[0], -1))

        return c_matrix

    def _adjoint(self, x_matrix: torch.tensor) -> torch.tensor:
        """
        operator to map c-matrix back to k-space [K, n_R] -> [kx, ky]
        :param x_matrix:
        :return: flattened k-space vector
        """
        # build indices
        if x_matrix.shape[0] < x_matrix.shape[1]:
            # want neighborhood dim to be in column
            x_matrix = x_matrix.T
        # store shapes
        sm, sk = x_matrix.shape
        # get dims
        n_tch = int(sk / self.neighborhood_size)
        # extract sub-matrices - they are concatenated in neighborhood dimension for all t-ch images
        t_ch_idxs = (torch.arange(self.neighborhood_size)[:, None] +
                     torch.arange(n_tch)[None, :] * self.neighborhood_size)
        x_matrix = x_matrix[:, t_ch_idxs]

        # build k_space
        k_space_recon = torch.zeros((*self.k_space_dims[:2], n_tch), dtype=torch.complex128).to(x_matrix.device)
        # # fill k_space
        log_module.debug(f"build k-space from c-matrix")
        for idx_nb in range(self.neighborhood_size):
            k_space_recon[
                self.neighborhood_indices[:, idx_nb, 0], self.neighborhood_indices[:, idx_nb, 1]
            ] += x_matrix[:, idx_nb]
        # mask = self.p_star_p > 0
        # k_space_recon[mask] /= self.p_star_p[mask]

        return torch.reshape(k_space_recon, self.k_space_dims)

    def _get_neighborhood_indices(self) -> torch.Tensor:
        return get_idx_2d_circular_neighborhood_patches_in_shape(
            shape_2d=self.k_space_dims[:2], nb_radius=self.radius
        )

    def _get_neighborhood_indices_point_sym(self) -> torch.Tensor:
        return None

    @property
    def neighborhood_size(self) -> int:
        return self.neighborhood_indices.shape[1]


class S(MatrixOperatorLowRank2D):
    def __init__(self, k_space_dims_x_y_ch_t: tuple, nb_radius: int = 3,
                 device: torch.device = torch.get_default_device()):
        super().__init__(k_space_dims_x_y_ch_t=k_space_dims_x_y_ch_t, nb_radius=nb_radius, device=device)

    def _operator(self, k_space: torch.tensor) -> torch.tensor:
        # need shape into 2D if not given like this
        k_space = torch.reshape(k_space, (self.k_space_dims[0], self.k_space_dims[1], -1))
        # build S matrix
        log_module.debug(f"build s matrix")
        # we build the matrices per channel / time image
        s_p = k_space[self.neighborhood_indices[:, :, 0], self.neighborhood_indices[:, :, 1]]
        s_m = k_space[self.neighborhood_indices_pt_sym[:, :, 0], self.neighborhood_indices_pt_sym[:, :, 1]]
        # concatenate along respective dimensions
        s_matrix = torch.concatenate((
            torch.concatenate([(s_p - s_m).real, (-s_p + s_m).imag], dim=1),
            torch.concatenate([(s_p + s_m).imag, (s_p + s_m).real], dim=1)
            ), dim=0
        )
        # now we want the neighborhoods of individual t-ch info to be concatenated into the neighborhood axes.
        s_matrix = torch.reshape(torch.movedim(s_matrix, -1, 1), (s_matrix.shape[0], -1))
        return s_matrix

    def _adjoint(self, x_matrix: torch.tensor) -> torch.tensor:
        # store shapes
        sm, sk = x_matrix.shape
        # get dims
        snb = self.neighborhood_size
        n_tch = int(sk / snb)
        # extract sub-matrices - they are concatenated in neighborhood dimension for all t-ch images
        t_ch_idxs = torch.arange(snb)[:, None] + torch.arange(n_tch)[None, :] * snb
        x_matrix = x_matrix[:, t_ch_idxs]

        matrix_u, matrix_d = torch.tensor_split(x_matrix, 2, dim=0)
        srp_m_srm, msip_p_sim = torch.tensor_split(matrix_u, 2, dim=1)
        sip_p_sim, srp_p_srm = torch.tensor_split(matrix_d, 2, dim=1)
        # extract sub-sub
        srp = srp_m_srm + srp_p_srm
        srm = - srp_m_srm + srp_p_srm
        sip = sip_p_sim - msip_p_sim
        sim = msip_p_sim + sip_p_sim

        # build k_space
        k_space_recon = torch.zeros((*self.k_space_dims[:2], n_tch), dtype=torch.complex128).to(x_matrix.device)
        # # fill k_space
        log_module.debug(f"build k-space from s-matrix")
        nb = int(snb / 2)
        for idx_nb in range(nb):
            k_space_recon[
                self.neighborhood_indices[:, idx_nb, 0], self.neighborhood_indices[:, idx_nb, 1]
            ] += srp[:, idx_nb] + 1j * sip[:, idx_nb]
            k_space_recon[
                self.neighborhood_indices_pt_sym[:, idx_nb, 0], self.neighborhood_indices_pt_sym[:, idx_nb, 1]
            ] += srm[:, idx_nb] + 1j * sim[:, idx_nb]

        # mask = self.p_star_p > 0
        # k_space_recon[mask] /= self.p_star_p[mask]

        return torch.reshape(k_space_recon, self.k_space_dims)

    def _get_neighborhood_indices(self) -> (torch.Tensor, torch.Tensor):
        # we want to index through all circular neighborhoods completely included in the shape,
        indices = get_idx_2d_circular_neighborhood_patches_in_shape(
            shape_2d=self.k_space_dims[:2], nb_radius=self.radius,
        )
        return indices

    def _get_neighborhood_indices_point_sym(self):
        # additionally for each of the indices we want to find the corresponding point symmetric position
        # for this we find all grid points
        shape_grid = get_idx_2d_rectangular_grid(
            size_x=self.k_space_dims[0] - 2 * self.radius, size_y=self.k_space_dims[1] - 2 * self.radius
        ) + self.radius
        # need their corresponding point symmetrical position
        indices_point_sym = 2 * torch.floor(
            torch.tensor(self.k_space_dims[:2]) / 2
        ) - shape_grid - 1
        # and for those find all neighborhoods
        nb_grid = get_idx_2d_grid_circle_within_radius(radius=self.radius)
        # and combine
        indices_point_sym = indices_point_sym[:, None] + nb_grid[None, :]
        return indices_point_sym.to(torch.int)

    @property
    def neighborhood_size(self) -> int:
        return self.neighborhood_indices.shape[1] + self.neighborhood_indices_pt_sym.shape[1]


# Test code to verify that matrix sizes and operators work for the phantom data.
# It sets everything and runs one iteration C-Loraks and one iteration S-Loraks without optimizing
# anything.
# TODO: Extract good test methods for building, applying, etc. operators.
if __name__ == '__main__':
    # set up logging
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s -- %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    logging.info("load phantom")
    # load phantom
    size_x, size_y = (256, 256)
    sl_phantom = SheppLogan(size_xy=(size_x, size_y))
    sl_k_space = fft(sl_phantom, axes=(0, 1), img_to_k=True)
    # artifically "skip lines"in phase direction
    sl_k_space[:, torch.randint(low=0, high=size_y, size=(50,))] = 0.0

    num_samples = 1000
    r = 4

    op = C(nb_radius=r, k_space_dims_x_y_ch_t=(size_x, size_y, 1, 1))
    # construct c matrix
    m = op.operator(sl_k_space)

    target_rank = 25
    logging.info(f"C: matrix size: {m.shape}, target rank: {target_rank}")

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
    k_recon = op.operator_adjoint(s_lr_approx_c)
    img_recon_c = fft(k_recon, axes=(0, 1), img_to_k=False)
    psp_c = op.p_star_p

    op = S(nb_radius=r, k_space_dims_x_y_ch_t=(size_x, size_y, 1, 1))
    # construct c matrix
    m = op.operator(sl_k_space)
    target_rank = 40
    logging.info(f"S: matrix size: {m.shape}, target rank: {target_rank}")

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
    k_recon = op.operator_adjoint(s_lr_approx_s)
    img_recon_s = fft(k_recon, axes=(0, 1), img_to_k=False)

    # new version
    psp_new = s_adjoint_operator(
        s_operator(
            k_space_x_y_ch_t=torch.ones_like(k_recon), indices=get_idx_2d_circular_neighborhood_patches_in_shape(
                shape_2d=(size_x, size_y), nb_radius=r
            )
        ),
        indices=get_idx_2d_circular_neighborhood_patches_in_shape(
            shape_2d=(size_x, size_y), nb_radius=r
        ),
        k_space_dims=(size_x, size_y)
    )

    m_new = s_operator(
        sl_k_space, get_idx_2d_circular_neighborhood_patches_in_shape((size_x, size_y), nb_radius=r)
    )

    # do svd
    u, s, v = torch.linalg.svd(m_new, full_matrices=False)
    s[target_rank:] = 0.0

    # recon
    s_lr_approx_s = torch.matmul(
        torch.matmul(
            u, torch.diag(s).to(dtype=u.dtype)
        ),
        v
    )

    # get back to k - space
    k_recon = s_adjoint_operator(
        s_lr_approx_s, get_idx_2d_circular_neighborhood_patches_in_shape((size_x, size_y), nb_radius=r),
        (size_x, size_y)
    )

    img_recon_s_new = fft(k_recon, axes=(0, 1), img_to_k=False)

    sl_us_phantom = fft(sl_k_space, axes=(0, 1), img_to_k=False)

    plots = [sl_phantom,  img_recon_c, psp_c, sl_us_phantom, img_recon_s, img_recon_s_new, op.p_star_p, psp_new]

    fig = psub.make_subplots(
        rows=2, cols=len(plots)
    )
    for idx_i, i in enumerate(plots):
        r = 1 + int(idx_i / int(len(plots) / 2))
        c = int(idx_i % int(len(plots) / 2)) + 1
        if i is not None:
            fig.add_trace(
                go.Heatmap(z=torch.squeeze(torch.abs(i)).numpy(), colorscale='Magma', showscale=False),
                row=r, col=c
            )

    fig_path = plib.Path("./examples/recon/loraks_arxv/phantom_recon").absolute().with_suffix(".html")

    logging.info(f"write file: {fig_path}")
    fig.write_html(fig_path.as_posix())
