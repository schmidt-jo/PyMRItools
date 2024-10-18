""" script functions and operators needed for loraks implementation - use torch"""
import logging
from abc import ABC, abstractmethod
import pathlib as plib

import torch
import plotly.graph_objects as go
import plotly.subplots as psub

from pymritools.utils.phantom import SheppLogan
from pymritools.utils import (
    fft,
    get_idx_2d_circular_neighborhood_patches_in_shape,
    get_idx_2d_rectangular_grid,
    get_idx_2d_grid_circle_within_radius
)

log_module = logging.getLogger(__name__)


def shift_read_dir(data: torch.tensor, read_dir: int, forward: bool = True):
    if forward:
        return torch.movedim(data, read_dir, 0)
    else:
        return torch.movedim(data, 0, read_dir)


def get_k_radius_grid_points(radius: int) -> torch.tensor:
    """
    fn to generate all integer grid points in 2D within a radius
    :param radius: [int]
    :return: tensor, dim [#pts, 2] of pts xy within radius
    """
    # generate the neighborhood
    tmp_axis = torch.arange(-radius, radius + 1)
    # generate set of all possible points
    tmp_pts = torch.tensor([
        (x, y) for x in tmp_axis for y in tmp_axis if (x ** 2 + y ** 2 <= radius ** 2)
    ])
    return tmp_pts.to(torch.int)


def get_neighborhood(pt_xy: torch.tensor, radius: int):
    """
    fn to generate int grid points within radius around point (pt_xy)
    :param pt_xy: point in 2D gridded space dim [2]
    :param radius: integer value of radius
    :return: torch tensor with neighborhood grid points, dim [#pts, 2]
    """
    return pt_xy + get_k_radius_grid_points(radius=radius)


class LoraksOperator(ABC):
    """
    Base implementation of LORAKS operator,
    We want to implement the operator slice wise, for k-space data [x, y, ch, t].
    Merging of channel and time data allows for complementary sampling schemes per echo and
    is supposed to improve performance.
    Essentially there is only the C and S version (G was shown to behave suboptimal),
    This is the common base class
    """
    def __init__(self, k_space_dims_x_y_ch_t: tuple, nb_radius: int = 3,
                 device: torch.device = torch.get_default_device()):
        # save params
        self.radius: int = nb_radius
        self.k_space_dims: tuple = self._expand_dims_to_x_y_ch_t(k_space_dims_x_y_ch_t)
        self.device: torch.device = device

        # calculate the shape for combined x-y and ch-t dims
        self.reduced_k_space_dims = (
            k_space_dims_x_y_ch_t[0] * k_space_dims_x_y_ch_t[1],  # xy
            k_space_dims_x_y_ch_t[2] * k_space_dims_x_y_ch_t[3]  # ch - t
        )
        # need to build psp once with ones, such that we can extract it from the method
        self.p_star_p: torch.Tensor = torch.ones(
            (*self.k_space_dims[:2], self.reduced_k_space_dims[-1]), device=self.device, dtype=torch.int
        )
        self.neighborhood_indices: torch.Tensor = self._get_neighborhood_indices()
        self.neighborhood_indices_pt_sym: torch.Tensor = self._get_neighborhood_indices_point_sym()
        # update p_star_p, want this to be 2d + reduced last dim
        self.p_star_p = torch.reshape(
            torch.abs(self._get_p_star_p()), (*self.k_space_dims[:2], self.reduced_k_space_dims[-1])
        )

    @staticmethod
    def _expand_dims_to_x_y_ch_t(in_data: torch.Tensor | tuple):
        if torch.is_tensor(in_data):
            shape = in_data.shape
            while shape.__len__() < 4:
                # want the dimensions to be [x, y, ch, t], assume to be lacking time and or channel information
                in_data = in_data.unsqueeze(-1)
                shape = in_data.shape
        else:
            while in_data.__len__() < 4:
                # want the dimensions to be [x, y, ch, t], assume to be lacking time and or channel information
                in_data = (*in_data, 1)
            shape = in_data
        if shape.__len__() > 4:
            err = f"Operator only implemented for <4D data."
            log_module.error(err)
            raise AttributeError(err)
        return in_data

    @abstractmethod
    def _get_neighborhood_indices(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _get_neighborhood_indices_point_sym(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def neighborhood_size(self):
        return NotImplementedError

    def operator(self, k_space_x_y_ch_t: torch.Tensor) -> torch.Tensor:
        """ k-space input in 4d, [x, y, ch, t]"""
        # check for correct data shape, expand if necessary
        k_space_x_y_ch_t = self._expand_dims_to_x_y_ch_t(k_space_x_y_ch_t)
        return self._operator(k_space_x_y_ch_t)

    def operator_adjoint(self, x_matrix: torch.tensor) -> torch.tensor:
        return torch.squeeze(self._adjoint(x_matrix=x_matrix))

    @abstractmethod
    def _operator(self, k_space: torch.tensor) -> torch.tensor:
        """ to be implemented for each loraks type mode"""
        raise NotImplementedError

    @abstractmethod
    def _adjoint(self, x_matrix: torch.tensor) -> torch.tensor:
        raise NotImplementedError

    def _get_p_star_p(self):
        return self.operator_adjoint(
            self.operator(
                torch.ones(self.k_space_dims, device=self.device, dtype=torch.complex128)
            )
        )

    # def _get_k_space_pt_idxs(self, include_offset: bool = False, point_symmetric_mirror: bool = False):
    #     # give all points except radius
    #     offset_x = 0
    #     offset_y = 0
    #     if include_offset:
    #         # need to compute differences for odd and even k-space dims
    #         if self.k_space_dims[0] % 2 < 1e-5:
    #             offset_x = 1
    #         if self.k_space_dims[1] % 2 < 1e-5:
    #             offset_y = 1
    #     x_aranged = torch.arange(self.radius + offset_x, self.k_space_dims[0] - self.radius)
    #     y_aranged = torch.arange(self.radius + offset_y, self.k_space_dims[1] - self.radius)
    #     if point_symmetric_mirror:
    #         x_aranged = torch.flip(x_aranged, dims=(0,))
    #         y_aranged = torch.flip(y_aranged, dims=(0,))
    #     return torch.tensor([
    #         (x, y)
    #         for x in x_aranged
    #         for y in y_aranged
    #     ]).to(torch.int)

    # def _get_half_space_k_dims(self):
    #     k_center_minus = torch.floor(
    #         torch.tensor([
    #             (self.k_space_dims[0] - 1) / 2, self.k_space_dims[1] - 1
    #         ])).to(torch.int)
    #     k_center_plus = torch.ceil(
    #         torch.tensor([
    #             (self.k_space_dims[0] - 1) / 2, 0
    #         ])).to(torch.int)
    #     k = torch.min(torch.min(k_center_minus[0], self.k_space_dims[0] - k_center_plus[0])).item()
    #     nx_ny = torch.tensor([
    #         (x, y) for x in torch.arange(self.radius, k - self.radius)
    #         for y in torch.arange(self.radius, self.k_space_dims[1] - self.radius)
    #     ])
    #     minus_nx_ny_idxs = k_center_minus - nx_ny
    #     plus_nx_ny_idxs = k_center_plus + nx_ny
    #     return minus_nx_ny_idxs.to(torch.int), plus_nx_ny_idxs.to(torch.int)
    #
    # def get_point_symmetric_neighborhoods(self):
    #     # get neighborhood points
    #     kn_pts = get_k_radius_grid_points(radius=self.radius)
    #     # build neighborhoods
    #     # if even number of k-space points, k-space center aka 0 is considered positive.
    #     # ie. there is one more negative line/point than positives.
    #     # If odd center is exactly in the middle and we have one equal positive and negatives.
    #     # In this scenario, the central line would be present twice in the matrix (s-matrix)
    #     # get k-space-indexes
    #     p_nx_ny_idxs = self._get_k_space_pt_idxs(include_offset=True, point_symmetric_mirror=False)
    #     m_nx_ny_idxs = self._get_k_space_pt_idxs(include_offset=True, point_symmetric_mirror=True)
    #     p_nb_nx_ny_idxs = p_nx_ny_idxs[:, None] + kn_pts[None, :]
    #     # build inverted indexes - point symmetric origin in center
    #     m_nb_nx_ny_idxs = m_nx_ny_idxs[:, None] + kn_pts[None, :]
    #     return p_nb_nx_ny_idxs.to(torch.int), m_nb_nx_ny_idxs.to(torch.int)
    #
    # def get_lin_neighborhoods(self):
    #     # get neighborhood points
    #     kn_pts = get_k_radius_grid_points(radius=self.radius)
    #     # build neighborhoods
    #     # get k-space-indexes
    #     p_nx_ny_idxs = self._get_k_space_pt_idxs(include_offset=False, point_symmetric_mirror=False)
    #     p_nb_nx_ny_idxs = p_nx_ny_idxs[:, None] + kn_pts[None, :]
    #     return p_nb_nx_ny_idxs.to(torch.int)


class C(LoraksOperator):
    def __init__(self, k_space_dims_x_y_ch_t: tuple, nb_radius: int = 3):
        super().__init__(k_space_dims_x_y_ch_t=k_space_dims_x_y_ch_t, nb_radius=nb_radius)

    def _operator(self, k_space_x_y_ch_t: torch.tensor) -> torch.tensor:
        """
        operator to map k-space vector to loraks c matrix
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
        mask = self.p_star_p > 0
        k_space_recon[mask] /= self.p_star_p[mask]

        return torch.reshape(k_space_recon, self.k_space_dims)

    def _get_neighborhood_indices(self) -> torch.Tensor:
        return get_idx_2d_circular_neighborhood_patches_in_shape(
            shape_2d=self.k_space_dims[:2], nb_radius=self.radius, device=self.device
        )

    def _get_neighborhood_indices_point_sym(self) -> torch.Tensor:
        return None

    @property
    def neighborhood_size(self) -> int:
        return self.neighborhood_indices.shape[1]


class S(LoraksOperator):
    def __init__(self, k_space_dims_x_y_ch_t: tuple, nb_radius: int = 3):
        super().__init__(k_space_dims_x_y_ch_t=k_space_dims_x_y_ch_t, nb_radius=nb_radius)

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

        mask = self.p_star_p > 0
        k_space_recon[mask] /= self.p_star_p[mask]

        return torch.reshape(k_space_recon, self.k_space_dims)

    def _get_neighborhood_indices(self) -> (torch.Tensor, torch.Tensor):
        # we want to index through all circular neighborhoods completely included in the shape,
        indices = get_idx_2d_circular_neighborhood_patches_in_shape(
            shape_2d=self.k_space_dims[:2], nb_radius=self.radius, device=self.device,
        )
        return indices

    def _get_neighborhood_indices_point_sym(self):
        # additionally for each of the indices we want to find the corresponding point symmetric position
        # for this we find all grid points
        shape_grid = get_idx_2d_rectangular_grid(
            size_x=self.k_space_dims[0] - 2 * self.radius, size_y=self.k_space_dims[1] - 2 * self.radius,
            device=self.device
        ) + self.radius
        # need their corresponding point symmetrical position
        indices_point_sym = 2 * torch.floor(
            torch.tensor(self.k_space_dims[:2], device=self.device) / 2
        ) - shape_grid - 1
        # and for those find all neighborhoods
        nb_grid = get_idx_2d_grid_circle_within_radius(radius=self.radius, device=self.device)
        # and combine
        indices_point_sym = indices_point_sym[:, None] + nb_grid[None, :]
        return indices_point_sym.to(torch.int)

    @property
    def neighborhood_size(self) -> int:
        return self.neighborhood_indices.shape[1] + self.neighborhood_indices_pt_sym.shape[1]


if __name__ == '__main__':
    # set up logging
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s -- %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    logging.info("load phantom")
    # load phantom
    size_x, size_y = (256, 256)
    sl_phantom = SheppLogan(size_xy=(size_x, size_y))
    sl_k_space = fft(sl_phantom, axes=(0, 1), img_to_k=True)
    num_samples = 1000
    r = 4

    op = S(nb_radius=r, k_space_dims_x_y_ch_t=(size_x, size_y, 1, 1))
    # construct c matrix
    m = op.operator(sl_k_space)

    target_rank = 60
    logging.info(f"matrix size: {m.shape}, target rank: {target_rank}")

    # do svd
    u, s, v = torch.linalg.svd(m, full_matrices=False)
    s[target_rank:] = 0.0

    # recon
    s_lr_approx = torch.matmul(
        torch.matmul(
            u, torch.diag(s).to(dtype=u.dtype)
        ),
        v
    )

    # get back to k - space
    k_recon = op.operator_adjoint(s_lr_approx)
    img_recon = fft(k_recon, axes=(0, 1), img_to_k=False)

    plots = [sl_phantom, img_recon, op.p_star_p]
    fig = psub.make_subplots(
        rows=1, cols=len(plots)
    )
    for idx_i, i in enumerate(plots):
        fig.add_trace(
            go.Heatmap(z=torch.squeeze(torch.abs(i)).numpy(), colorscale='Magma', showscale=False),
            row=1, col=idx_i + 1
        )

    fig_path = plib.Path("./examples/recon/loraks/phantom_recon").absolute().with_suffix(".html")

    logging.info(f"write file: {fig_path}")
    fig.write_html(fig_path.as_posix())

