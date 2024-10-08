""" script functions and operators needed for loraks implementation - use torch"""
import torch
import logging
import pathlib as plib
from pymritools.utils.phantom import SheppLogan

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


class LoraksOperator:
    def __init__(self, k_space_dims: tuple, radius: int = 3):
        # save params
        self.radius: int = radius
        self.k_space_dims: tuple = k_space_dims  # [xy, t-ch]
        while self.k_space_dims.__len__() < 4:
            # want the dimensions to be [x, y, ch, t]
            self.k_space_dims = (*self.k_space_dims, 1)
        self.reduced_k_space_dims = (
            k_space_dims[0] * k_space_dims[1],  # xy
            k_space_dims[2] * k_space_dims[3]  # t-ch
        )
        self.nb_coos_point_symmetric = self.get_point_symmetric_neighborhoods()
        self.nb_coos_linear = self.get_lin_neighborhoods()

    def operator(self, k_space: torch.tensor) -> torch.tensor:
        """ k-space input in 2d, [xy, ch_t(possibly batched)]"""
        # check for single channel data
        if k_space.shape.__len__() == 1:
            k_space = k_space[:, None]
        return self._operator(k_space)

    def _operator(self, k_space: torch.tensor) -> torch.tensor:
        """ to be implemented for each loraks type mode"""
        raise NotImplementedError

    def operator_adjoint(self, x_matrix: torch.tensor) -> torch.tensor:
        return torch.squeeze(self._adjoint(x_matrix=x_matrix))

    def _adjoint(self, x_matrix: torch.tensor) -> torch.tensor:
        raise NotImplementedError

    def p_star_p(self, k_vector):
        return self.operator_adjoint(self.operator(k_vector))

    def _get_k_space_pt_idxs(self, include_offset: bool = False, point_symmetric_mirror: bool = False):
        # give all points except radius
        offset_x = 0
        offset_y = 0
        if include_offset:
            # need to compute differences for odd and even k-space dims
            if self.k_space_dims[0] % 2 < 1e-5:
                offset_x = 1
            if self.k_space_dims[1] % 2 < 1e-5:
                offset_y = 1
        x_aranged = torch.arange(self.radius + offset_x, self.k_space_dims[0] - self.radius)
        y_aranged = torch.arange(self.radius + offset_y, self.k_space_dims[1] - self.radius)
        if point_symmetric_mirror:
            x_aranged = torch.flip(x_aranged, dims=(0,))
            y_aranged = torch.flip(y_aranged, dims=(0,))
        return torch.tensor([
            (x, y)
            for x in x_aranged
            for y in y_aranged
        ]).to(torch.int)

    def _get_half_space_k_dims(self):
        k_center_minus = torch.floor(
            torch.tensor([
                (self.k_space_dims[0] - 1) / 2, self.k_space_dims[1] - 1
            ])).to(torch.int)
        k_center_plus = torch.ceil(
            torch.tensor([
                (self.k_space_dims[0] - 1) / 2, 0
            ])).to(torch.int)
        k = torch.min(torch.min(k_center_minus[0], self.k_space_dims[0] - k_center_plus[0])).item()
        nx_ny = torch.tensor([
            (x, y) for x in torch.arange(self.radius, k - self.radius)
            for y in torch.arange(self.radius, self.k_space_dims[1] - self.radius)
        ])
        minus_nx_ny_idxs = k_center_minus - nx_ny
        plus_nx_ny_idxs = k_center_plus + nx_ny
        return minus_nx_ny_idxs.to(torch.int), plus_nx_ny_idxs.to(torch.int)

    def get_point_symmetric_neighborhoods(self):
        # get neighborhood points
        kn_pts = get_k_radius_grid_points(radius=self.radius)
        # build neighborhoods
        # if even number of k-space points, k-space center aka 0 is considered positive.
        # ie. there is one more negative line/point than positives.
        # If odd center is exactly in the middle and we have one equal positive and negatives.
        # In this scenario, the central line would be present twice in the matrix (s-matrix)
        # get k-space-indexes
        p_nx_ny_idxs = self._get_k_space_pt_idxs(include_offset=True, point_symmetric_mirror=False)
        m_nx_ny_idxs = self._get_k_space_pt_idxs(include_offset=True, point_symmetric_mirror=True)
        p_nb_nx_ny_idxs = p_nx_ny_idxs[:, None] + kn_pts[None, :]
        # build inverted indexes - point symmetric origin in center
        m_nb_nx_ny_idxs = m_nx_ny_idxs[:, None] + kn_pts[None, :]
        return p_nb_nx_ny_idxs.to(torch.int), m_nb_nx_ny_idxs.to(torch.int)

    def get_lin_neighborhoods(self):
        # get neighborhood points
        kn_pts = get_k_radius_grid_points(radius=self.radius)
        # build neighborhoods
        # get k-space-indexes
        p_nx_ny_idxs = self._get_k_space_pt_idxs(include_offset=False, point_symmetric_mirror=False)
        p_nb_nx_ny_idxs = p_nx_ny_idxs[:, None] + kn_pts[None, :]
        return p_nb_nx_ny_idxs.to(torch.int)


class C(LoraksOperator):
    def __init__(self, k_space_dims: tuple, radius: int = 3):
        super().__init__(k_space_dims=k_space_dims, radius=radius)

    def _operator(self, k_space: torch.tensor) -> torch.tensor:
        """
        operator to map k-space vector to loraks c matrix
        :param k_space: flattened k_space vector
        :return: C matrix, dims [(kx - 2R)*(ky - 2R)
        """
        # put k_space back into 2D slice and 3rd dim is concatenated t-ch
        k_space = torch.reshape(k_space, (self.k_space_dims[0], self.k_space_dims[1], -1))
        # get indices for whole k-space
        p_nb_nx_ny = self.nb_coos_linear
        # extract from matrix
        c_matrix = k_space[p_nb_nx_ny[:, :, 0], p_nb_nx_ny[:, :, 1]]
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
        p_nb_nx_ny_idxs = self.get_lin_neighborhoods()
        if x_matrix.shape[0] < x_matrix.shape[1]:
            # want neighborhood dim to be in column
            x_matrix = x_matrix.T
        # store shapes
        sm, sk = x_matrix.shape
        # get dims
        nb = p_nb_nx_ny_idxs.shape[1]
        n_tch = int(sk / nb)
        # extract sub-matrices - they are concatenated in neighborhood dimension for all t-ch images
        t_ch_idxs = torch.arange(nb)[:, None] + torch.arange(n_tch)[None, :] * nb
        x_matrix = x_matrix[:, t_ch_idxs]

        # build k_space
        k_space_recon = torch.zeros((*self.k_space_dims[:2], n_tch), dtype=torch.complex128).to(x_matrix.device)
        # # fill k_space
        log_module.debug(f"build k-space from c-matrix")
        for idx_nb in range(nb):
            k_space_recon[
                p_nb_nx_ny_idxs[:, idx_nb, 0], p_nb_nx_ny_idxs[:, idx_nb, 1]
            ] += x_matrix[:, idx_nb]
        return torch.reshape(k_space_recon, (-1, n_tch))


class S(LoraksOperator):
    def __init__(self, k_space_dims: tuple, radius: int = 3):
        super().__init__(k_space_dims=k_space_dims, radius=radius)

    def _operator(self, k_space: torch.tensor) -> torch.tensor:
        # build neighborhoods
        p_nb_nx_ny_idxs , m_nb_nx_ny_idxs = self.nb_coos_point_symmetric
        # need to separate xy dims
        k_space = torch.reshape(k_space, (self.k_space_dims[0], self.k_space_dims[1], -1))
        # build S matrix
        log_module.debug(f"build s matrix")
        # we build the matrices per channel / time image
        s_p = k_space[p_nb_nx_ny_idxs[:, :, 0], p_nb_nx_ny_idxs[:, :, 1]]
        s_m = k_space[m_nb_nx_ny_idxs[:, :, 0], m_nb_nx_ny_idxs[:, :, 1]]
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
        # build neighborhood
        p_nb_nx_ny_idxs, m_nb_nx_ny_idxs = self.nb_coos_point_symmetric
        # store shapes
        sm, sk = x_matrix.shape
        # get dims
        nb = p_nb_nx_ny_idxs.shape[1]
        snb = nb * 2
        n_tch = int(sk / snb)
        # extract sub-matrices - they are concatenated in neighborhood dimension for all t-ch images
        t_ch_idxs = torch.arange(snb)[:, None] + torch.arange(n_tch)[None, :] * snb
        x_matrix = x_matrix[:, t_ch_idxs]

        srp_m_srm = x_matrix[:int(sm / 2), :int(snb / 2)]
        msip_p_sim = x_matrix[:int(sm / 2), int(snb / 2):]
        sip_p_sim = x_matrix[int(sm / 2):, :int(snb / 2)]
        srp_p_srm = x_matrix[int(sm / 2):, int(snb / 2):]
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
                p_nb_nx_ny_idxs[:, idx_nb, 0], p_nb_nx_ny_idxs[:, idx_nb, 1]
            ] += srp[:, idx_nb] + 1j * sip[:, idx_nb]
            k_space_recon[
                m_nb_nx_ny_idxs[:, idx_nb, 0], m_nb_nx_ny_idxs[:, idx_nb, 1]
            ] += srm[:, idx_nb] + 1j * sim[:, idx_nb]
        return torch.reshape(k_space_recon, (-1, n_tch))

    def get_half_space_k_dims(self):
        return self._get_half_space_k_dims()


class G(LoraksOperator):
    def __init__(self, k_space_dims: tuple, radius: int = 3):
        super().__init__(k_space_dims=k_space_dims, radius=radius)

    def _operator(self, k_space: torch.tensor) -> torch.tensor:
        # build k_space
        k_space = torch.reshape(k_space, self.k_space_dims)
        # get neighborhood points
        kn_pts = get_k_radius_grid_points(radius=self.radius)
        # get indices for centered origin k-space
        m_nx_ny_idxs, p_nx_ny_idxs = self._get_half_space_k_dims()
        p_nb_nx_ny_idxs = p_nx_ny_idxs[:, None] + kn_pts[None, :]

        # build gr and gi
        g_r = k_space[m_nx_ny_idxs[:, 0], m_nx_ny_idxs[:, 1]].real
        g_i = k_space[m_nx_ny_idxs[:, 0], m_nx_ny_idxs[:, 1]].imag
        # build Gr, Gi matrices
        G_r = k_space[p_nb_nx_ny_idxs[:, :, 0], p_nb_nx_ny_idxs[:, :, 1]].real
        G_i = k_space[p_nb_nx_ny_idxs[:, :, 0], p_nb_nx_ny_idxs[:, :, 1]].imag

        # build G matrix
        g_matrix_a = torch.concatenate([-g_r[:, None], G_r, -G_i], dim=1)
        g_matrix_b = torch.concatenate([g_i[:, None], G_i, G_r], dim=1)
        g_matrix = torch.concatenate((g_matrix_a, g_matrix_b), dim=0)

        return g_matrix

    def _adjoint(self, x_matrix: torch.tensor) -> torch.tensor:
        # get from G back to k - vector
        # get neighborhood points
        kn_pts = get_k_radius_grid_points(radius=self.radius)
        # get indices for centered origin k-space
        m_nx_ny_idxs, p_nx_ny_idxs = self._get_half_space_k_dims()
        p_nb_nx_ny_idxs = p_nx_ny_idxs[:, None] + kn_pts[None, :]
        # get shapes
        gk, gm = x_matrix.shape
        # extract matrices
        recon_gr = -x_matrix[:int(gk / 2), 0]
        recon_gi = x_matrix[int(gk / 2):, 0]
        recon_Gr_a = x_matrix[:int(gk / 2), 1:1 + int((gm - 1) / 2)]
        recon_Gr_b = x_matrix[int(gk / 2):, 1 + int((gm - 1) / 2):]
        recon_Gr = recon_Gr_a / 2 + recon_Gr_b / 2
        recon_Gi_a = -x_matrix[:int(gk / 2), 1 + int((gm - 1) / 2):]
        recon_Gi_b = x_matrix[int(gk / 2):, 1:1 + int((gm - 1) / 2)]
        recon_Gi = recon_Gi_a / 2 + recon_Gi_b / 2

        # build k_space and count
        recon_k_space = torch.zeros(self.k_space_dims, dtype=torch.complex128).to(x_matrix.device)
        # fill g's
        recon_k_space[m_nx_ny_idxs[:, 0], m_nx_ny_idxs[:, 1]] += recon_gr + 1j * recon_gi
        # fill G's
        for n_idx in range(kn_pts.shape[0]):
            recon_k_space[
                p_nb_nx_ny_idxs[:, n_idx, 0], p_nb_nx_ny_idxs[:, n_idx, 1]
            ] += recon_Gr[:, n_idx] + 1j * recon_Gi[:, n_idx]
        return torch.flatten(recon_k_space)


def get_operator_from_mode(mode: str, radius: int, k_space_dims: tuple) -> S | C | G:
    d_available = {
        "s": S, "S": S,
        "c": C, "C": C,
        "g": G, "G": G
    }
    if d_available.get(mode) is None:
        err = f"mode {mode} not available, Choose one of :{d_available.keys()}"
        log_module.error(err)
        raise AssertionError(err)
    return d_available.get(mode)(radius=radius, k_space_dims=k_space_dims)


def construct_low_rank_matrix_from_svd(matrix_tensor: torch.tensor, rank: int):
    """
    compute the svd, truncate singular values above given rank and reconstruct reduced rank c matrix.
    Batch ready

    :param matrix_tensor: c matrix dim [batch, K - num k-space entries, N_R - size of local neighborhood]
    :param rank: integer rank to keep
    :return: rank reduced c matrix [batch, K, N_R]
    """
    M = matrix_tensor
    device = M.device
    transposed = False
    if M.shape[-2] < M.shape[-1]:
        # want k-space-flat dimension to be penultimate dim
        M = torch.moveaxis(M, -1, -2)
        transposed = True
    log_module.debug(f"start svd")
    if torch.cuda.is_available():
        # svd with big whole slice whole t whole ch data took 16 s opposed to 7 min on cpu
        gpu = torch.device("cuda:0")
        M = M.to(gpu)
    u, s, v = torch.linalg.svd(M, full_matrices=False)
    log_module.debug(f"finished svd")
    log_module.debug(f"rank given {rank}, rank computed, i.e. max num of singular values: {s.shape[0]}. "
                     f"using lower one.")
    rank = min(s.shape[-1], rank)
    # compute rank deficient matrix, assume batch dimensions in first
    if s.shape.__len__() < 2:
        s = s[None, :]
    s_rd = torch.zeros_like(s)
    s_rd[:, :rank] = s[:, :rank]
    # use diag_embed to create batched diag matrices, use squeeze in case there is no batch dimension
    s_m = torch.squeeze(torch.diag_embed(s_rd).to(M.dtype))
    m_rd = torch.matmul(torch.matmul(u, s_m), v)
    if transposed:
        m_rd = torch.moveaxis(m_rd, -1, -2)
    # want to return c with K,N_R dims but v to be vectors of right null space
    # check matrix rank
    m_rd = m_rd.to(device)
    # rank_c = torch.linalg.matrix_rank(m_rd)
    # log_module.debug(f"verify rank: given {rank}, computed: {rank_c}")
    # free gpu memory
    del u, s, v, M, s_rd, s_m
    return m_rd


if __name__ == '__main__':
    # set up logging
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s -- %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    logging.info("load phantom")
    phantom_file = plib.Path("./external_files/sl_phantom_256.npy").absolute()
    fig_path = plib.Path("./external_files/figs_wt").absolute()
    # load phantom
    size_x, size_y = (256, 256)
    sl_phantom = SheppLogan(size_xy=(size_x, size_y))
    sl_k_space = torch.fft.fftshift(torch.fft.fft2(sl_phantom))
    num_samples = 1000
    r = 5
    k_pts = torch.tensor([
        (x, y) for x in torch.arange(sl_k_space.shape[0]) for y in torch.arange(sl_k_space.shape[1])
    ])
    operator_p_c = C(radius=r, k_space_dims=(size_x, size_y))
    # construct c matrix
    c = C.operator(torch.flatten(sl_k_space))

    # construct low rank rep
    rank = 73
    logging.info(f"SVD")
    c_lr, u, s, v = construct_low_rank_matrix_from_svd(matrix_tensor=c, rank=rank)
    # build nullspace vector reps
    _, _, nv = torch.linalg.svd(c_lr, full_matrices=False)

    # f functions are given within the neighborhood
    f_vecs = nv.conj()[rank:]
    # verify nullspace
    logging.info(f"verify nullspace: "
                 f"{torch.allclose(torch.matmul(c_lr, f_vecs.T), torch.zeros(1, dtype=torch.complex128))}")
    k_space_f = torch.zeros((f_vecs.shape[0], *sl_k_space.shape), dtype=torch.complex128)
    # need to fill the central neighborhood
    c_pts = torch.tensor([int(sl_k_space.shape[0] / 2), int(sl_k_space.shape[1] / 2)])[None, :]
    kn_pts = get_k_radius_grid_points(radius=r) + c_pts
    k_space_f[:, kn_pts[:, 0], kn_pts[:, 1]] = f_vecs

    # plotting.plot_nullspace_rep(k=k_space_f, fig_path=fig_path, static_image=False)
