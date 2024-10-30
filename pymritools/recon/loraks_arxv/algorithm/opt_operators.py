import torch

# C_get_k_space_pt_idxs_kernel
# S_get_k_space_pt_idxs_kernel
# quadratic_neighborhood_get_k_space_pt_idxs_kernel

# indices = C_get_k_space_pt_idxs_kernel(..)
# xxx = c_operator(k_space, indices)

def get_C_k_space_pt_idxs(
        nx: int,
        ny: int,
        radius: int,
        device: torch.device = torch.get_default_device()
):
    # give all points except radius
    x_indices = torch.arange(radius, nx - radius, dtype=torch.int, device=device)
    y_indices = torch.arange(radius, ny - radius, dtype=torch.int, device=device)
    x_grid, y_grid = torch.meshgrid(x_indices, y_indices, indexing='ij')
    return torch.stack([x_grid, y_grid], dim=-1).reshape(-1, 2)

def c_operator(k_space: torch.Tensor, indices: torch.Tensor):
    # put k_space back into 2D slice and 3rd dim is concatenated t-ch
    # TODO: How to squeeze the last two dimensions
    dims = k_space.shape
    k_space = torch.reshape(k_space, (dims[0], dims[1], -1))
    # extract from matrix
    c_matrix = k_space[indices[:, :, 0], indices[:, :, 1]]
    # now we want the neighborhoods of individual t-ch info to be concatenated into the neighborhood axes.
    c_matrix = torch.reshape(torch.movedim(c_matrix, -1, 1), (c_matrix.shape[0], -1))
    return c_matrix

def c_adjoint_operator(c_matrix: torch.Tensor, indices: torch.Tensor, k_space_dims):
    if c_matrix.shape[0] < c_matrix.shape[1]:
        # want neighborhood dim to be in column
        c_matrix = c_matrix.T
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
    return torch.reshape(k_space_recon, (-1, n_tch))


def get_k_space_pt_idxs(
        nx: int,
        ny: int,
        radius: int,
        include_offset: bool = False,
        point_symmetric_mirror: bool = False,
        device: torch.device = torch.get_default_device()
):
    # give all points except radius
    offset_x = 0
    offset_y = 0
    # if include_offset:
    #     # need to compute differences for odd and even k-space dims
    #     if nx % 2 == 0:
    #         offset_x = 1
    #     if ny % 2 == 0:
    #         offset_y = 1
    x_indices = torch.arange(radius + offset_x, nx - radius, dtype=torch.int, device=device)
    y_indices = torch.arange(radius + offset_y, ny - radius, dtype=torch.int, device=device)
    # if point_symmetric_mirror:
    #     x_indices = torch.flip(x_indices, dims=(0,))
    #     y_indices = torch.flip(y_indices, dims=(0,))
    x_grid, y_grid = torch.meshgrid(x_indices, y_indices, indexing='ij')
    return torch.stack([x_grid, y_grid], dim=-1).reshape(-1, 2)

def get_k_radius_grid_points(radius: int, device: torch.device = torch.get_default_device()) -> torch.tensor:
    tmp_axis = torch.arange(-radius, radius + 1, device=device)
    xx, yy = torch.meshgrid(tmp_axis, tmp_axis, indexing='ij')
    mask = (xx ** 2 + yy ** 2) <= radius ** 2
    tmp_pts = torch.stack([xx[mask], yy[mask]], dim=1)
    return tmp_pts

def get_k_radius_grid_points_jochen(radius: int) -> torch.tensor:
    tmp_axis = torch.arange(-radius, radius + 1)
    tmp_pts = torch.tensor([
        (x, y) for x in tmp_axis for y in tmp_axis if (x ** 2 + y ** 2 <= radius ** 2)
    ])
    return tmp_pts.to(torch.int)


# def get_point_symmetric_neighborhoods(radius: int):
#     # get neighborhood points
#     kn_pts = get_k_radius_grid_points(radius)
#     # build neighborhoods
#     # if even number of k-space points, k-space center aka 0 is considered positive.
#     # ie. there is one more negative line/point than positives.
#     # If odd center is exactly in the middle and we have one equal positive and negatives.
#     # In this scenario, the central line would be present twice in the matrix (s-matrix)
#     # get k-space-indexes
#     p_nx_ny_idxs = self.get_k_space_pt_idxs(include_offset=True, point_symmetric_mirror=False)
#     m_nx_ny_idxs = self.get_k_space_pt_idxs(include_offset=True, point_symmetric_mirror=True)
#     p_nb_nx_ny_idxs = p_nx_ny_idxs[:, None] + kn_pts[None, :]
#     # build inverted indexes - point symmetric origin in center
#     m_nb_nx_ny_idxs = m_nx_ny_idxs[:, None] + kn_pts[None, :]
#     return p_nb_nx_ny_idxs.to(torch.int), m_nb_nx_ny_idxs.to(torch.int)
#
#
# def s_operator():
#     def _operator(self, k_space: torch.tensor) -> torch.tensor:
#         # build neighborhoods
#         p_nb_nx_ny_idxs, m_nb_nx_ny_idxs = self.nb_coos_point_symmetric
#         # need to separate xy dims
#         k_space = torch.reshape(k_space, (self.k_space_dims[0], self.k_space_dims[1], -1))
#         # build S matrix
#         log_module.debug(f"build s matrix")
#         # we build the matrices per channel / time image
#         s_p = k_space[p_nb_nx_ny_idxs[:, :, 0], p_nb_nx_ny_idxs[:, :, 1]]
#         s_m = k_space[m_nb_nx_ny_idxs[:, :, 0], m_nb_nx_ny_idxs[:, :, 1]]
#         # concatenate along respective dimensions
#         s_matrix = torch.concatenate((
#             torch.concatenate([(s_p - s_m).real, (-s_p + s_m).imag], dim=1),
#             torch.concatenate([(s_p + s_m).imag, (s_p + s_m).real], dim=1)
#         ), dim=0
#         )
#         # now we want the neighborhoods of individual t-ch info to be concatenated into the neighborhood axes.
#         s_matrix = torch.reshape(torch.movedim(s_matrix, -1, 1), (s_matrix.shape[0], -1))
#         return s_matrix
