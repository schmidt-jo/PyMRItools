import torch
from pymritools.recon.loraks_dev_cleanup.matrix_indexing import get_linear_indices, get_circular_nb_indices_in_2d_shape


def c_operator(k_space: torch.Tensor, indices: torch.Tensor, matrix_shape: tuple):
    """
    Maps from k-space with shape (nxyz, ncem) into the neighborhood-representation
    :param k_space: k-space with shape (nxyz, ncem)
    :param indices: neighborhood mapping with shape (nxyz valid, nb)
    :param matrix_shape: Shape of the neighborhood matrix
    :return: neighborhood representation with shape (nxyz valid, nb * ncem)
    """
    return k_space.view(-1)[indices].view(matrix_shape)


def c_adjoint_operator(c_matrix: torch.Tensor, indices: torch.Tensor, k_space_dims: tuple):
    adj_c_matrix = torch.zeros(k_space_dims, dtype=c_matrix.dtype, device=c_matrix.device).view(-1)
    # Here is the crucial step - eliminate loops in the adjoint operation by using 1D linear indexing and
    # torch index_add
    adj_c_matrix = adj_c_matrix.index_add(0, indices.view(-1), c_matrix.view(-1))
    return adj_c_matrix.view(k_space_dims)


def s_operator(k_space: torch.Tensor, indices: torch.Tensor, indices_rev: torch.Tensor, matrix_shape: tuple):
    match k_space.dtype:
        case torch.float32:
            dtype = torch.complex64
        case torch.float64:
            dtype = torch.complex128
        case _:
            dtype = torch.complex64
    # we merge the spatial dimensions and keep the "batch" dimensions along to concatenate in front
    # (usually channels, echoes)
    k_space = k_space.view(*k_space.shape[:-2], -1)

    # effectively c - matrix in each channel
    s_p = k_space[..., indices].view(-1, *matrix_shape)
    #  but with mirrored indexing around k-space center
    s_m = k_space[..., indices_rev].view(-1, *matrix_shape)

    # calculate the quadrants
    # upper left and right
    s_p_m = (s_p - s_m).to(dtype)
    s_u = torch.concatenate([s_p_m.real, -s_p_m.imag], dim=1)
    # lower left and right
    s_p_m = (s_p + s_m).to(dtype)
    s_d = torch.concatenate([s_p_m.imag, s_p_m.real], dim=1)

    # concatenate upper and lower
    s = torch.concatenate([s_u, s_d], dim=-1).contiguous()
    # dims [ne nc, 2 * nb, ~ 2 * nxy]
    # we now merge the channel and echoes into the neighborhood dimension
    # to dims [2 * nc * ne * nb, ~ 2 * nxy
    return s.view(-1, s.shape[-1])


def s_adjoint_operator(matrix: torch.Tensor, indices: torch.Tensor, indices_rev: torch.Tensor, k_space_dims: tuple):
    # the input matrix is dims [2 * nb * nc * ne,  2 * nxy]
    # want to extract the channels and echoes first
    s = torch.reshape(matrix, (k_space_dims[0], indices.shape[0] * 2, -1))
    # get the quadrants
    s_u, s_d = torch.tensor_split(s, 2, dim=-1)

    s_ll, s_lr = torch.tensor_split(s_d, 2, dim=1)
    s_ul, s_ur = torch.tensor_split(s_u, 2, dim=1)

    spmm = s_ul - 1j * s_ur
    sppm = s_lr + 1j * s_ll

    s_p = (0.5 * (spmm + sppm)).contiguous()
    s_m = (0.5 * (sppm - spmm)).contiguous()

    adj_s_matrix = torch.zeros(k_space_dims, dtype=s_p.dtype, device=s_p.device).view(k_space_dims[0], -1)
    # Here is the crucial step - eliminate loops in the adjoint operation by using 1D linear indexing and
    # torch index_add
    adj_s_matrix = adj_s_matrix.index_add(1, indices.view(-1), s_p.view(s_p.shape[0], -1))
    adj_s_matrix = adj_s_matrix.index_add(1, indices_rev.view(-1), s_m.view(s_m.shape[0], -1))

    return adj_s_matrix.view(k_space_dims)


class Operator:
    def __init__(self, k_space_shape: tuple, patch_shape: tuple, sample_directions: tuple, device: torch.device):
        self.k_space_shape = k_space_shape
        self.patch_shape = patch_shape
        self.sample_directions = sample_directions
        self.device = device
        self.indices, self.matrix_shape = get_linear_indices(
            k_space_shape=k_space_shape, patch_shape=patch_shape, sample_directions=sample_directions
        )

    @property
    def count_matrix(self):
        in_ones = torch.ones(self.k_space_shape, dtype=torch.complex64, device=self.device)
        matrix = self.operator(in_ones)
        return self.adjoint_operator(matrix).to(torch.int)

    def operator(self, k_space: torch.Tensor):
        raise NotImplementedError("To be implemented by subclass")

    def adjoint_operator(self, matrix: torch.Tensor):
        raise NotImplementedError("To be implemented by subclass")


class C(Operator):
    def __init__(self, k_space_shape: tuple, patch_shape: tuple, sample_directions: tuple, device: torch.device):
        super().__init__(k_space_shape, patch_shape, sample_directions, device)

    def operator(self, k_space: torch.Tensor):
        return c_operator(k_space=k_space, indices=self.indices, matrix_shape=self.matrix_shape)

    def adjoint_operator(self, matrix: torch.Tensor):
        return c_adjoint_operator(c_matrix=matrix, indices=self.indices, k_space_dims=self.k_space_shape)


class S(Operator):
    def __init__(self, k_space_shape: tuple, patch_shape: tuple, sample_directions: tuple, device: torch.device):
        super().__init__(k_space_shape, patch_shape, sample_directions, device)
        self.indices = get_circular_nb_indices_in_2d_shape(
            k_space_2d_shape=k_space_shape[:2], nb_radius=max(patch_shape) // 2,
            reversed=False
        ).to(device)
        self.indices_rev = get_circular_nb_indices_in_2d_shape(
            k_space_2d_shape=k_space_shape[:2], nb_radius=max(patch_shape) // 2,
            reversed=True
        ).to(self.indices.device)

    def operator(self, k_space: torch.Tensor):
        return s_operator(k_space=k_space, indices=self.indices, indices_rev=self.indices_rev,
                          matrix_shape=self.matrix_shape)

    def adjoint_operator(self, matrix: torch.Tensor):
        return s_adjoint_operator(matrix=matrix, indices=self.indices, indices_rev=self.indices_rev)
