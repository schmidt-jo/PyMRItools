import torch


def c_operator(k_space: torch.Tensor, indices: torch.Tensor, matrix_shape: tuple):
    """
    Maps from k-space with shape (nxyz, ncem) into the neighborhood-representation
    :param k_space: k-space with shape (nxyz, ncem)
    :param indices: neighborhood mapping with shape (nxyz valid, nb)
    :param matrix_shape: Shape of the neighborhood matrix
    :return: neighborhood representation with shape (nxyz valid, nb * ncem)
    """
    return k_space.view(-1)[indices].view(matrix_shape)


def c_adjoint_operator(c_matrix: torch.Tensor, indices: torch.Tensor, k_space_dims: tuple, device=None):
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
            dtype = torch.complex128
    k_space = k_space.view(*k_space.shape[:-2], -1)
    # effectively c - matrix in each channel
    s_p = k_space[..., indices].view(-1, *matrix_shape)
    s_m = k_space[..., indices_rev].view(-1, *matrix_shape)

    s_p_m = (s_p - s_m).to(dtype)
    s_u = torch.concatenate([s_p_m.real, -s_p_m.imag], dim=1)

    s_p_m = (s_p + s_m).to(dtype)
    s_d = torch.concatenate([s_p_m.imag, s_p_m.real], dim=1)

    s = torch.concatenate([s_u, s_d], dim=-1).contiguous()
    return s.view(-1, s.shape[-1])


