import logging
import pathlib as plib

import torch


def c_operator(k_space: torch.Tensor, indices: torch.Tensor, matrix_shape: tuple):
    """
    Maps from k-space with shape (nxyz, ncem) into the neighborhood-representation
    :param k_space: k-space with shape (nxyz, ncem)
    :param indices: neighborhood mapping with shape (nxyz valid, nb)
    :return: neighborhood representation with shape (nxyz valid, nb * ncem)
    """
    return k_space.view(-1)[indices].view(matrix_shape)


def c_adjoint_operator(c_matrix: torch.Tensor, indices: torch.Tensor, k_space_dims: tuple, device=None):
    adj_c_matrix = torch.zeros(k_space_dims, dtype=c_matrix.dtype, device=c_matrix.device).view(-1)
    # Here is the crucial step - eliminate loops in the adjoint operation by using 1D linear indexing and
    # torch index_add
    adj_c_matrix = adj_c_matrix.index_add(0, indices.view(-1), c_matrix.view(-1))
    return adj_c_matrix.view(k_space_dims)


def s_operator(k_space: torch.Tensor, indices: torch.Tensor, matrix_shape: tuple):
    # test s mapping
    k_flipped = torch.flip(k_space, dims=(0, 1))
    s_p = k_space.view(-1)[indices].view(matrix_shape)
    s_m = k_flipped.view(-1)[indices].view(matrix_shape)

    return torch.cat(
        (
            torch.cat([(s_p - s_m).real, (-s_p + s_m).imag], dim=1),
            torch.cat([(s_p + s_m).imag, (s_p + s_m).real], dim=1)
        ), dim=0
    )


def s_operator_mem_opt(k_space: torch.Tensor, indices: torch.Tensor, matrix_shape: tuple):
    k_flip = torch.flip(k_space, dims=(0, 1))
    s_p = k_space.view(-1)[indices]
    s_m = k_flip.view(-1)[indices]
    result = torch.empty(2 * len(indices), 2, device=k_space.device)
    s_p_minus_s_m = (s_p - s_m)
    result[:len(indices), 0] = s_p_minus_s_m.real
    result[:len(indices), 1] = -s_p_minus_s_m.imag
    s_p_plus_s_m = (s_p + s_m)
    result[len(indices):, 0] = s_p_plus_s_m.imag
    result[len(indices):, 1] = s_p_plus_s_m.real

    shape = torch.tensor(matrix_shape) * torch.tensor([2, 2])
    return result.view(shape.tolist())


def s_adjoint_operator(s_matrix: torch.Tensor, indices: torch.Tensor, k_space_dims: tuple):
    # allocat
    dim = torch.prod(torch.tensor(k_space_dims))

    # split s matrix
    matrix_u, matrix_d = torch.tensor_split(s_matrix, 2, dim=0)
    srp_m_srm, msip_p_sim = torch.tensor_split(matrix_u, 2, dim=1)
    sip_p_sim, srp_p_srm = torch.tensor_split(matrix_d, 2, dim=1)

    # calculate srp, srm, etc.
    srp = srp_m_srm + srp_p_srm
    srm = -srp_m_srm + srp_p_srm
    sip = sip_p_sim - msip_p_sim
    sim = msip_p_sim + sip_p_sim

    dtype = torch.complex64 if s_matrix.dtype == torch.float32 else torch.complex128
    # crucial preventing loops: index add usage of 1d indexing
    adj_s_matrix_p = torch.zeros(dim, dtype=dtype, device=s_matrix.device).index_add(
        0, indices.view(-1), (srp + 1j * sip).view(-1)
    )
    adj_s_matrix_m = torch.zeros(dim, dtype=dtype, device=s_matrix.device).index_add(
        0, indices.view(-1), (srm + 1j * sim).view(-1)
    )

    # combine
    adj_s_matrix_p = adj_s_matrix_p.view(k_space_dims)
    adj_s_matrix_m = torch.flip(adj_s_matrix_m.view(k_space_dims), dims=(0, 1))
    return (adj_s_matrix_p + adj_s_matrix_m) / 2
