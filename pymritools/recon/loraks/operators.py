import torch
from typing import Tuple

from pymritools.recon.loraks.matrix_indexing import get_linear_indices
from pymritools.recon.loraks.loraks import OperatorType


def c_operator(k_space: torch.Tensor, indices: torch.Tensor, matrix_shape: tuple):
    """
    Maps from k-space with shape (nxyz, ncem) into the neighborhood-representation
    :param k_space: k-space with shape (nxyz, ncem)
    :param indices: neighborhood mapping with shape (nxyz valid, nb)
    :param matrix_shape: Shape of the neighborhood matrix
    :return: neighborhood representation with shape (nxyz valid, nb * ncem)
    """
    matrix = k_space.view(*k_space.shape[:-2], -1)[..., indices].view(*k_space.shape[:-2], *matrix_shape).mT
    # merge batch dims with neighborhood dim
    return torch.reshape(matrix, (-1, matrix.shape[-1]))


def c_adjoint_operator(c_matrix: torch.Tensor, indices: torch.Tensor, k_space_dims: tuple):
    # allocate the output
    adj_c_matrix = torch.zeros(k_space_dims, dtype=c_matrix.dtype, device=c_matrix.device).view(*k_space_dims[:-2], -1)
    # we need to unfold the neighborhood / channel dimension into neighborhood and channels and transpose to fit indices
    c_matrix = c_matrix.view(*k_space_dims[:-2], -1, c_matrix.shape[-1]).mT.contiguous()
    # Here is the crucial step - eliminate loops in the adjoint operation by using 1D linear indexing and
    # torch index_add
    adj_c_matrix = adj_c_matrix.index_add(-1, indices.view(-1), c_matrix.view(*k_space_dims[:-2], -1))
    return adj_c_matrix.view(k_space_dims)


def s_operator_matlab(k_space: torch.Tensor, indices: torch.Tensor, indices_rev: torch.Tensor, matrix_shape: tuple):
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
    #  but with mirrored indexing around the k-space center
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


def s_operator(k_space: torch.Tensor, indices: torch.Tensor, matrix_shape: tuple):
    if not k_space.is_complex():
        match k_space.dtype:
            case torch.float32:
                dtype = torch.complex64
            case torch.float64:
                dtype = torch.complex128
            case _:
                dtype = torch.complex64
    else:
        dtype = k_space.dtype
    # we merge the spatial dimensions and keep the "combination" dimension along to concatenate in front
    # (usually channels, echoes)
    k_space = k_space.view(*k_space.shape[:-2], -1)

    # we effectively build a c - matrix in each channel/ combination
    s_p = k_space[..., indices].view(-1, *matrix_shape).mT
    #  but with mirrored indexing around the k-space center
    indices_rev = indices.view(matrix_shape).flip(dims=(0,))
    s_m = k_space[..., indices_rev].view(-1, *matrix_shape).mT
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
    # to dims [2 * nc * ne * nb, ~ 2 * nxy]
    return s.view(-1, s.shape[-1])


def s_adjoint_operator_arxv(matrix: torch.Tensor, indices: torch.Tensor, indices_rev: torch.Tensor, k_space_dims: tuple):
    # the input matrix dimensions are [2 * nb * nc * ne,  2 * nxy]
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


def s_adjoint_operator_like_matlab(matrix: torch.Tensor, indices: torch.Tensor, indices_rev: torch.Tensor,
                                   k_space_dims: tuple):
    # the input matrix dimensions are [2 * nb * nc * ne,  2 * nxy]
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


def s_adjoint_operator(matrix: torch.Tensor, indices: torch.Tensor, k_space_dims: tuple):
    # the input matrix dimensions are [2 * nb * nc * ne,  2 * nxy]
    matrix_shape = matrix.shape
    # want to extract the channels and echoes first
    dim_combined = torch.tensor(k_space_dims[:-2]).prod().item()
    s = torch.reshape(matrix, (dim_combined, -1, matrix_shape[-1]))

    # get the quadrants
    s_u, s_d = torch.tensor_split(s, 2, dim=-1)

    s_ll, s_lr = torch.tensor_split(s_d, 2, dim=1)
    s_ul, s_ur = torch.tensor_split(s_u, 2, dim=1)

    spmm = s_ul - 1j * s_ur
    sppm = s_lr + 1j * s_ll

    s_p = (0.5 * (spmm + sppm)).mT.contiguous()
    s_m = (0.5 * (sppm - spmm)).mT.contiguous()

    adj_s_matrix = torch.zeros(k_space_dims, dtype=s_p.dtype, device=s_p.device).view(*k_space_dims[:-2], -1)
    # Here is the crucial step - eliminate loops in the adjoint operation by using 1D linear indexing and
    # torch index_add
    # get the point symmetric indices
    indices_rev = indices.view(s_m.shape[-2:]).flip(dims=(0,))

    adj_s_matrix = adj_s_matrix.index_add(-1, indices.view(-1), s_p.view(*s_p.shape[:-2], -1))
    adj_s_matrix = adj_s_matrix.index_add(-1, indices_rev.view(-1), s_m.view(*s_m.shape[:-2], -1))

    return adj_s_matrix.view(k_space_dims) / 2

def calculate_matrix_size(k_space_shape: Tuple,
                          patch_shape: Tuple,
                          sample_directions: Tuple,
                          matrix_type: OperatorType) -> tuple[int, int]:
    """
        Calculates the operator matrix size for a given k-space shape, patch shape, sample directions,
        and matrix type.

        Args:
        k_space_shape (Tuple): The shape of the k-space which impacts the number of sampling points.
        patch_shape (Tuple): The patch shape to extract from the k-space.
        sample_directions (Tuple): Directional sampling patterns for patches.
        matrix_type (OperatorType): Type of operator matrix

        Returns:
        tuple[int, int]: A tuple containing two integers representing the dimensions of the
        operator matrix based on the given inputs and the specified operator type.
    """
    k_space_shape = torch.tensor(k_space_shape)
    patch_shape = torch.tensor(patch_shape)
    sample_directions = torch.tensor(sample_directions)

    patch_sizes = torch.where(patch_shape == 0, torch.tensor(1), patch_shape)
    patch_sizes = torch.where(patch_shape == -1, k_space_shape, patch_sizes)

    sample_steps = torch.where(sample_directions == 0, torch.tensor(1), k_space_shape - patch_sizes + 1)
    if matrix_type == OperatorType.C:
        return torch.prod(patch_sizes).item(),  torch.prod(sample_steps).item()
    elif matrix_type == OperatorType.S:
        return 2 * torch.prod(patch_sizes).item(), 2 * torch.prod(sample_steps).item()
    else:
        raise ValueError(f"Unknown matrix type: {matrix_type}")


class Operator:
    def __init__(
            self, k_space_shape: Tuple, nb_side_length, device: torch.device,
            operator_type: OperatorType = OperatorType.S
    ):
        # For now we sample only the spatial directions that are involved (last 2 dims),
        # the rest is done via reshape / view
        self.operator_type: OperatorType = operator_type
        self.k_space_shape: tuple = k_space_shape
        self.patch_shape: tuple = (nb_side_length, nb_side_length)
        self.sample_directions: tuple = (1, 1)
        self.device: torch.device = device
        self.indices, self.matrix_shape = get_linear_indices(
            k_space_shape=k_space_shape[-2:], patch_shape=self.patch_shape, sample_directions=self.sample_directions
        )
        self.indices = self.indices.to(self.device)
        self._cm: torch.Tensor = None

    @property
    def count_matrix(self):
        if self._cm is None:
            in_ones = torch.ones(self.k_space_shape, dtype=torch.complex64, device=self.device)
            matrix = self.forward(in_ones)
            cm = torch.real(self.adjoint(matrix)).to(torch.int)
            cm[cm < 1] = 1
            self._cm = cm
        return self._cm

    def forward(self, k_space: torch.Tensor):
        match self.operator_type:
            case OperatorType.C:
                return c_operator(k_space=k_space, indices=self.indices, matrix_shape=self.matrix_shape)
            case OperatorType.S:
                return s_operator(k_space=k_space, indices=self.indices, matrix_shape=self.matrix_shape)
            case _:
                raise ValueError(f"Operator type {self.operator_type} not supported.")

    def adjoint(self, matrix: torch.Tensor):
        match self.operator_type:
            case OperatorType.C:
                return c_adjoint_operator(c_matrix=matrix, indices=self.indices, k_space_dims=self.k_space_shape)
            case OperatorType.S:
                return s_adjoint_operator(matrix=matrix, indices=self.indices, k_space_dims=self.k_space_shape)
            case _:
                raise ValueError(f"Operator type {self.operator_type} not supported.")

    @property
    def matrix_size(self) -> tuple[int, int]:
        return calculate_matrix_size(
            k_space_shape=self.k_space_shape, patch_shape=self.patch_shape,
            sample_directions=self.sample_directions, matrix_type=self.operator_type
        )

    @property
    def reversed_indices(self) -> torch.Tensor:
        # give the indices reversed in the spatial dimension (neighborhood remains same) and linearize again
        return self.indices.view(*self.matrix_shape).flip(dims=(0,)).view(-1)